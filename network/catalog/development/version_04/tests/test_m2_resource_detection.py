"""Resource limits used by the M2 UI scheduler."""
import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core')))
import M2_mosaic_ui as ui


@contextmanager
def resource_files(files):
    """Simulate both reads and existence checks using the same file map."""
    def stat(path):
        if files.get(path) is None:
            raise FileNotFoundError(path)
        return Mock()

    with patch.object(ui, '_read_resource_file', side_effect=lambda p: files.get(p)), \
         patch.object(ui.os, 'stat', side_effect=stat):
        yield


class ResourceDetectionTests(unittest.TestCase):
    def test_cgroup_v2_real_root_without_controller_files(self):
        files = {
            '/proc/self/cgroup': '0::/parent/job',
            '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/parent/job/cpu.max': '400000 100000',
            '/sys/fs/cgroup/parent/cpu.max': '200000 100000',
            '/sys/fs/cgroup/parent/job/memory.max': '8589934592',
            '/sys/fs/cgroup/parent/job/memory.current': '3221225472',
            '/sys/fs/cgroup/parent/memory.max': 'max',
        }
        with resource_files(files):
            self.assertEqual(ui._cgroup_limits(), (2, 8*1024**3, 5*1024**3))

    def test_cgroup_v2_subtree_mount_root_missing_is_unknown(self):
        files = {
            '/proc/self/cgroup': '0::/parent/job',
            '/proc/self/mountinfo': '30 20 0:27 /parent /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/job/cpu.max': '400000 100000',
            '/sys/fs/cgroup/job/memory.max': 'max',
        }
        with resource_files(files):
            cpu, memory, available = ui._cgroup_limits()
        self.assertEqual(cpu, 1)
        self.assertIs(memory, ui._CGROUP_UNKNOWN)
        self.assertIsNone(available)

    def test_cgroup_v2_root_existing_limits_are_not_ignored(self):
        files = {
            '/proc/self/cgroup': '0::/job',
            '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/job/cpu.max': '400000 100000',
            '/sys/fs/cgroup/cpu.max': '200000 100000',
            '/sys/fs/cgroup/job/memory.max': 'max',
            '/sys/fs/cgroup/memory.max': '8589934592',
            '/sys/fs/cgroup/memory.current': '3221225472',
        }
        with resource_files(files):
            self.assertEqual(ui._cgroup_limits(), (2, 8*1024**3, 5*1024**3))

    def test_cgroup_v2_unreadable_root_is_not_assumed_absent(self):
        files = {
            '/proc/self/cgroup': '0::/job',
            '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/job/cpu.max': '400000 100000',
            '/sys/fs/cgroup/job/memory.max': 'max',
        }
        with patch.object(ui, '_read_resource_file', side_effect=lambda p: files.get(p)), \
             patch.object(ui.os, 'stat', side_effect=PermissionError):
            cpu, memory, available = ui._cgroup_limits()
        self.assertEqual(cpu, 1)
        self.assertIs(memory, ui._CGROUP_UNKNOWN)
        self.assertIsNone(available)

    def test_cgroup_v2_nested_limits_and_memory(self):
        files = {
            '/proc/self/cgroup': '0::/parent/job\n',
            '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
            '/sys/fs/cgroup/parent/job/cpu.max': '350000 100000',
            '/sys/fs/cgroup/parent/cpu.max': '200000 100000',
            '/sys/fs/cgroup/cpu.max': 'max 100000',
            '/sys/fs/cgroup/parent/job/memory.max': '8589934592',
            '/sys/fs/cgroup/parent/job/memory.current': '3221225472',
            '/sys/fs/cgroup/parent/memory.max': 'max',
            '/sys/fs/cgroup/memory.max': 'max',
        }
        with resource_files(files):
            cpu, total, available = ui._cgroup_limits()
        self.assertEqual((cpu, total, available), (2, 8 * 1024**3, 5 * 1024**3))

    def test_cgroup_v1_fractional_quota_rounds_down(self):
        files = {
            '/proc/self/cgroup': '4:cpu,cpuacct:/group\n',
            '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n',
            '/sys/fs/cgroup/cpu/group/cpu.cfs_quota_us': '150000',
            '/sys/fs/cgroup/cpu/group/cpu.cfs_period_us': '100000',
        }
        with resource_files(files):
            self.assertEqual(ui._cgroup_limits()[0], 1)

    def test_affinity_works_without_psutil(self):
        with patch.object(ui.os, 'cpu_count', return_value=16), \
             patch.object(ui.os, 'process_cpu_count', return_value=16, create=True), \
             patch.object(ui.os, 'sched_getaffinity', return_value={0, 1, 2}, create=True), \
             patch.object(ui, '_cgroup_limits', return_value=(2, None, None)), \
             patch.dict(sys.modules, {'psutil': None}):
            self.assertEqual(ui.detect_m2_resources()[0], 2)

    def test_missing_parent_memory_usage_forces_unknown_available(self):
        files = {
            '/proc/self/cgroup': '0::/parent/job',
            '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/parent/job/memory.max': '8589934592',
            '/sys/fs/cgroup/parent/job/memory.current': '3221225472',
            '/sys/fs/cgroup/parent/memory.max': '17179869184',
            '/sys/fs/cgroup/memory.max': 'max',
        }
        with resource_files(files):
            self.assertEqual(ui._cgroup_limits()[1:], (8 * 1024**3, None))

    def test_joint_thread_budget_and_conservative_cap(self):
        gib = 1024**3
        for cpus in range(1, 65):
            workers = ui.choose_m2_workers(10, cpus, 64*gib, 48*gib, 200*gib)
            threads = ui.choose_gdal_threads(cpus, workers, 48*gib)
            self.assertLessEqual(workers * threads, cpus)
            self.assertLessEqual(threads, 8)

    def test_unknown_cpu_never_enables_parallelism(self):
        gib = 1024**3
        self.assertEqual(ui.choose_m2_workers(5, None, 40*gib, 30*gib, 150*gib), 1)
        self.assertEqual(ui.choose_gdal_threads(None, 1, 30*gib), 1)

    def test_missing_or_malformed_cgroup_limit_is_conservative(self):
        for value in (None, 'broken'):
            files = {
                '/proc/self/cgroup': '0::/job',
                '/proc/self/mountinfo': '30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
                '/sys/fs/cgroup/job/cpu.max': value,
                '/sys/fs/cgroup/cpu.max': 'max 100000',
                '/sys/fs/cgroup/job/memory.max': value,
                '/sys/fs/cgroup/memory.max': 'max',
            }
            with self.subTest(value=value), \
                 resource_files(files):
                cpu, memory, available = ui._cgroup_limits()
                self.assertEqual(cpu, 1)
                self.assertIs(memory, ui._CGROUP_UNKNOWN)
                self.assertIsNone(available)

    def test_unknown_memory_cgroup_disables_host_memory_parallelism(self):
        gib = 1024**3
        memory = SimpleNamespace(total=64*gib, available=48*gib)
        psutil = SimpleNamespace(virtual_memory=lambda: memory,
                                 Process=lambda: SimpleNamespace(),
                                 Error=Exception)
        with patch.object(ui, '_cgroup_limits', return_value=(1, ui._CGROUP_UNKNOWN, None)), \
             patch.dict(sys.modules, {'psutil': psutil}):
            cpu, total, available, _ = ui.detect_m2_resources()
        self.assertEqual((cpu, total, available), (1, None, None))

    def test_no_cgroup_files_on_windows_keep_normal_fallback(self):
        with patch.object(ui.sys, 'platform', 'win32'), \
             patch.object(ui, '_read_resource_file', return_value=None):
            self.assertEqual(ui._cgroup_limits(), (None, None, None))

    def test_mountinfo_escaped_mount_and_root_are_decoded(self):
        files = {
            '/proc/self/cgroup': '0::/parent dir/job',
            '/proc/self/mountinfo': r'30 20 0:27 /parent\040dir /sys/fs/cgroup/my\040mount rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/my mount/job/cpu.max': '200000 100000',
            '/sys/fs/cgroup/my mount/cpu.max': 'max 100000',
            '/sys/fs/cgroup/my mount/job/memory.max': 'max',
            '/sys/fs/cgroup/my mount/memory.max': 'max',
        }
        with resource_files(files):
            self.assertEqual(ui._cgroup_limits()[0], 2)

    def test_cgroup_headroom_is_not_host_available_memory(self):
        with patch.object(ui.os, 'process_cpu_count', return_value=8, create=True), \
             patch.object(ui, '_cgroup_limits', return_value=(4, 16*1024**3, 12*1024**3)), \
             patch.dict(sys.modules, {'psutil': None}):
            _, _, available, _ = ui.detect_m2_resources()
        self.assertIsNone(available)

    def test_selected_budget_is_passed_to_each_gdal_product(self):
        gib = 1024**3
        metas = [
            {'sensor': 'sentinel2', 'year': 2025, 'month': 1,
             'period_key': '2025_01', 'band': band, 'mosaic': 'minnbr'}
            for band in ('red', 'blue')
        ]
        obj = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=meta)
                                       for i, meta in enumerate(metas)},
                              update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        with patch.object(ui, 'detect_m2_resources', return_value=(4, 40*gib, 30*gib, 150*gib)), \
             patch.object(ui, 'process_country_mosaic_item', return_value=['done']) as process, \
             patch.object(ui, 'cell_log'), \
             patch.object(ui.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui.start_mosaic_assembly(obj)
        self.assertEqual(process.call_count, 2)
        self.assertTrue(all(call.kwargs['gdal_threads'] == 2 for call in process.call_args_list))


if __name__ == '__main__':
    unittest.main()
