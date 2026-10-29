import importlib.util
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ("ee", "requests", "PIL", "yaml")), "dependências do Earth Engine indisponíveis")
class DownloaderProgressTests(unittest.TestCase):
    def test_territory_manager_uses_in_list_for_multiple_values(self):
        from unittest.mock import MagicMock
        from mapbiomas_data.core.territory_manager import TerritoryManager
        class Config:
            territories = {"biomes": {"selecionados": {"name": "Biomas", "source": "asset", "filter_in": {"field": "Bioma", "values": ["Cerrado", "Caatinga"]}}}}
        ee = MagicMock()
        with patch.dict("sys.modules", {"ee": ee}):
            TerritoryManager(Config()).get_feature_collection("selecionados")
        ee.Filter.inList.assert_called_once_with("Bioma", ["Cerrado", "Caatinga"])
        ee.FeatureCollection.return_value.filter.assert_called_once_with(ee.Filter.inList.return_value)

    def test_retry_and_callback_failure_do_not_repeat_successful_download(self):
        from mapbiomas_data.core.ee_downloader import EEDownloader
        class Config:
            def get_processing_config(self, key):
                return {}
        class Response:
            def raise_for_status(self):
                pass
            def iter_content(self, chunk_size):
                return [b"png"]
        downloader = EEDownloader(Config())
        events = []
        def callback(kind, *args):
            events.append((kind, args))
            if kind == "frame_done":
                raise RuntimeError("widget indisponível")
        downloader.progress_callback = callback
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(downloader, "_get_thumb_url", return_value="url"), \
             patch("mapbiomas_data.core.ee_downloader.requests.get", side_effect=[OSError("rede"), Response()]) as request, \
             patch("mapbiomas_data.core.ee_downloader.time.sleep"):
            files = downloader.download_frames(None, ["ano_2023"], {}, None, tmp)
        self.assertEqual(len(files), 1)
        self.assertEqual(request.call_count, 2)
        self.assertEqual([event[0] for event in events], ["frame_start", "frame_retry", "frame_done"])

    def test_failure_then_next_frame_keeps_actual_index(self):
        from mapbiomas_data.core.ee_downloader import EEDownloader
        class Config:
            def get_processing_config(self, key):
                return {}
        class Response:
            def raise_for_status(self):
                pass
            def iter_content(self, chunk_size):
                return [b"png"]
        downloader = EEDownloader(Config())
        events = []
        downloader.progress_callback = lambda kind, *args: events.append((kind, args))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(downloader, "_get_thumb_url", return_value="url"), \
             patch("mapbiomas_data.core.ee_downloader.requests.get", side_effect=[OSError("rede")] * 3 + [Response()]), \
             patch("mapbiomas_data.core.ee_downloader.time.sleep"):
            downloader.download_frames(None, ["ano_2022", "ano_2023"], {}, None, tmp)
        starts = [args[0] for kind, args in events if kind == "frame_start"]
        self.assertEqual(starts, [1, 2])
        self.assertTrue(any(kind == "frame_failed" for kind, _ in events))
