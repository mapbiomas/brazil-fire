# -*- coding: utf-8 -*-
"""Legenda dinamica baseada nos parametros retornados pela API."""

import json
from urllib.parse import parse_qs, unquote, urlparse

from qgis.PyQt.QtCore import QTimer
from .compat import Qt  # Qt5/Qt6 enum compatibility proxy
from qgis.PyQt.QtGui import QColor, QPixmap
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QApplication
from qgis.gui import QgsLayerTreeEmbeddedWidgetProvider

from .mapbiomas_api import MapBiomasApiClient
from .parametros import base_url, upload_base_url


class LegendCheckBox(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_click_modifiers = Qt.NoModifier

    def mousePressEvent(self, event):
        self._last_click_modifiers = event.modifiers()
        super().mousePressEvent(event)

    def consume_last_click_modifiers(self):
        modifiers = self._last_click_modifiers
        self._last_click_modifiers = Qt.NoModifier
        return modifiers

    def nextCheckState(self):
        self.setCheckState(Qt.Unchecked if self.checkState() == Qt.Checked else Qt.Checked)


class DynamicApiLegendWidget(QWidget):
    def __init__(self, layer, legend_items, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.api_client = MapBiomasApiClient(base_url=base_url, upload_base_url=upload_base_url)
        self._row_checkboxes = {}
        self._row_pixels = {}
        self._row_parent = {}
        self._row_children = {}
        self._row_indent = {}
        self._updating_ui = False
        self._pending_context = None
        self._pending_selected_pixels = []
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(3000)
        self._debounce_timer.timeout.connect(self._apply_pending_update)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        context = self._read_layer_context()
        self._current_pixel_values = set(self._as_int_list(context.get("pixelValues", [])))

        rows = self._build_rows(legend_items)
        visible_rows = 0

        for row in rows:
            parent_id = row.get("parent_id")
            self._add_row(
                layout,
                row_id=row["id"],
                color=row["color"],
                label=row["label"],
                pixel_values=row["pixel_values"],
                indent=row.get("indent", 0),
                bold=row.get("bold", False),
                parent_id=parent_id,
            )
            visible_rows += 1

        for row_id in list(self._row_parent.keys()):
            self._recompute_parent_state(row_id)

        self.setLayout(layout)
        self.setMinimumHeight(max(20, 5 + visible_rows * 22))
        self.setMinimumWidth(240)

    @staticmethod
    def _as_int_list(values):
        output = []
        for value in values:
            try:
                output.append(int(value))
            except (TypeError, ValueError):
                continue
        return output

    def _read_layer_context(self):
        raw = self.layer.customProperty("firelegend/context_json", "{}")
        try:
            return json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _normalize_territory_ids(value):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else None
        if isinstance(value, (list, tuple, set)):
            ids = [str(item).strip() for item in value if str(item).strip()]
            return ids or None
        normalized = str(value).strip()
        return [normalized] if normalized else None

    def _territory_ids_from_layer_source(self):
        source = str(self.layer.source() or "")
        if not source:
            return None

        source_params = parse_qs(source, keep_blank_values=False)
        raw_url_list = source_params.get("url")
        if not raw_url_list:
            return None

        decoded_url = unquote(raw_url_list[0])
        query = parse_qs(urlparse(decoded_url).query, keep_blank_values=False)
        values = query.get("territoryId") or query.get("territoryIds")
        return self._normalize_territory_ids(values)

    def _get_context_territory_ids(self, context):
        direct = self._normalize_territory_ids(context.get("territoryIds"))
        if direct:
            return direct

        legacy = self._normalize_territory_ids(context.get("territoryId"))
        if legacy:
            context["territoryIds"] = legacy
            return legacy

        from_source = self._territory_ids_from_layer_source()
        if from_source:
            context["territoryIds"] = from_source
            return from_source

        return None

    def _build_rows(self, legend_items):
        rows = []

        # Quando a API traz hierarquia em children, aplica recuo pai/filho.
        has_children = any(
            isinstance(item.get("children"), list) and item.get("children")
            for item in legend_items
        )

        def _item_id(item):
            return item.get("id") or item.get("key") or item.get("pixelValue")

        def _row_id(item):
            return f"item:{_item_id(item)}"

        def _append_row(item, indent=0, bold=False, forced_color=None, forced_pixels=None, parent_id=None):
            pixel_values = forced_pixels
            if pixel_values is None:
                pixel_value = item.get("pixelValue")
                if pixel_value is None:
                    return
                pixel_values = self._as_int_list([pixel_value])

            if not pixel_values:
                return

            rows.append(
                {
                    "id": _row_id(item),
                    "label": self._get_label(item.get("name")),
                    "color": forced_color or item.get("color") or "#808080",
                    "pixel_values": pixel_values,
                    "bold": bold,
                    "indent": indent,
                    "parent_id": parent_id,
                }
            )

        if has_children:
            child_ids = {
                _item_id(child)
                for item in legend_items
                for child in item.get("children", [])
                if _item_id(child) is not None
            }

            root_items = [item for item in legend_items if _item_id(item) not in child_ids]

            for item in root_items:
                parent_label = self._get_label(item.get("name"))
                parent_color = item.get("color") or "#808080"

                children = item.get("children", [])
                child_pixels = self._as_int_list(
                    [child.get("pixelValue") for child in children if child.get("pixelValue") is not None]
                )

                parent_pixels = []
                if item.get("pixelValue") is not None:
                    parent_pixels.append(item.get("pixelValue"))
                parent_pixels.extend(child_pixels)
                parent_pixels = self._as_int_list(parent_pixels)

                if parent_pixels:
                    _append_row(
                        item,
                        indent=0,
                        bold=True,
                        forced_color=parent_color,
                        forced_pixels=parent_pixels,
                        parent_id=None,
                    )

                for child in children:
                    _append_row(
                        child,
                        indent=1,
                        bold=False,
                        forced_color=child.get("color") or parent_color,
                        parent_id=_row_id(item),
                    )

            return rows

        # Fallback: se vier plano, tenta inferir pai/filho pela key (ex.: natural / natural_xxx).
        id_to_item = {_item_id(item): item for item in legend_items if _item_id(item) is not None}
        key_to_item = {
            str(item.get("key") or "").strip().lower(): item
            for item in legend_items
            if str(item.get("key") or "").strip()
        }

        children_by_parent = {}
        for item in legend_items:
            key = str(item.get("key") or "").strip().lower()
            if not key or "_" not in key:
                continue
            parent_key = key.split("_", 1)[0]
            if parent_key in key_to_item:
                children_by_parent.setdefault(parent_key, []).append(item)

        if children_by_parent:
            emitted_ids = set()
            for parent_key, children in children_by_parent.items():
                parent = key_to_item.get(parent_key)
                if parent is None:
                    continue

                children = sorted(children, key=lambda child: int(child.get("order") or 0))
                parent_pixels = []
                if parent.get("pixelValue") is not None:
                    parent_pixels.append(parent.get("pixelValue"))
                parent_pixels.extend([child.get("pixelValue") for child in children if child.get("pixelValue") is not None])
                parent_pixels = self._as_int_list(parent_pixels)

                if _item_id(parent) not in emitted_ids:
                    _append_row(parent, indent=0, bold=True, forced_pixels=parent_pixels, parent_id=None)
                    emitted_ids.add(_item_id(parent))

                for child in children:
                    child_id = _item_id(child)
                    if child_id in emitted_ids:
                        continue
                    _append_row(child, indent=1, parent_id=_row_id(parent))
                    emitted_ids.add(child_id)

            for item in legend_items:
                item_id = _item_id(item)
                if item_id in emitted_ids:
                    continue
                _append_row(item, indent=0)
            return rows

        if self._should_group_families(legend_items):
            grouped_items = self._group_legend_items(legend_items)
            for group_key in self._group_order(grouped_items):
                group = grouped_items[group_key]
                group_pixels = self._as_int_list(
                    [item.get("pixelValue") for item in group.get("children", []) if item.get("pixelValue") is not None]
                )
                if not group_pixels:
                    continue
                rows.append(
                    {
                        "id": f"group:{group_key}",
                        "label": self._group_label(group_key),
                        "color": self._group_color(group_key),
                        "pixel_values": group_pixels,
                        "bold": True,
                        "indent": 0,
                    }
                )
            return rows

        for item in legend_items:
            _append_row(item, indent=0, parent_id=None)
        return rows

    @staticmethod
    def _normalize_color(color):
        return str(color or "").strip().lower()

    @staticmethod
    def _group_color(group_key):
        colors = {
            "natural": "#66bb6a",
            "anthropic": "#ffe082",
            "not_defined": "#d5d5e5",
        }
        return colors.get(group_key, "#808080")

    @staticmethod
    def _group_label(group_key):
        labels = {
            "natural": "Natural",
            "anthropic": "Antropico",
            "not_defined": "Não definido",
        }
        return labels.get(group_key, "Outros")

    def _infer_group_key(self, item):
        key = str(item.get("key", "")).lower()
        color = self._normalize_color(item.get("color"))

        if key.startswith("natural") or color == "#66bb6a":
            return "natural"
        if key.startswith("anthropic") or color == "#ffe082":
            return "anthropic"
        if key.startswith("not_defined") or color == "#d5d5e5":
            return "not_defined"
        return None

    def _should_group_families(self, legend_items):
        return any(self._infer_group_key(item) in {"natural", "anthropic", "not_defined"} for item in legend_items)

    def _group_legend_items(self, legend_items):
        groups = {
            "natural": {"header": None, "children": []},
            "anthropic": {"header": None, "children": []},
            "not_defined": {"header": None, "children": []},
            "other": {"header": None, "children": []},
        }

        for item in legend_items:
            group_key = self._infer_group_key(item) or "other"
            group = groups.setdefault(group_key, {"header": None, "children": []})
            group["children"].append(item)

            if group_key in {"natural", "anthropic", "not_defined"} and str(item.get("key", "")).lower() == group_key:
                group["header"] = item

        for group_key in ("natural", "anthropic", "not_defined", "other"):
            group = groups.get(group_key)
            if not group or group["header"] is not None or not group["children"]:
                continue
            group["header"] = group["children"][0]

        return groups

    @staticmethod
    def _group_order(groups):
        ordered = ["natural", "anthropic", "not_defined", "other"]
        return [group_key for group_key in ordered if group_key in groups and groups[group_key]["children"]]

    def _add_row(self, layout, row_id, color, label, pixel_values, indent=0, bold=False, parent_id=None):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        if indent > 0:
            # Em alguns contextos do QGIS, margem de layout pode nao aparecer;
            # o spacer explicito garante recuo visual dos subitens.
            row.addSpacing(20 * int(indent))

        pix = QPixmap(18, 18)
        pix.fill(QColor(color))

        lbl_color = QLabel()
        lbl_color.setPixmap(pix)

        text_indent = "   " * max(0, int(indent))
        checkbox = LegendCheckBox(f"{text_indent}{label}")
        checkbox.setTristate(True)

        is_checked = all(pixel in self._current_pixel_values for pixel in pixel_values)
        is_unchecked = all(pixel not in self._current_pixel_values for pixel in pixel_values)
        if is_checked:
            checkbox.setCheckState(Qt.Checked)
        elif is_unchecked:
            checkbox.setCheckState(Qt.Unchecked)
        else:
            checkbox.setCheckState(Qt.PartiallyChecked)

        checkbox.stateChanged.connect(lambda _state, rid=row_id: self._on_row_state_changed(rid))

        self._row_checkboxes[row_id] = checkbox
        self._row_pixels[row_id] = list(pixel_values)
        self._row_parent[row_id] = parent_id
        self._row_indent[row_id] = int(indent)
        self._row_children.setdefault(row_id, [])
        if parent_id:
            self._row_children.setdefault(parent_id, []).append(row_id)

        if bold:
            font = checkbox.font()
            font.setBold(True)
            checkbox.setFont(font)

        row.addWidget(lbl_color)
        row.addWidget(checkbox)
        row.addStretch()
        layout.addLayout(row)

    def _is_row_checked(self, row_id):
        checkbox = self._row_checkboxes.get(row_id)
        if checkbox is None:
            return False
        return checkbox.checkState() == Qt.Checked

    def _is_row_partial(self, row_id):
        checkbox = self._row_checkboxes.get(row_id)
        if checkbox is None:
            return False
        return checkbox.checkState() == Qt.PartiallyChecked

    def _set_row_state(self, row_id, state):
        checkbox = self._row_checkboxes.get(row_id)
        if checkbox is None:
            return
        checkbox.setCheckState(state)

    def _descendants_of(self, row_id):
        descendants = []
        stack = list(self._row_children.get(row_id, []))
        while stack:
            current = stack.pop(0)
            descendants.append(current)
            stack[0:0] = list(self._row_children.get(current, []))
        return descendants

    def _recompute_parent_state(self, row_id):
        parent_id = self._row_parent.get(row_id)
        if not parent_id:
            return

        child_ids = self._row_children.get(parent_id, [])
        if not child_ids:
            return

        checked_count = 0
        partial_count = 0
        for child_id in child_ids:
            state = self._row_checkboxes[child_id].checkState()
            if state == Qt.Checked:
                checked_count += 1
            elif state == Qt.PartiallyChecked:
                partial_count += 1

        if checked_count == len(child_ids):
            parent_state = Qt.Checked
        elif checked_count == 0 and partial_count == 0:
            parent_state = Qt.Unchecked
        else:
            parent_state = Qt.PartiallyChecked

        self._updating_ui = True
        try:
            self._set_row_state(parent_id, parent_state)
        finally:
            self._updating_ui = False

        self._recompute_parent_state(parent_id)

    def _collect_selected_pixels(self):
        selected = set()
        for row_id, checkbox in self._row_checkboxes.items():
            if checkbox.checkState() != Qt.Checked:
                continue
            selected.update(self._row_pixels.get(row_id, []))
        return sorted(selected)

    def _persist_context_and_schedule_refresh(self):
        selected_pixels = self._collect_selected_pixels()
        context = self._read_layer_context()
        context["territoryIds"] = self._get_context_territory_ids(context)
        context["pixelValues"] = selected_pixels
        self.layer.setCustomProperty("firelegend/context_json", json.dumps(context, ensure_ascii=False))

        if not selected_pixels:
            if self._debounce_timer.isActive():
                self._debounce_timer.stop()
            self.layer.setOpacity(0.0)
            self.layer.triggerRepaint()
            return

        self.layer.setOpacity(1.0)
        self._pending_context = context
        self._pending_selected_pixels = list(selected_pixels)
        self._debounce_timer.start()

    def _on_row_state_changed(self, row_id):
        if self._updating_ui:
            return

        checkbox = self._row_checkboxes.get(row_id)
        if checkbox is None:
            return

        modifiers = checkbox.consume_last_click_modifiers()
        if modifiers & Qt.ControlModifier:
            self._updating_ui = True
            try:
                for current_row_id in self._row_checkboxes.keys():
                    self._set_row_state(current_row_id, Qt.Checked if current_row_id == row_id else Qt.Unchecked)
            finally:
                self._updating_ui = False

            for current_row_id in list(self._row_parent.keys()):
                self._recompute_parent_state(current_row_id)

            self._persist_context_and_schedule_refresh()
            return

        if modifiers & Qt.ShiftModifier:
            self._updating_ui = True
            try:
                for current_row_id in self._row_checkboxes.keys():
                    self._set_row_state(current_row_id, Qt.Unchecked if current_row_id == row_id else Qt.Checked)
            finally:
                self._updating_ui = False

            for current_row_id in list(self._row_parent.keys()):
                self._recompute_parent_state(current_row_id)

            self._persist_context_and_schedule_refresh()
            return

        state = checkbox.checkState()
        descendants = self._descendants_of(row_id)
        is_parent = bool(self._row_children.get(row_id))

        self._updating_ui = True
        try:
            if is_parent and state != Qt.PartiallyChecked:
                for child_id in descendants:
                    self._set_row_state(child_id, state)
        finally:
            self._updating_ui = False

        self._recompute_parent_state(row_id)
        self._persist_context_and_schedule_refresh()

    def _apply_pending_update(self):
        context = self._pending_context or self._read_layer_context()
        territory_ids = self._get_context_territory_ids(context)
        selected_pixels = self._pending_selected_pixels or self._as_int_list(context.get("pixelValues", []))

        if not selected_pixels:
            self.layer.setOpacity(0.0)
            self.layer.triggerRepaint()
            return

        try:
            map_url = self.api_client.get_map_url(
                region=context.get("region", "brazil"),
                subthemeKey=context.get("subthemeKey"),
                legendKey=context.get("legendKey"),
                pixelValue=selected_pixels,
                year=context.get("year"),
                territoryId=territory_ids,
            )
        except Exception as exc:
            print(f"Falha ao atualizar legenda dinamica: {exc}")
            return

        if not map_url:
            return

        xyz_uri = f"type=xyz&url={map_url}"
        self.layer.setDataSource(xyz_uri, self.layer.name(), "wms")
        self.layer.triggerRepaint()

    @staticmethod
    def _get_label(name):
        if isinstance(name, dict):
            return name.get("pt-BR") or name.get("en-US") or next(iter(name.values()), "Sem legenda")
        if isinstance(name, str) and name:
            return name
        return "Sem legenda"


class DynamicApiLegendProvider(QgsLayerTreeEmbeddedWidgetProvider):
    def id(self):
        return "dynamicapilegend_v8"

    def name(self):
        return "Legenda dinamica da API MapBiomas"

    def createWidget(self, layer, widgetIndex):
        items_json = layer.customProperty("firelegend/items_json", "[]")
        try:
            legend_items = json.loads(items_json)
        except (TypeError, ValueError):
            legend_items = []
        return DynamicApiLegendWidget(layer, legend_items)
