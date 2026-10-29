import math
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image as PILImage, ImageDraw, ImageFont


def _parse_color(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _find_font() -> Optional[str]:
    candidates = [
        "C:\\Windows\\Fonts\\Arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\Calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    import subprocess, sys
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans"],
            capture_output=True, text=True, timeout=5,
        )
        fp = result.stdout.strip()
        if fp and os.path.exists(fp):
            return fp
    except Exception:
        pass

    return None


def _ensure_rgb(image: PILImage.Image) -> PILImage.Image:
    if image.mode == "RGBA":
        bg = PILImage.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _render_rgb_legend(draw: ImageDraw.Draw, width: int, y: int, margin: int,
                       entries: List[Dict[str, str]], vmin: float, vmax: float,
                       font_size: int = 42):
    box_size = 28
    font_title = FrameProcessor._make_font(font_size)
    font_detail = FrameProcessor._make_font(font_size - 8)

    line1_h = draw.textbbox((0, 0), "Ag", font=font_title)[3] + 8
    line2_h = draw.textbbox((0, 0), "Ag", font=font_detail)[3] + 8

    draw.text((margin, y), "Composição RGB:", font=font_title, fill=(0, 0, 0))

    y2 = y + line1_h
    detail_prefix = f"  min: {int(vmin)}, max: {int(vmax)}, bandas:"
    prefix_w = draw.textbbox((0, 0), detail_prefix, font=font_detail)[2]

    draw.text((margin, y2), "  " + detail_prefix.lstrip(), font=font_detail, fill=(0, 0, 0))

    bx = margin + prefix_w + 16 + 20
    for entry in entries:
        color = _parse_color(entry["color"])
        lbl = entry["label"]
        lbl_w = draw.textbbox((0, 0), lbl, font=font_detail)[2]

        draw.rectangle([bx, y2, bx + box_size, y2 + box_size], fill=color)
        draw.rectangle([bx, y2, bx + box_size, y2 + box_size], outline=(100, 100, 100), width=2)
        bx += box_size + 6
        draw.text((bx, y2), lbl, font=font_detail, fill=(0, 0, 0))
        bx += lbl_w + 30


class FrameProcessor:
    """Processar frames individuais (labels, redimensionamento, etc)."""

    FONT_PATH = _find_font()

    @staticmethod
    def _rgb_legend_height(draw: ImageDraw.Draw, font_size: int) -> int:
        font_title = FrameProcessor._make_font(font_size)
        font_detail = FrameProcessor._make_font(font_size - 8)
        line1_h = draw.textbbox((0, 0), "Ag", font=font_title)[3] + 8
        line2_h = draw.textbbox((0, 0), "Ag", font=font_detail)[3] + 8
        return line1_h + line2_h + 12

    @staticmethod
    def add_year_label(
        image_path: str,
        text: str,
        position: str = "top_left",
        font_size: int = 120,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        outline_color: Tuple[int, int, int] = (0, 0, 0),
        padding_top: int = 0,
        bar_color: Optional[Tuple[int, int, int]] = None,
        max_width: Optional[int] = None,
        subtitle: Optional[str] = None,
        subtitle_size: Optional[int] = None,
        line2: Optional[str] = None,
        line2_size: Optional[int] = None,
        draw_north: bool = False,
    ) -> None:
        image = _ensure_rgb(PILImage.open(image_path))

        font1 = FrameProcessor._make_font(font_size)
        font2 = FrameProcessor._make_font(subtitle_size or font_size - 10) if subtitle else None
        font3 = FrameProcessor._make_font(line2_size or subtitle_size or font_size - 10) if line2 else None

        if padding_top > 0:
            bg = bar_color if bar_color else (0, 0, 0)
            padded = PILImage.new("RGB", (image.width, image.height + padding_top), bg)
            padded.paste(image, (0, padding_top))
            image = padded

        draw = ImageDraw.Draw(image)

        if max_width is None:
            max_width = image.width - 40

        positions = {
            "top_left": (20, 20),
            "top_right": (image.width - 200, 20),
            "bottom_left": (20, image.height - font_size - 20),
            "bottom_right": (image.width - 200, image.height - font_size - 20),
        }
        pos = positions.get(position, (20, 20))

        l1_lines = FrameProcessor._wrap_text(draw, text, font1, max_width)
        l1_h = draw.textbbox((0, 0), "Ay", font=font1)[3] + 4
        l2_lines = FrameProcessor._wrap_text(draw, subtitle, font2, max_width) if subtitle and font2 else []
        l2_h = draw.textbbox((0, 0), "Ay", font=font2)[3] + 4 if l2_lines else 0
        l3_lines = FrameProcessor._wrap_text(draw, line2, font3, max_width) if line2 and font3 else []
        l3_h = draw.textbbox((0, 0), "Ay", font=font3)[3] + 4 if l3_lines else 0

        gap_between = 8
        total_lines = l1_lines + l2_lines + l3_lines
        line_heights = [l1_h] * len(l1_lines) + [l2_h] * len(l2_lines) + [l3_h] * len(l3_lines)
        total_w = max(draw.textbbox((0, 0), l, font=font1)[2] for l in l1_lines + [""]) if l1_lines else 0
        if l2_lines:
            total_w = max(total_w, max(draw.textbbox((0, 0), l, font=font2)[2] for l in l2_lines))
        if l3_lines:
            total_w = max(total_w, max(draw.textbbox((0, 0), l, font=font3)[2] for l in l3_lines))
        total_h = len(l1_lines) * l1_h + (gap_between if l2_lines else 0) + len(l2_lines) * l2_h
        total_h += (gap_between if l3_lines else 0) + len(l3_lines) * l3_h

        if bar_color is None:
            bar_padding = 12
            draw.rectangle(
                [pos[0] - bar_padding, pos[1] - bar_padding,
                 pos[0] + total_w + bar_padding, pos[1] + total_h + bar_padding],
                fill=(40, 40, 40),
            )

            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cy = pos[1]
                for l in l1_lines:
                    draw.text((pos[0] + dx, cy + dy), l, font=font1, fill=outline_color)
                    cy += l1_h
                cy += gap_between
                for l in l2_lines:
                    draw.text((pos[0] + dx, cy + dy), l, font=font2, fill=outline_color)
                    cy += l2_h
                cy += gap_between
                for l in l3_lines:
                    draw.text((pos[0] + dx, cy + dy), l, font=font3, fill=outline_color)
                    cy += l3_h

        cy = pos[1]
        for l in l1_lines:
            draw.text((pos[0], cy), l, font=font1, fill=text_color)
            cy += l1_h
        cy += gap_between
        for l in l2_lines:
            draw.text((pos[0], cy), l, font=font2, fill=text_color)
            cy += l2_h
        cy += gap_between
        for l in l3_lines:
            draw.text((pos[0], cy), l, font=font3, fill=text_color)
            cy += l3_h

        if draw_north:
            north_font = FrameProcessor._make_font(max(14, int(font_size * 0.4)))
            nb = draw.textbbox((0, 0), "N", font=north_font)
            nw = nb[2] - nb[0]
            arrow_size = max(12, int(font_size * 0.12))
            nc = image.width - 20 - max(nw, arrow_size) // 2
            ny = pos[1] + l1_h // 2
            draw.polygon(
                [(nc, ny - arrow_size), (nc - 7, ny - 2), (nc + 7, ny - 2)],
                fill=text_color,
            )
            draw.line([(nc, ny - 2), (nc, ny + 4)], fill=text_color, width=2)
            draw.line([(nc - 6, ny + 4), (nc + 6, ny + 4)], fill=text_color, width=2)
            draw.text((nc - nw // 2, ny + 8), "N", font=north_font, fill=text_color)

        image.save(image_path)

    @staticmethod
    def add_year_only(image_path: str, year_text: str, font_size: int = 160, draw_north: bool = False) -> None:
        padding = max(130, font_size + 20)
        FrameProcessor.add_year_label(
            image_path, year_text,
            position="top_left",
            font_size=font_size,
            padding_top=padding,
            bar_color=(255, 255, 255),
            text_color=(0, 0, 0),
            draw_north=draw_north,
        )

    @staticmethod
    def batch_add_year_only(image_paths: Dict[str, str]) -> None:
        for path, label in image_paths.items():
            try:
                FrameProcessor.add_year_only(path, label)
            except Exception as e:
                print(f"Erro ao adicionar ano em {path}: {e}")

    @staticmethod
    def batch_add_labels(
        image_paths: Dict[str, str],
        position: str = "top_left",
        font_size: int = 48,
        padding_top: int = 0,
        **kwargs,
    ) -> None:
        for path, label in image_paths.items():
            try:
                FrameProcessor.add_year_label(path, label, position, font_size=font_size, padding_top=padding_top, **kwargs)
            except Exception as e:
                print(f"Erro ao adicionar label em {path}: {e}")

    @staticmethod
    def add_frame_header(
        image_path: str,
        line1: str,
        line2: str,
        line1_size: int = 80,
        line2_size: int = 80,
        padding_top: int = 160,
        gap: int = 10,
        subtitle: Optional[str] = None,
        subtitle_size: Optional[int] = None,
    ) -> None:
        image = _ensure_rgb(PILImage.open(image_path))
        padded = PILImage.new("RGB", (image.width, image.height + padding_top + gap), (255, 255, 255))
        padded.paste(image, (0, padding_top + gap))
        draw = ImageDraw.Draw(padded)

        font1 = FrameProcessor._make_font(line1_size)
        font2 = FrameProcessor._make_font(line2_size)
        font_sub = FrameProcessor._make_font(subtitle_size or line1_size - 10) if subtitle else None

        max_w = padded.width - 40
        l1_lines = FrameProcessor._wrap_text(draw, line1, font1, max_w)
        l1_h = draw.textbbox((0, 0), "Ay", font=font1)[3] + 6
        sub_lines = FrameProcessor._wrap_text(draw, subtitle, font_sub, max_w) if subtitle and font_sub else []
        sub_h = draw.textbbox((0, 0), "Ay", font=font_sub)[3] + 4 if sub_lines else 0
        l2_lines = FrameProcessor._wrap_text(draw, line2, font2, max_w)
        l2_h = draw.textbbox((0, 0), "Ay", font=font2)[3] + 4 if l2_lines else 0

        y = 20
        for l in l1_lines:
            draw.text((20, y), l, font=font1, fill=(0, 0, 0))
            y += l1_h
        if sub_lines:
            y += 4
            for l in sub_lines:
                draw.text((20, y), l, font=font_sub, fill=(0, 0, 0))
                y += sub_h
        l2_y = y + 8
        for l in l2_lines:
            draw.text((20, l2_y), l, font=font2, fill=(0, 0, 0))
            l2_y += l2_h

        image.close()
        padded.save(image_path)

    @staticmethod
    def batch_add_frame_headers(
        image_paths: List[str],
        line1: str,
        label_map: Dict[str, str],
        line1_size: int = 80,
        line2_size: int = 80,
        padding_top: int = 150,
        gap: int = 20,
        subtitle: Optional[str] = None,
        subtitle_size: Optional[int] = None,
    ) -> None:
        for path in image_paths:
            try:
                line2 = label_map.get(path, "")
                FrameProcessor.add_frame_header(path, line1, line2, line1_size, line2_size, padding_top, gap=gap, subtitle=subtitle, subtitle_size=subtitle_size)
            except Exception as e:
                print(f"Erro ao adicionar header em {path}: {e}")

    @staticmethod
    def _truncate_label(lbl: str, max_chars: int = 80) -> str:
        if len(lbl) > max_chars:
            print(f"  [AVISO] Label truncada ({len(lbl)} chars, max {max_chars}): '{lbl}'")
            return lbl[:max_chars-3].rstrip() + "..."
        return lbl

    @staticmethod
    def _make_font(size: int) -> ImageFont.FreeTypeFont:
        size = max(6, int(size))
        try:
            if FrameProcessor.FONT_PATH:
                return ImageFont.truetype(FrameProcessor.FONT_PATH, size)
        except (IOError, OSError):
            pass
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines = []
        cur = ""
        for w in words:
            test = f"{cur} {w}".strip()
            if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def _layout_discrete(
        width: int, entries: List[Tuple[Tuple[int,int,int], str]],
        tfont: ImageFont.FreeTypeFont,
        box_size: int, gap: int, margin: int,
        min_font: int = 8, max_cols: int = 4,
    ) -> Tuple[int, int, ImageFont.FreeTypeFont, int]:
        max_lbl = max(tfont.getlength(lbl) for _, lbl in entries)
        entry_w = box_size + gap
        tfs = tfont.size

        for try_cols in range(min(max_cols, len(entries)), 0, -1):
            avail = (width - 2 * margin) // try_cols - entry_w
            fs = tfs
            while fs > min_font and max_lbl > avail:
                fs -= 2
                tfont = FrameProcessor._make_font(fs)
                max_lbl = max(tfont.getlength(lbl) for _, lbl in entries)
            if max_lbl <= avail:
                return try_cols, math.ceil(len(entries) / try_cols), tfont, fs

        return 1, len(entries), tfont, tfs

    @staticmethod
    def add_legend(
        image_path: str,
        palette: List[str],
        vmin: float = 0,
        vmax: float = 1,
        label: str = "",
        font_size: int = 80,
        discrete_labels: Optional[List[str]] = None,
        cmap_type: str = "sequential",
        legend_order: Optional[List[int]] = None,
        prefix_labels: bool = True,
    ) -> None:
        image = _ensure_rgb(PILImage.open(image_path))
        width = image.width

        colors = [_parse_color(c) for c in palette]
        n = len(colors)
        is_discrete = (n <= 2) or (cmap_type == "categorical")

        margin = 25
        label_h = 55 if label else 0

        if is_discrete:
            box_size = 28
            row_h = 40
            entries = []
            indices = legend_order if legend_order else range(n)
            for i in indices:
                if not discrete_labels or i >= len(discrete_labels) or not discrete_labels[i]:
                    continue
                raw = discrete_labels[i]
                lbl = f"{i} - {raw}" if prefix_labels else raw
                lbl = FrameProcessor._truncate_label(lbl)
                entries.append((colors[i], lbl))
            n_entries = len(entries)

            tfont = FrameProcessor._make_font(font_size - 14)
            cols, rows, tfont, _ = FrameProcessor._layout_discrete(
                width, entries, tfont, box_size, 14, margin, max_cols=3)
            content_h = rows * row_h
        else:
            box_size = 50
            content_h = box_size + 50

        legend_h = label_h + content_h + margin * 2

        padded = PILImage.new("RGB", (width, image.height + legend_h), (255, 255, 255))
        padded.paste(image, (0, 0))
        draw = ImageDraw.Draw(padded)

        sep_y = image.height
        draw.line([(0, sep_y), (width, sep_y)], fill=(200, 200, 200), width=1)

        try:
            lfont = ImageFont.truetype(FrameProcessor.FONT_PATH, font_size) if FrameProcessor.FONT_PATH else ImageFont.load_default()
        except (IOError, OSError):
            lfont = ImageFont.load_default()

        y = image.height + margin

        if label:
            draw.text((20, y), label, font=lfont, fill=(0, 0, 0))
        y += label_h

        if is_discrete:
            col_w = (width - 2 * margin) // cols
            for idx, (color, lbl) in enumerate(entries):
                col = idx // rows if cols > 1 else idx // n_entries
                row = idx % rows if cols > 1 else idx
                bx = margin + col * col_w
                by = y + row * row_h
                draw.rectangle([bx, by, bx + box_size, by + box_size], fill=color)
                draw.rectangle([bx, by, bx + box_size, by + box_size], outline=(100, 100, 100), width=2)
                draw.text((bx + box_size + 10, by + 2), lbl, font=tfont, fill=(0, 0, 0))
        else:
            bar_x, bar_y = margin, y
            bar_w = width - 2 * margin
            for i in range(n - 1):
                x0 = int(bar_x + i * bar_w / (n - 1))
                x1 = int(bar_x + (i + 1) * bar_w / (n - 1))
                draw.rectangle([x0, bar_y, x1, bar_y + box_size], fill=colors[i])
            draw.rectangle([int(bar_x + (n - 1) * bar_w / (n - 1)), bar_y, int(bar_x + bar_w), bar_y + box_size], fill=colors[-1])

            num_ticks = min(n, 6)
            for i in range(num_ticks):
                val = vmin + (vmax - vmin) * i / (num_ticks - 1) if num_ticks > 1 else vmin
                xi = int(bar_x + i * bar_w / (num_ticks - 1))
                lbl = str(int(val))
                bbox = draw.textbbox((0, 0), lbl, font=tfont)
                lw = bbox[2] - bbox[0]
                draw.text((xi - lw // 2, bar_y + box_size + 5), lbl, font=tfont, fill=(0, 0, 0))
                draw.line([(xi, bar_y + box_size), (xi, bar_y + box_size + 5)], fill=(0, 0, 0))

        image.close()
        padded.save(image_path)

    @staticmethod
    def batch_add_legends(
        image_paths: List[str],
        palette: List[str],
        vmin: float = 0,
        vmax: float = 1,
        label: str = "",
        font_size: int = 80,
        cmap_type: str = "sequential",
    ) -> None:
        for path in image_paths:
            try:
                FrameProcessor.add_legend(path, palette, vmin, vmax, label, font_size, cmap_type=cmap_type)
            except Exception as e:
                print(f"Erro ao adicionar legenda em {path}: {e}")

    @staticmethod
    def add_bottom_bar(
        image_path: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        palette: List[str],
        vmin: float = 0,
        vmax: float = 1,
        font_size: int = 80,
        discrete_labels: Optional[List[str]] = None,
        cmap_type: str = "sequential",
        show_legend: bool = True,
        show_scale: bool = True,
        show_north: bool = True,
        rgb_legend: Optional[Dict[str, Any]] = None,
        legend_order: Optional[List[int]] = None,
        prefix_labels: bool = True,
    ) -> None:
        image = _ensure_rgb(PILImage.open(image_path))
        width = image.width

        colors = [_parse_color(c) for c in palette]
        n = len(colors)
        is_discrete = (n <= 2) or (cmap_type == "categorical")

        margin = 25
        gap = 45

        north_w = 100
        north_gap = int(60 * font_size / 80)
        north_zone = (north_w + margin + north_gap) if show_north else 0

        row1_h = max(60, int(font_size * 1.3))
        tfont = FrameProcessor._make_font(font_size - 14)
        if not show_legend:
            legend_content_h = 0
            entries = []
        elif rgb_legend and rgb_legend.get("entries"):
            box_size = 28
            probe = PILImage.new("RGB", (1, 1))
            pdraw = ImageDraw.Draw(probe)
            legend_content_h = FrameProcessor._rgb_legend_height(pdraw, font_size)
            entries = []
        elif n == 0:
            box_size = 28
            legend_content_h = 90
            entries = []
        elif is_discrete:
            box_size = 28
            row_h = 50
            entries = []
            indices = legend_order if legend_order else range(n)
            for i in indices:
                if not discrete_labels or i >= len(discrete_labels) or not discrete_labels[i]:
                    continue
                raw = discrete_labels[i]
                lbl = f"{i} - {raw}" if prefix_labels else raw
                lbl = FrameProcessor._truncate_label(lbl)
                entries.append((colors[i], lbl))
            n_entries = len(entries)

            tfont = FrameProcessor._make_font(font_size - 14)
            cols, rows, tfont, tfs = FrameProcessor._layout_discrete(
                width, entries, tfont, box_size, 14, margin, max_cols=3)
            # Ajusta a altura da linha e o tamanho da caixa ao tamanho real da fonte
            # (labels curtos nao encolhem a fonte e passariam a se sobrepor).
            row_h = max(row_h, int(tfs * 1.5))
            box_size = max(box_size, int(tfs * 0.5))
            legend_content_h = rows * row_h
        else:
            box_size = 50
            legend_content_h = box_size + 50
        row2_h = legend_content_h + 10

        s1 = margin + row1_h + margin if show_scale else 0
        s2 = gap + row2_h + margin if show_legend else 0
        section_h = s1 + s2

        padded = PILImage.new("RGB", (width, image.height + section_h), (255, 255, 255))
        padded.paste(image, (0, 0))
        draw = ImageDraw.Draw(padded)

        sep_y = image.height
        if show_scale:
            draw.line([(0, sep_y), (width, sep_y)], fill=(200, 200, 200), width=1)

        sfont = FrameProcessor._make_font(font_size - 12)

        y = sep_y + margin

        if show_scale:
            center_lat = (lat_min + lat_max) / 2
            km_per_deg = 111.32 * math.cos(math.radians(center_lat))
            map_km = (lon_max - lon_min) * km_per_deg
            km_per_px = map_km / width

            scale_max_w = width - margin - north_zone
            nice = [1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 500]
            target_km = km_per_px * scale_max_w * 0.5
            scale_km = min(nice, key=lambda x: abs(x - target_km))
            bar_px = int(scale_km / km_per_px) if km_per_px > 0 else scale_max_w
            bar_px = min(bar_px, scale_max_w)

            sbx = margin
            sby = y + 5
            bar_cy = sby + 20

            lbl0 = "0"
            lbl1 = f"{int(scale_km)} km"
            b0 = draw.textbbox((0, 0), lbl0, font=sfont)
            b1 = draw.textbbox((0, 0), lbl1, font=sfont)
            b0w = b0[2] - b0[0]
            b1w = b1[2] - b1[0]

            draw.line([(sbx, bar_cy), (sbx + bar_px, bar_cy)], fill=(0, 0, 0), width=3)
            draw.line([(sbx, bar_cy - 5), (sbx, bar_cy + 5)], fill=(0, 0, 0), width=3)
            draw.line([(sbx + bar_px, bar_cy - 5), (sbx + bar_px, bar_cy + 5)], fill=(0, 0, 0), width=3)

            draw.text((sbx - b0w // 2, bar_cy + 8), lbl0, font=sfont, fill=(0, 0, 0))
            draw.text((sbx + bar_px - b1w // 2, bar_cy + 8), lbl1, font=sfont, fill=(0, 0, 0))

            if show_north:
                right_cx = width - margin - north_w // 2
                nac = bar_cy - 8
                arrow_size = 14
                draw.polygon([(right_cx, nac - arrow_size), (right_cx - 7, nac - 2), (right_cx + 7, nac - 2)], fill=(0, 0, 0))
                draw.line([(right_cx, nac - 2), (right_cx, nac + 4)], fill=(0, 0, 0), width=2)
                draw.line([(right_cx - 6, nac + 4), (right_cx + 6, nac + 4)], fill=(0, 0, 0), width=2)
                nlabel = "N"
                nb = draw.textbbox((0, 0), nlabel, font=tfont)
                nw = nb[2] - nb[0]
                nh = nb[3] - nb[1]
                draw.text((right_cx - nw // 2, bar_cy + 8), nlabel, font=tfont, fill=(0, 0, 0))

        ly = y + (row1_h + gap if show_scale else 0)

        if show_legend:
            if rgb_legend and rgb_legend.get("entries"):
                _render_rgb_legend(draw, width, ly, margin, rgb_legend["entries"],
                                   vmin, vmax, font_size=font_size)
            elif n == 0:
                bx = margin
                by = ly + 5
                draw.rectangle([bx, by, bx + box_size, by + box_size], fill=(128, 128, 128))
                draw.rectangle([bx, by, bx + box_size, by + box_size], outline=(100, 100, 100), width=2)
                draw.text((bx + box_size + 10, by), "Sem vegetação", font=tfont, fill=(0, 0, 0))
                draw.text((margin, by + box_size + 10), "Cada mancha de vegetação possui uma cor única", font=tfont, fill=(0, 0, 0))
            elif is_discrete:
                col_w = (width - 2 * margin) // cols
                for idx, (color, lbl) in enumerate(entries):
                    col = idx % cols
                    row = idx // cols
                    bx = margin + col * col_w
                    by = ly + row * row_h
                    box_y = by + (row_h - box_size) // 2
                    draw.rectangle([bx, box_y, bx + box_size, box_y + box_size], fill=color)
                    draw.rectangle([bx, box_y, bx + box_size, box_y + box_size], outline=(100, 100, 100), width=2)
                    txt_bbox = draw.textbbox((0, 0), lbl, font=tfont)
                    txt_h = txt_bbox[3] - txt_bbox[1]
                    txt_y = by + (row_h - txt_h) // 2
                    draw.text((bx + box_size + 10, txt_y), lbl, font=tfont, fill=(0, 0, 0))
            else:
                bar_x = margin
                bar_w = width - 2 * margin
                bar_y_pos = ly
                for i in range(n - 1):
                    x0 = int(bar_x + i * bar_w / (n - 1))
                    x1 = int(bar_x + (i + 1) * bar_w / (n - 1))
                    draw.rectangle([x0, bar_y_pos, x1, bar_y_pos + box_size], fill=colors[i])
                draw.rectangle([int(bar_x + (n - 1) * bar_w / (n - 1)), bar_y_pos, int(bar_x + bar_w), bar_y_pos + box_size], fill=colors[-1])

                num_ticks = min(n, 6)
                for i in range(num_ticks):
                    val = vmin + (vmax - vmin) * i / (num_ticks - 1) if num_ticks > 1 else vmin
                    xi = int(bar_x + i * bar_w / (num_ticks - 1))
                    lbl = str(int(val))
                    bbox = draw.textbbox((0, 0), lbl, font=tfont)
                    lw = bbox[2] - bbox[0]
                    if i == 0:
                        tx = margin
                    elif i == num_ticks - 1:
                        tx = width - margin - lw
                    else:
                        tx = xi - lw // 2
                    draw.text((tx, bar_y_pos + box_size + 5), lbl, font=tfont, fill=(0, 0, 0))
                    draw.line([(xi, bar_y_pos + box_size), (xi, bar_y_pos + box_size + 5)], fill=(0, 0, 0))

        image.close()
        padded.save(image_path)

    @staticmethod
    def batch_add_bottom_bars(
        image_paths: List[str],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        palette: List[str],
        vmin: float = 0,
        vmax: float = 1,
        font_size: int = 50,
        discrete_labels: Optional[List[str]] = None,
        cmap_type: str = "sequential",
        show_legend: bool = True,
        show_scale: bool = True,
        show_north: bool = True,
        rgb_legend: Optional[Dict[str, Any]] = None,
        legend_order: Optional[List[int]] = None,
        prefix_labels: bool = True,
    ) -> None:
        for path in image_paths:
            try:
                FrameProcessor.add_bottom_bar(path, lon_min, lon_max, lat_min, lat_max, palette, vmin, vmax, font_size, discrete_labels=discrete_labels, cmap_type=cmap_type, show_legend=show_legend, show_scale=show_scale, show_north=show_north, rgb_legend=rgb_legend, legend_order=legend_order, prefix_labels=prefix_labels)
            except Exception as e:
                print(f"Erro ao adicionar barra inferior em {path}: {e}")

    @staticmethod
    def add_scale_bar(
        image_path: str,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        bar_width_px: int = 350,
        font_size: int = 40,
    ) -> None:
        image = _ensure_rgb(PILImage.open(image_path))
        width = image.width
        center_lat = (lat_min + lat_max) / 2
        km_per_deg = 111.32 * _math.cos(_math.radians(center_lat))
        map_km = (lon_max - lon_min) * km_per_deg
        km_per_px = map_km / width
        target_km = km_per_px * bar_width_px
        nice = [1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 500]
        scale_km = min(nice, key=lambda x: abs(x - target_km))
        bar_px = int(scale_km / km_per_px) if km_per_px > 0 else bar_width_px

        bar_h = 120
        padded = PILImage.new("RGB", (width, image.height + bar_h), (255, 255, 255))
        padded.paste(image, (0, 0))
        draw = ImageDraw.Draw(padded)

        try:
            font = ImageFont.truetype(FrameProcessor.FONT_PATH, font_size) if FrameProcessor.FONT_PATH else ImageFont.load_default()
            sfont = ImageFont.truetype(FrameProcessor.FONT_PATH, font_size - 8) if FrameProcessor.FONT_PATH else ImageFont.load_default()
        except (IOError, OSError):
            font = sfont = ImageFont.load_default()

        bx = width - bar_px - 120
        by = image.height + 56

        draw.line([(bx, by), (bx + bar_px, by)], fill=(0, 0, 0), width=3)
        draw.line([(bx, by - 6), (bx, by + 6)], fill=(0, 0, 0), width=3)
        draw.line([(bx + bar_px, by - 6), (bx + bar_px, by + 6)], fill=(0, 0, 0), width=3)

        lbl0 = "0"
        lbl1 = f"{int(scale_km)} km"
        b0 = draw.textbbox((0, 0), lbl0, font=font)
        draw.text((bx - (b0[2] - b0[0]) // 2, by + 12), lbl0, font=font, fill=(0, 0, 0))
        b1 = draw.textbbox((0, 0), lbl1, font=font)
        draw.text((bx + bar_px - (b1[2] - b1[0]) // 2, by + 12), lbl1, font=font, fill=(0, 0, 0))

        ncx = width - 45
        ncy = by
        arrow_size = 14
        draw.polygon([(ncx, ncy - arrow_size), (ncx - 7, ncy), (ncx + 7, ncy)], fill=(0, 0, 0))
        nlabel = "N"
        nb = draw.textbbox((0, 0), nlabel, font=sfont)
        nw = nb[2] - nb[0]
        nh = nb[3] - nb[1]
        draw.text((ncx - nw // 2, ncy - arrow_size - nh - 4), nlabel, font=sfont, fill=(0, 0, 0))

        image.close()
        padded.save(image_path)

    @staticmethod
    def batch_add_scale_bars(
        image_paths: List[str],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> None:
        for path in image_paths:
            try:
                FrameProcessor.add_scale_bar(path, lon_min, lon_max, lat_min, lat_max)
            except Exception as e:
                print(f"Erro ao adicionar escala em {path}: {e}")

    @staticmethod
    def add_margin(image_path: str, margin_px: int = 30) -> None:
        image = _ensure_rgb(PILImage.open(image_path))
        padded = PILImage.new("RGB", (image.width + margin_px * 2, image.height + margin_px * 2), (255, 255, 255))
        padded.paste(image, (margin_px, margin_px))
        image.close()
        padded.save(image_path)

    @staticmethod
    def batch_add_margins(image_paths: List[str], margin_px: int = 30) -> None:
        for path in image_paths:
            try:
                FrameProcessor.add_margin(path, margin_px)
            except Exception as e:
                print(f"Erro ao adicionar margem em {path}: {e}")

    @staticmethod
    def resize_image(image_path: str, target_height: int) -> str:
        image = PILImage.open(image_path)
        aspect = image.width / image.height
        new_w = int(target_height * aspect)
        resized = image.resize((new_w, target_height), PILImage.Resampling.LANCZOS)
        resized.save(image_path)
        return image_path

    @staticmethod
    def batch_resize(image_paths: List[str], target_height: int) -> List[str]:
        result = []
        widths = []
        for path in image_paths:
            try:
                FrameProcessor.resize_image(path, target_height)
                img = PILImage.open(path)
                widths.append(img.width)
                result.append(path)
                img.close()
            except Exception as e:
                print(f"Erro ao redimensionar {path}: {e}")
        if widths:
            max_w = max(widths)
            for path in result:
                img = _ensure_rgb(PILImage.open(path))
                if img.width < max_w:
                    padded = PILImage.new("RGB", (max_w, img.height), (255, 255, 255))
                    padded.paste(img, (0, 0))
                    img.close()
                    padded.save(path)
                else:
                    img.close()
        return result

    @staticmethod
    def render_legend_overlay(
        width: int,
        palette: List[str],
        vmin: float = 0,
        vmax: float = 1,
        font_size: int = 80,
        discrete_labels: Optional[List[str]] = None,
        cmap_type: str = "sequential",
        label: str = "",
        rgb_legend: Optional[Dict[str, Any]] = None,
        legend_order: Optional[List[int]] = None,
        prefix_labels: bool = True,
        output_path: Optional[str] = None,
    ) -> str:
        colors = [_parse_color(c) for c in palette]
        n = len(colors)
        is_discrete = (n <= 2) or (cmap_type == "categorical")
        margin = 25
        label_h = font_size + 20 if label else 0

        if rgb_legend and rgb_legend.get("entries"):
            probe = PILImage.new("RGB", (1, 1))
            pdraw = ImageDraw.Draw(probe)
            content_h = FrameProcessor._rgb_legend_height(pdraw, font_size)
            tfont = FrameProcessor._make_font(font_size - 14)
            entries_rgb = rgb_legend["entries"]
        elif is_discrete and discrete_labels:
            box_size = max(20, int(font_size * 0.5))
            row_h = box_size + 18
            entries = []
            indices = legend_order if legend_order else range(n)
            for i in indices:
                if i >= len(discrete_labels) or not discrete_labels[i]:
                    continue
                raw = discrete_labels[i]
                lbl = f"{i} - {raw}" if prefix_labels else raw
                lbl = FrameProcessor._truncate_label(lbl)
                entries.append((colors[i], lbl))
            if entries:
                tfont = FrameProcessor._make_font(font_size - 14)
                cols, rows, tfont, tfs = FrameProcessor._layout_discrete(
                    width, entries, tfont, box_size, 14, margin, max_cols=3)
                # A altura da linha precisa acompanhar a fonte real; labels curtos
                # nao encolhem a fonte e as linhas se sobrepunham.
                row_h = max(row_h, int(tfs * 1.35))
                box_size = max(box_size, int(tfs * 0.5))
                content_h = rows * row_h
            else:
                box_size = max(20, int(font_size * 0.5))
                content_h = box_size + font_size + 10
                tfont = FrameProcessor._make_font(font_size - 14)
        else:
            box_size = max(30, int(font_size * 0.6))
            content_h = box_size + font_size + 10
            tfont = FrameProcessor._make_font(font_size - 14)

        legend_h = label_h + content_h + margin * 2
        img = PILImage.new("RGB", (width, legend_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            lfont = ImageFont.truetype(FrameProcessor.FONT_PATH, font_size) if FrameProcessor.FONT_PATH else ImageFont.load_default()
        except (IOError, OSError):
            lfont = ImageFont.load_default()

        y = margin
        if label:
            draw.text((20, y), label, font=lfont, fill=(0, 0, 0))
        y += label_h

        if rgb_legend and rgb_legend.get("entries"):
            _render_rgb_legend(draw, width, y - label_h + 5, margin, entries_rgb, vmin, vmax, font_size=font_size)
        elif is_discrete and discrete_labels and entries:
            col_w = (width - 2 * margin) // cols
            for idx, (color, lbl) in enumerate(entries):
                col = idx // rows if cols > 1 else idx // len(entries)
                row = idx % rows if cols > 1 else idx
                bx = margin + col * col_w
                by = y + row * row_h
                box_y = by + (row_h - box_size) // 2
                draw.rectangle([bx, box_y, bx + box_size, box_y + box_size], fill=color)
                draw.rectangle([bx, box_y, bx + box_size, box_y + box_size], outline=(100, 100, 100), width=2)
                txt_bbox = draw.textbbox((0, 0), lbl, font=tfont)
                txt_h = txt_bbox[3] - txt_bbox[1]
                txt_y = by + (row_h - txt_h) // 2 - txt_bbox[1]
                draw.text((bx + box_size + 10, txt_y), lbl, font=tfont, fill=(0, 0, 0))
        else:
            bar_x, bar_y = margin, y
            bar_w = width - 2 * margin
            for i in range(n - 1):
                x0 = int(bar_x + i * bar_w / (n - 1))
                x1 = int(bar_x + (i + 1) * bar_w / (n - 1))
                draw.rectangle([x0, bar_y, x1, bar_y + box_size], fill=colors[i])
            draw.rectangle([int(bar_x + (n - 1) * bar_w / (n - 1)), bar_y, int(bar_x + bar_w), bar_y + box_size], fill=colors[-1])
            num_ticks = min(n, 6)
            for i in range(num_ticks):
                val = vmin + (vmax - vmin) * i / (num_ticks - 1) if num_ticks > 1 else vmin
                xi = int(bar_x + i * bar_w / (num_ticks - 1))
                lbl = str(int(val))
                bbox = draw.textbbox((0, 0), lbl, font=tfont)
                lw = bbox[2] - bbox[0]
                if i == 0:
                    tx = margin
                elif i == num_ticks - 1:
                    tx = width - margin - lw
                else:
                    tx = xi - lw // 2
                draw.text((tx, bar_y + box_size + 8), lbl, font=tfont, fill=(0, 0, 0))
                draw.line([(xi, bar_y + box_size), (xi, bar_y + box_size + 8)], fill=(0, 0, 0))

        if output_path:
            img.save(output_path)
            return output_path
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            img.save(tmp)
            return tmp

    @staticmethod
    def paste_overlay_below(image_path: str, overlay_path: str) -> None:
        image = _ensure_rgb(PILImage.open(image_path))
        overlay = PILImage.open(overlay_path)
        new_h = image.height + overlay.height
        result = PILImage.new("RGB", (image.width, new_h), (255, 255, 255))
        result.paste(image, (0, 0))
        result.paste(overlay, (0, image.height))
        image.close()
        overlay.close()
        result.save(image_path)

    @staticmethod
    def batch_paste_overlay_below(image_paths: List[str], overlay_path: str) -> None:
        for path in image_paths:
            try:
                FrameProcessor.paste_overlay_below(path, overlay_path)
            except Exception as e:
                print(f"Erro ao colar overlay em {path}: {e}")
