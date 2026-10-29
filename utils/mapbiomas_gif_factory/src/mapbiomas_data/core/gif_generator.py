import math
import os
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image as PILImage, ImageDraw, ImageFont
from ..utils.file_utils import ensure_dir, clean_filename


class GIFGenerator:
    """Gerar GIFs animados a partir de frames de imagens."""

    def __init__(self, frame_duration: int = 300, loop_count: int = 0, quality: int = 95):
        self.frame_duration = frame_duration
        self.loop_count = loop_count
        self.quality = quality

    def create_gif(
        self,
        image_paths: List[str],
        output_dir: str,
        filename: str = "output.gif",
        sort_frames: bool = True,
        viz_palette: Optional[List[str]] = None,
    ) -> str:
        if not image_paths:
            raise ValueError("Nenhuma imagem fornecida para o GIF")

        if sort_frames:
            image_paths = sorted(image_paths)

        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(image_paths)} frames ausentes para o GIF: "
                f"{missing[0]}" + (f", ..." if len(missing) > 1 else "")
            )

        images = []
        for path in image_paths:
            img = PILImage.open(path)
            if img.mode == "RGBA":
                bg = PILImage.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        if not images:
            raise ValueError("Nenhuma imagem pôde ser carregada")

        common_size = self._get_common_size(images)
        resized = [img.resize(common_size, PILImage.Resampling.LANCZOS) for img in images]

        # --- Palette strategy ---
        # Com paleta viz definida (produtos categóricos/contínuos): paleta
        # compartilhada, preserva as cores exatas das classes.
        # Sem paleta viz (produtos RGB com cores fotográficas): paleta adaptativa
        # POR FRAME com dithering - o GIF grava paleta local em cada frame e as
        # cores ficam tão fiéis quanto possível ao PNG original.
        use_shared_palette = bool(viz_palette)
        optimize_palette = True

        if not use_shared_palette:
            resized_rgb = [
                img.quantize(
                    colors=256,
                    method=PILImage.Quantize.MEDIANCUT,
                    dither=PILImage.Dither.FLOYDSTEINBERG,
                )
                for img in resized
            ]
            optimize_palette = False
        else:
            # --- Build palette from known viz colors + sampled data ---
            known_colors = set()
            known_colors.add((255, 255, 255))  # fundo
            known_colors.add((0, 0, 0))        # texto

            for hex_color in viz_palette:
                hex_color = hex_color.lstrip("#")
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                known_colors.add(rgb)

            # Sample frames to capture remaining colors
            sample_count = min(12, len(resized))
            palette_collage = PILImage.new("RGB", (sample_count * 64, 64))
            for i, img in enumerate(resized[:sample_count]):
                strip = img.resize((64, 64), PILImage.Resampling.LANCZOS)
                palette_collage.paste(strip, (i * 64, 0))

            # Build palette: known colors first, then MEDIANCUT for remaining slots
            known_list = list(known_colors)
            remaining_slots = 256 - len(known_list)

            if remaining_slots >= 0:
                # Enough room: place known colors at start of palette
                palette_pixels = []
                for r, g, b in known_list:
                    palette_pixels.extend([r, g, b])
                palette_pixels += [0] * (768 - len(palette_pixels))
                known_palette_img = PILImage.new("P", (1, 1))
                known_palette_img.putpalette(palette_pixels)

                # Quantize sampled collage to fill remaining slots
                if remaining_slots > 0:
                    sampled = palette_collage.quantize(
                        colors=remaining_slots,
                        method=PILImage.Quantize.MEDIANCUT,
                        dither=PILImage.Dither.NONE,
                    )
                    # Merge palettes: known + sampled
                    final_palette = known_list[:]
                    sampled_pal = sampled.getpalette()
                    for y in range(sampled.height):
                        for x in range(sampled.width):
                            if len(final_palette) >= 256:
                                break
                            idx = sampled.getpixel((x, y))
                            offset = idx * 3
                            if offset + 2 < len(sampled_pal):
                                color = (sampled_pal[offset], sampled_pal[offset + 1], sampled_pal[offset + 2])
                                if color not in known_colors:
                                    final_palette.append(color)
                        if len(final_palette) >= 256:
                            break

                    final_pixels = []
                    for r, g, b in final_palette[:256]:
                        final_pixels.extend([r, g, b])
                    final_pixels += [0] * (768 - len(final_pixels))
                    final_img = PILImage.new("P", (1, 1))
                    final_img.putpalette(final_pixels)
                    shared_palette = final_img
                else:
                    shared_palette = known_palette_img
            else:
                # More than 256 known colors: use MEDIANCUT on full collage
                shared_palette = palette_collage.quantize(
                    colors=256,
                    method=PILImage.Quantize.MEDIANCUT,
                    dither=PILImage.Dither.NONE,
                )

            resized_rgb = [
                img.quantize(palette=shared_palette, dither=PILImage.Dither.NONE)
                for img in resized
            ]

        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, clean_filename(filename))
        if not output_path.endswith(".gif"):
            output_path += ".gif"

        resized_rgb[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=resized_rgb[1:],
            duration=self.frame_duration,
            loop=self.loop_count,
            optimize=optimize_palette,
        )
        return output_path

    def _get_common_size(self, images: List[PILImage.Image]) -> tuple:
        sizes = [img.size for img in images]
        min_w = min(s[0] for s in sizes)
        min_h = min(s[1] for s in sizes)
        return (min_w, min_h)

    def create_collage(
        self,
        image_paths: List[str],
        output_dir: str,
        filename: str = "collage.png",
        grid_size: Optional[int] = None,
        cell_labels: Optional[List[str]] = None,
        font_path: Optional[str] = None,
        cell_height: int = 300,
        force_horizontal: bool = False,
    ) -> str:
        if not image_paths:
            raise ValueError("Nenhuma imagem fornecida para a colagem")

        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(image_paths)} frames ausentes para a colagem: "
                f"{missing[0]}" + (f", ..." if len(missing) > 1 else "")
            )

        images = []
        for p in image_paths:
            img = PILImage.open(p)
            if img.mode == "RGBA":
                bg = PILImage.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        if not images:
            raise ValueError("Nenhuma imagem pôde ser carregada")

        n = len(images)
        if force_horizontal:
            n_cols = n
            n_rows = 1
        elif grid_size is not None:
            n_cols = grid_size
            n_rows = math.ceil(n / n_cols)
        else:
            if n <= 3:
                n_cols = n
            else:
                n_cols = min(max(math.ceil(n / 5), 2), 10)
            n_rows = math.ceil(n / n_cols)

        target_h = cell_height
        cell_pad = 75 if cell_labels else 0

        resized = []
        for img in images:
            aspect = img.width / img.height
            new_w = int(target_h * aspect)
            small = img.resize((new_w, target_h), PILImage.Resampling.LANCZOS)
            if cell_labels:
                padded = PILImage.new("RGB", (new_w, target_h + cell_pad), color="white")
                padded.paste(small, (0, cell_pad))
                resized.append(padded)
            else:
                resized.append(small)

        frame_w = max(img.width for img in resized)
        frame_h = max(img.height for img in resized)

        grid_w = n_cols * frame_w
        grid_h = n_rows * frame_h

        collage = PILImage.new("RGB", (grid_w, grid_h), color="white")

        for idx, img in enumerate(resized):
            row = idx // n_cols
            col = idx % n_cols
            x = col * frame_w
            y = row * frame_h
            collage.paste(img, (x, y))

        if cell_labels:
            draw = ImageDraw.Draw(collage)
            try:
                if font_path and os.path.exists(font_path):
                    label_font = ImageFont.truetype(font_path, 60)
                else:
                    label_font = ImageFont.load_default()
            except (IOError, OSError):
                label_font = ImageFont.load_default()

            for idx, label in enumerate(cell_labels):
                if idx >= len(images):
                    break
                row = idx // n_cols
                col = idx % n_cols
                cx = col * frame_w
                cy = row * frame_h
                draw.text((cx + 10, cy + 10), label, font=label_font, fill=(0, 0, 0))

        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, clean_filename(filename))
        collage.save(output_path, quality=self.quality)
        return output_path
