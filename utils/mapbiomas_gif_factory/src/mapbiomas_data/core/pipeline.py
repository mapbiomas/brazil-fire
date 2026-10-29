import json
import os
import re
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ee
from PIL import Image as PILImage

from ..config import ConfigLoader
from ..core import (
    DatasetManager,
    TerritoryManager,
    VisualizationManager,
    GIFGenerator,
    FrameProcessor,
)
from ..core.ee_downloader import EEDownloader
from ..core.ee_transforms import run_processor, PROCESSOR_REGISTRY, only_coverage
from ..core.state_manager import StateManager
from ..utils.file_utils import ensure_dir


class Pipeline:
    """Orquestrador completo do pipeline: download -> processar -> GIF."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.datasets = DatasetManager(config)
        self.territories = TerritoryManager(config)
        self.visualizations = VisualizationManager(config)
        self.downloader = EEDownloader(config)
        self.gif_generator = GIFGenerator(
            frame_duration=config.get_processing_config("gif_creation").get("frame_duration", 300),
            loop_count=config.get_processing_config("gif_creation").get("loop_count", 0),
            quality=config.get_processing_config("gif_creation").get("quality", 95),
        )
        self.frame_processor = FrameProcessor()

    def run(
        self,
        dataset_id: str,
        product_id: str,
        territory_id: str,
        viz_key: Optional[str] = None,
        output_dir: Optional[str] = None,
        create_collage: bool = True,
        add_labels: bool = True,
        vertical_dimension: int = 2000,
        max_bands: int = 0,
        band_names_filter: Optional[List[str]] = None,
        cell_height: int = 300,
        resume: bool = False,
        font_scale: float = 1.0,
    ) -> Dict[str, Any]:
        fs = lambda n: int(n * font_scale)
        result = {
            "dataset": dataset_id,
            "product": product_id,
            "territory": territory_id,
            "status": "started",
            "frames": [],
            "gif_path": None,
        }

        timings = {}
        t_start = time.perf_counter()

        try:
            try:
                ee.Initialize(project=self.config.ee_project_id)
            except Exception:
                pass
            product_info = self.datasets.get_product(dataset_id, product_id)
            territory_info = self.territories.get_territory(territory_id)
            region_fc = self.territories.get_feature_collection(territory_id)
            overlay_fc = self.territories.get_overlay_fc(territory_id)

            if region_fc is None:
                return {**result, "status": "error", "error": "EE não disponível"}

            bounds_ee = region_fc.geometry().bounds().getInfo()
            coords = bounds_ee['coordinates'][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            bounds = {
                'lon_min': min(lons), 'lon_max': max(lons),
                'lat_min': min(lats), 'lat_max': max(lats),
            }

            if not viz_key:
                viz_key = product_info.get("visualization")
            if not viz_key:
                viz_key = "fire"
            viz_params = self.visualizations.get_viz_params(viz_key)

            if not output_dir:
                output_dir = os.path.join(
                    self.config.get_output_dir(),
                    dataset_id,
                    product_id,
                    territory_id,
                )
            ensure_dir(output_dir)

            frames_pure_dir = os.path.join(output_dir, "frames_pure")
            frames_clean_dir = os.path.join(output_dir, "frames_clean")
            collages_dir = os.path.join(output_dir, "collages_combined")
            collages_clean_dir = os.path.join(output_dir, "collages_clean")
            frames_maps_dir = os.path.join(output_dir, "frames_maps")
            gifs_dir = os.path.join(output_dir, "gifs")
            overlays_dir = os.path.join(output_dir, "overlays")
            metadata_dir = os.path.join(output_dir, "metadata")
            csv_dir = os.path.join(metadata_dir, "csv")
            for d in [frames_pure_dir, frames_clean_dir, frames_maps_dir,
                      gifs_dir, collages_dir, collages_clean_dir, overlays_dir, metadata_dir, csv_dir]:
                ensure_dir(d)

            state = StateManager(output_dir)
            if not resume:
                state.clear_all()

            print(f"\n{'='*60}")
            print(f"Pipeline: {product_info.get('name', product_id)}")
            print(f"Território: {territory_info['name']}")
            print(f"Visualização: {viz_params.get('name', viz_key)}")
            print(f"Saída: {output_dir}")
            if resume:
                print(f"Modo: resume (pulando etapas ja concluidas)")
            print(f"{'='*60}")

            image = self._load_product_image(product_info)

            bands_slice = product_info.get("bands_slice")
            if bands_slice and len(bands_slice) == 2:
                image = image.slice(bands_slice[0], bands_slice[1])
                print(f"Bandas fatiadas: [{bands_slice[0]}:{bands_slice[1]}]")

            real_band_names = self.downloader.get_band_names(image)
            yaml_bands = product_info.get("bands", [])
            if yaml_bands:
                missing_bands = [b for b in yaml_bands if b not in real_band_names]
                if missing_bands:
                    raise ValueError(f"Bandas não encontradas: {missing_bands}")
                valid = [b for b in yaml_bands if b in real_band_names]
                if valid:
                    band_names = valid
                else:
                    band_names = real_band_names
            else:
                band_names = real_band_names
            if band_names_filter:
                band_names = [b for b in band_names if any(b.endswith(s) for s in band_names_filter)]
                print(f"  Filtradas para anos específicos: {band_names_filter}")
            if max_bands > 0 and len(band_names) > max_bands:
                band_names = band_names[:max_bands]
            print(f"Bandas: {len(band_names)} (de {len(real_band_names)} disponíveis)")

            is_rgb = product_info.get("rgb", False)
            rgb_band_groups = None
            if is_rgb:
                groups = {}
                for b in band_names:
                    match = re.search(r'(\d{4})$', b)
                    if match:
                        year = match.group(1)
                        if year not in groups:
                            groups[year] = []
                        groups[year].append(b)
                CHANNEL_ORDER = product_info.get("channel_order") or {
                    'swir1': 0, 'nir': 1, 'red': 2,
                    'swir1_median': 0, 'nir_median': 1, 'red_median': 2,
                    'green_median': 3, 'blue_median': 4,
                }
                rgb_band_groups = [
                    (y, sorted(grp, key=lambda b: CHANNEL_ORDER.get(b.split('_')[0], 99)))
                    for y, grp in sorted(groups.items())
                ]
                if rgb_band_groups and viz_params.get("rgb_legend") is True:
                    first_group_labels = [b.rsplit("_", 1)[0].upper() for b in rgb_band_groups[0][1]]
                    channel_colors = ["#ff0000", "#00ff00", "#0000ff"]
                    viz_params["rgb_legend"] = {
                        "entries": [
                            {"label": lbl, "color": clr}
                            for lbl, clr in zip(first_group_labels, channel_colors)
                        ]
                    }

            self.downloader.vertical_dimension = vertical_dimension
            prefix = f"{product_id}_"

            if not state.is_complete("download"):
                print(f"\n[1/4] Baixando frames...")
                t1 = time.perf_counter()
                if is_rgb and rgb_band_groups:
                    frame_paths = self.downloader.download_rgb_frames(
                        image=image,
                        band_groups=rgb_band_groups,
                        viz_params=viz_params,
                        region_fc=region_fc,
                        output_dir=frames_pure_dir,
                        prefix=prefix,
                        overlay_fc=overlay_fc,
                    )
                else:
                    frame_paths = self.downloader.download_frames(
                        image=image,
                        band_names=band_names,
                        viz_params=viz_params,
                        region_fc=region_fc,
                        output_dir=frames_pure_dir,
                        prefix=prefix,
                    add_labels=add_labels,
                    overlay_fc=overlay_fc,
                )
                timings["download"] = round(time.perf_counter() - t1, 1)
                print(f"  ({timings['download']}s)")

                if not frame_paths:
                    return {**result, "status": "error", "error": "Nenhum frame baixado"}

                result["frames"] = frame_paths

                expected = len(rgb_band_groups) if (is_rgb and rgb_band_groups) else len(band_names)
                if len(frame_paths) != expected:
                    return {**result, "status": "error",
                            "error": f"Download incompleto: {len(frame_paths)}/{expected} frames"}

                for fp in frame_paths:
                    if not os.path.exists(fp):
                        return {**result, "status": "error", "error": f"Frame faltando apos download: {fp}"}
                    try:
                        PILImage.open(fp).verify()
                    except Exception as e:
                        return {**result, "status": "error",
                                "error": f"Frame corrompido: {fp} ({e})"}

                state.mark_complete("download")
            else:
                completed = state.get_completed()
                print(f"\n[1/4] Download ja concluido (resume)")
                frame_paths = self._load_existing_frames(frames_pure_dir, f"{product_id}_")
                expected = len(rgb_band_groups) if (is_rgb and rgb_band_groups) else len(band_names)
                if len(frame_paths) != expected:
                    print(f"  [AVISO] Resume com {len(frame_paths)}/{expected} frames. Refazendo download...")
                    state.clear_all()
                    return self.run(dataset_id, product_id, territory_id, viz_key, output_dir,
                                    create_collage, add_labels, vertical_dimension, max_bands,
                                    band_names_filter, cell_height, resume=False)
                result["frames"] = frame_paths
                timings["download"] = 0

            if not state.is_complete("resize"):
                if getattr(self, "progress_callback", None):
                    self.progress_callback("stage", "Preparando frames cartográficos")
                print(f"\n[2/4] Redimensionando frames...")
                t2 = time.perf_counter()
                FrameProcessor.batch_resize(frame_paths, vertical_dimension)
                timings["resize"] = round(time.perf_counter() - t2, 1)
                print(f"  ({timings['resize']}s)")
                state.mark_complete("resize")
            else:
                print(f"\n[2/4] Redimensionamento ja concluido (resume)")
                timings["resize"] = 0

            legend_overlay = None
            if create_collage or add_labels:
                if not state.is_complete("overlay_legend"):
                    print(f"\n[2b/4] Renderizando legendas...")
                    t_ol = time.perf_counter()
                    frame_w = PILImage.open(frame_paths[0]).width
                    legend_overlay = os.path.join(overlays_dir, "legend_frames.png")
                    FrameProcessor.render_legend_overlay(
                        width=frame_w, palette=viz_params.get("palette", ["fdfdfd", "800000"]),
                        vmin=viz_params.get("min", 0), vmax=viz_params.get("max", 1),
                        font_size=fs(54) if viz_params.get("rgb_legend") else fs(80),
                        discrete_labels=viz_params.get("discrete_labels"),
                        cmap_type=viz_params.get("cmap_type", "sequential"),
                        label="", rgb_legend=viz_params.get("rgb_legend"),
                        legend_order=viz_params.get("legend_order"),
                        prefix_labels=viz_params.get("prefix_labels", True),
                        output_path=legend_overlay,
                    )
                    timings["overlay_legend"] = round(time.perf_counter() - t_ol, 1)
                    print(f"  ({timings['overlay_legend']}s)")
                    state.mark_complete("overlay_legend")
                else:
                    legend_overlay = os.path.join(overlays_dir, "legend_frames.png")

            product_label = product_info.get("name", product_id)
            territory_name = territory_info["name"]
            collection_title = product_info.get("collection_title", "")
            if collection_title:
                title_line1 = territory_name
                title_line2 = f"{collection_title} - {product_label}"
                title_line3 = None
            else:
                title_line1 = product_label
                title_line2 = territory_name
                title_line3 = None

            year_offset = int(product_info.get("year_offset", 0) or 0)
            label_map = {}
            for fp in frame_paths:
                band_name = os.path.splitext(os.path.basename(fp))[0]
                if band_name.startswith(prefix):
                    band_name = band_name[len(prefix):]
                year_part = re.sub(r"^[a-z_]+", "", band_name).replace("_", "\u2192")
                if year_offset and re.fullmatch(r"\d{4}", year_part):
                    year_part = str(int(year_part) + year_offset)
                label_map[os.path.basename(fp)] = year_part

            need_clean = create_collage or add_labels
            clean_paths = []
            if need_clean:
                if not state.is_complete("frames_clean"):
                    print(f"\n[3a/4] Criando frames_clean (ano + escala + norte)...")
                    t3a = time.perf_counter()
                    for fp in frame_paths:
                        year_text = label_map.get(os.path.basename(fp), "")
                        if not year_text:
                            continue
                        clean_fp = os.path.join(frames_clean_dir, os.path.basename(fp))
                        shutil.copy2(fp, clean_fp)
                        FrameProcessor.add_year_only(clean_fp, year_text, font_size=fs(160), draw_north=bool(collection_title))
                        clean_paths.append(clean_fp)
                    FrameProcessor.batch_add_bottom_bars(
                        clean_paths, bounds['lon_min'], bounds['lon_max'],
                        bounds['lat_min'], bounds['lat_max'],
                        palette=viz_params.get("palette", ["fdfdfd", "800000"]),
                        vmin=viz_params.get("min", 0), vmax=viz_params.get("max", 1),
                        font_size=fs(80), discrete_labels=viz_params.get("discrete_labels"),
                        cmap_type=viz_params.get("cmap_type", "sequential"),
                        show_legend=False, show_scale=True, show_north=not bool(collection_title),
                        prefix_labels=viz_params.get("prefix_labels", True),
                    )
                    timings["frames_clean"] = round(time.perf_counter() - t3a, 1)
                    print(f"    {len(clean_paths)} frames em frames_clean/ ({timings['frames_clean']}s)")
                    state.mark_complete("frames_clean")
                else:
                    print(f"\n[3a/4] frames_clean ja existem (resume)")
                    timings["frames_clean"] = 0
                    clean_paths = [
                        os.path.join(frames_clean_dir, os.path.basename(fp))
                        for fp in frame_paths
                        if os.path.isfile(os.path.join(frames_clean_dir, os.path.basename(fp)))
                    ]

            collage_path = None
            special_paths = []
            map_paths = []
            if create_collage:
                if getattr(self, "progress_callback", None):
                    self.progress_callback("stage", "Montando colagens")
                if not state.is_complete("collage"):
                    print(f"\n[3b/4] Criando colagem principal...")
                    t3b = time.perf_counter()
                    collage_filename = f"{product_id}_{territory_id}_collage.png"
                    collage_path = self.gif_generator.create_collage(
                        image_paths=clean_paths,
                        output_dir=collages_dir,
                        filename=collage_filename,
                        cell_labels=None,
                        font_path=FrameProcessor.FONT_PATH,
                        cell_height=cell_height,
                    )
                    timings["collage_build"] = round(time.perf_counter() - t3b, 1)
                    print(f"    grid: {timings['collage_build']}s")
                    state.mark_complete("collage")
                else:
                    print(f"\n[3b/4] Colagem principal ja criada (resume)")
                    timings["collage_build"] = 0
                    collage_path = os.path.join(collages_dir, f"{product_id}_{territory_id}_collage.png")

                if not state.is_complete("special_collages"):
                    print(f"\n[3b2/4] Criando colagens especiais...")
                    t_sc = time.perf_counter()
                    special_modes = {
                        "decadal": {"grid_size": 2},
                        "quinzenal": {"force_horizontal": True},
                        "first_last": {"force_horizontal": True},
                        "last_six": {"grid_size": 3},
                    }
                    special_paths = self._build_special_collages(
                        collages_dir, product_id, territory_id,
                        clean_paths, frames_clean_dir, True,
                        special_modes, cell_height, fs,
                    )
                    timings["special_collages"] = round(time.perf_counter() - t_sc, 1)
                    print(f"    colagens especiais: {timings['special_collages']}s")
                    state.mark_complete("special_collages")
                else:
                    print(f"\n[3b2/4] Colagens especiais ja criadas (resume)")
                    timings["special_collages"] = 0
                    for mode in ["decadal", "quinzenal", "first_last", "last_six"]:
                        sp = os.path.join(collages_dir, f"{product_id}_{territory_id}_collage_{mode}.png")
                        if os.path.isfile(sp):
                            special_paths.append(sp)

                if not state.is_complete("collages_clean"):
                    print(f"\n[3b3/4] Salvando collages_clean...")
                    all_for_clean = [cp for cp in [collage_path] + special_paths if cp and os.path.isfile(cp)]
                    for cp in all_for_clean:
                        shutil.copy2(cp, os.path.join(collages_clean_dir, os.path.basename(cp)))
                    state.mark_complete("collages_clean")
                else:
                    print(f"\n[3b3/4] collages_clean ja existem (resume)")

                if add_labels:
                    if not state.is_complete("collage_labels"):
                        print(f"\n[3c/4] Adicionando titulo e legenda sob medida...")
                        t3c = time.perf_counter()
                        all_collages = [cp for cp in [collage_path] + special_paths if cp and os.path.isfile(cp)]
                        for cp in all_collages:
                            cimg = PILImage.open(cp)
                            cw = cimg.width
                            cimg.close()
                            cscale = max(1, cw // 2000)
                            title_font = fs(int(28 * cscale))
                            sub_font = fs(int(20 * cscale))
                            legend_at_width = os.path.join(overlays_dir, f"legend_{os.path.basename(cp)}")
                            FrameProcessor.render_legend_overlay(
                                width=cw, palette=viz_params.get("palette", ["fdfdfd", "800000"]),
                                vmin=viz_params.get("min", 0), vmax=viz_params.get("max", 1),
                                font_size=sub_font,
                                discrete_labels=viz_params.get("discrete_labels"),
                                cmap_type=viz_params.get("cmap_type", "sequential"),
                                label=viz_params.get("label", ""),
                                rgb_legend=viz_params.get("rgb_legend"),
                                legend_order=viz_params.get("legend_order"),
                                prefix_labels=viz_params.get("prefix_labels", True),
                                output_path=legend_at_width,
                            )
                            FrameProcessor.add_year_label(
                                cp, title_line1,
                                position="top_left",
                                font_size=title_font,
                                padding_top=max(130, title_font + 2 * sub_font + 60),
                                bar_color=(255, 255, 255),
                                text_color=(0, 0, 0),
                                subtitle=title_line2,
                                subtitle_size=sub_font,
                                line2=title_line3,
                                line2_size=sub_font,
                            )
                            FrameProcessor.paste_overlay_below(cp, legend_at_width)
                            FrameProcessor.add_margin(cp, 30)
                        timings["collage_labels"] = round(time.perf_counter() - t3c, 1)
                        print(f"    titulo/legenda: {timings['collage_labels']}s")
                        state.mark_complete("collage_labels")
                    else:
                        print(f"\n[3c/4] Titulo/legenda das colagens ja adicionados (resume)")
                        timings["collage_labels"] = 0

                result["collage_path"] = collage_path
                print(f"  Colagem: {collage_path}")

            if add_labels:
                if getattr(self, "progress_callback", None):
                    self.progress_callback("stage", "Finalizando frames e GIFs")
                if not state.is_complete("frames_maps"):
                    print(f"\n[3d/4] Criando frames_maps (frames completos)...")
                    t3d = time.perf_counter()
                    map_paths = []
                    for fp in clean_paths:
                        map_fp = os.path.join(frames_maps_dir, os.path.basename(fp))
                        shutil.copy2(fp, map_fp)
                        map_paths.append(map_fp)
                    FrameProcessor.batch_add_margins(map_paths, 30)
                    line2_map = {fp: (title_line3 or "") for fp in map_paths}
                    header_padding = 200 if collection_title else 140
                    FrameProcessor.batch_add_frame_headers(
                        map_paths, title_line1, line2_map,
                        line1_size=fs(54), line2_size=fs(42),
                        padding_top=header_padding, gap=10,
                        subtitle=title_line2, subtitle_size=fs(42),
                    )
                    if legend_overlay and os.path.exists(legend_overlay):
                        FrameProcessor.batch_paste_overlay_below(map_paths, legend_overlay)
                    timings["frames_maps"] = round(time.perf_counter() - t3d, 1)
                    print(f"    {len(map_paths)} frames em frames_maps/ ({timings['frames_maps']}s)")
                    state.mark_complete("frames_maps")
                else:
                    print(f"\n[3d/4] frames_maps ja existem (resume)")
                    timings["frames_maps"] = 0
                    map_paths = [
                        os.path.join(frames_maps_dir, os.path.basename(fp))
                        for fp in clean_paths
                    ]

                if not state.is_complete("gif"):
                    print(f"\n[4/4] Criando GIF principal...")
                    t5 = time.perf_counter()
                    frame_ms = self.config.get_processing_config("gif_creation").get("frame_duration", 300)
                    secs = frame_ms / 1000
                    sec_str = f"{secs:.1f}s".replace(".", "_")
                    gif_filename = f"{product_id}_{territory_id}_{sec_str}.gif"
                    gif_path = self.gif_generator.create_gif(
                        image_paths=map_paths,
                        output_dir=gifs_dir,
                        filename=gif_filename,
                        sort_frames=True,
                        viz_palette=viz_params.get("palette"),
                    )
                    timings["gif_creation"] = round(time.perf_counter() - t5, 1)
                    print(f"  ({timings['gif_creation']}s)")
                    result["gif_path"] = gif_path
                    result["status"] = "success"
                    print(f"  GIF: {gif_path}")
                    state.mark_complete("gif")
                else:
                    print(f"\n[4/4] GIF principal ja criado (resume)")
                    timings["gif_creation"] = 0
                    gif_path = self._find_existing_gif(gifs_dir, product_id, territory_id)
                    result["gif_path"] = gif_path

                if not state.is_complete("special_gifs"):
                    print(f"\n[4b/4] Criando GIFs especiais...")
                    t_sg = time.perf_counter()
                    special_modes = ["decadal", "quinzenal", "first_last", "last_six"]
                    from ..postprocessing.frame_selector import select_frames
                    for smode in special_modes:
                        selected = select_frames(map_paths, smode)
                        if len(selected) >= 2:
                            sgif_name = f"{product_id}_{territory_id}_gif_{smode}.gif"
                            self.gif_generator.create_gif(
                                image_paths=selected,
                                output_dir=gifs_dir,
                                filename=sgif_name,
                                sort_frames=True,
                                viz_palette=viz_params.get("palette"),
                            )
                    timings["special_gifs"] = round(time.perf_counter() - t_sg, 1)
                    print(f"    gifs especiais: {timings['special_gifs']}s")
                    state.mark_complete("special_gifs")
                else:
                    print(f"\n[4b/4] GIFs especiais ja criados (resume)")
                    timings["special_gifs"] = 0

            if not result.get("gif_path"):
                gif_path = self._find_existing_gif(gifs_dir, product_id, territory_id)
                result["gif_path"] = gif_path

            result["status"] = "success"
            print(f"\nSalvando metadados...")
            t6 = time.perf_counter()
            timings["total"] = round(time.perf_counter() - t_start, 1)
            metadata = self._build_metadata(
                dataset_id=dataset_id,
                product_id=product_id,
                product_info=product_info,
                territory_id=territory_id,
                territory_info=territory_info,
                viz_key=viz_key,
                viz_params=viz_params,
                gif_path=gif_path,
                frame_paths=clean_paths,
                output_dir=output_dir,
                collages_dir=collages_dir,
                timings=timings,
                vertical_dimension=vertical_dimension,
            )
            self._save_metadata_json(metadata, metadata_dir, product_id)
            self._save_run_json(metadata_dir, product_id, territory_id,
                                dataset_id, timings, t_start)
            self._save_csvs(csv_dir, output_dir, dataset_id, product_id, territory_id,
                            metadata, clean_paths, map_paths, gif_path,
                            special_paths, collage_path)
            timings["metadata_save"] = round(time.perf_counter() - t6, 1)
            result["metadata"] = metadata

            print(f"\n{'='*60}")
            print(f"Pipeline concluido com sucesso!")
            print(f"Frames: {len(frame_paths)}")
            print(f"GIF: {gif_path}")
            print(f"\nTempos:")
            for phase, secs in timings.items():
                if secs:
                    print(f"  {phase}: {secs}s")
            print(f"{'='*60}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"\n  [ERRO] Pipeline falhou: {e}")

        return result

    def _load_product_image(self, product_info: Dict[str, Any]) -> Any:
        import ee

        asset_id = product_info.get("asset", "")
        processor_name = product_info.get("processor")

        if processor_name:
            return run_processor(processor_name)
        if not asset_id:
            raise ValueError(f"Produto sem asset_id: {product_info.get('id')}")

        asset_type = product_info.get("asset_type", "image")
        mosaic = product_info.get("mosaic", False)
        mask_value = product_info.get("mask_value")

        return self.downloader.load_image(asset_id, asset_type, mosaic, mask_value)

    def _describe_processor(self, processor_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not processor_name:
            return None
        descs = {
            "build_edge_area":        {"processing": "empilhamento hierárquico de bordas (30-1000m)", "divide_by": None},
            "build_fragment_size":    {"processing": "empilhamento hierárquico de tamanhos (3-75ha)", "divide_by": None},
            "build_distance_100ha":   {"processing": "distância de isolamento para fragmentos >100ha", "divide_by": None},
            "build_distance_500ha":   {"processing": "distância de isolamento para fragmentos >500ha", "divide_by": None},
            "build_distance_1000ha":  {"processing": "distância de isolamento para fragmentos >1000ha", "divide_by": None},
            "build_secondary_vegetation_coverage": {"processing": "mod 100 (cobertura VS)", "divide_by": None, "mod_by": 100},
            "build_secondary_vegetation_age":      {"processing": "÷ 100 (idade VS)", "divide_by": 100},
            "build_fire_frequency":   {"processing": "÷ 100 (frequência de fogo col9)", "divide_by": 100},
            "build_fire_age":         {"processing": "÷ 100 (idade do fogo col9)", "divide_by": 100},
            "build_accumulated_burned_coverage":   {"processing": "mod 100 (cobertura queimada)", "divide_by": None, "mod_by": 100},
            "decode_fire_frequency_col101":        {"processing": "÷ 100, round (frequência col10.1)", "divide_by": 100},
            "decode_fire_age_col101":              {"processing": "÷ 100, round (idade fogo col10.1)", "divide_by": 100},
            "decode_secondary_vegetation_age_col101": {"processing": "÷ 100, round (idade VS col10.1)", "divide_by": 100},
            "build_secondary_vegetation_coverage_col101": {"processing": "mod 100 (cobertura VS col10.1)", "divide_by": None, "mod_by": 100},
            "build_fire_frequency_coverage_col101": {"processing": "mod 100 (cobertura freq fogo col10.1)", "divide_by": None, "mod_by": 100},
            "build_burned_natural_coverage_col101": {"processing": "mod 100, masked by freq≥1 (cobertura em áreas queimadas)", "divide_by": None},
            "build_burned_at_least_once_col101": {"processing": "÷ 100, ternary (0=não queimado, 1=recorrente, 2=primeira vez)", "divide_by": 100},
            "decode_edge_area_col101": {"processing": "reclassifica distância contínua em 8 bins (≤30 a ≤1000m), unmask=0 (cinza)", "divide_by": None},
            "decode_edge_age_col101": {"processing": "unmask(0) para fundo cinza, valores originais preservados (1-40 anos)", "divide_by": None},
            "decode_morphology_col101": {"processing": "unmask(0) para fundo cinza, valores originais preservados (1-6)", "divide_by": None},
            "decode_patch_id_col101": {"processing": "unmask(0) para fundo cinza, randomVisualizer para cores distintas por fragmento", "divide_by": None},
            "decode_patch_size_col101": {"processing": "unmask(0) para fundo cinza, valores 0-11 (tamanho do fragmento)", "divide_by": None},
            "decode_patch_size_fragments_col101": {"processing": "discretiza 0-10k ha em 10 bins + unmask(0)", "divide_by": None},
            "decode_patch_size_massifs_col101": {"processing": "discretiza >10k ha em 10 bins + unmask(0)", "divide_by": None},
            "decode_canopy_disturbance_col101": {"processing": "unmask(0) para fundo cinza, valores 0-12 (frequência de distúrbio de dossel)", "divide_by": None},
            "decode_logging_col101": {"processing": "unmask(0) para fundo cinza, binário corte seletivo", "divide_by": None},
            "build_natural_coverage_col101": {"processing": "remapa cobertura MapBiomas col10.1 para classes naturais (1=floresta, 2=não-florestal), demais=0", "divide_by": None},
        }
        return descs.get(processor_name, {"processing": processor_name, "divide_by": None})

    def _build_metadata(
        self,
        dataset_id: str,
        product_id: str,
        product_info: Dict[str, Any],
        territory_id: str,
        territory_info: Dict[str, Any],
        viz_key: str,
        viz_params: Dict[str, Any],
        gif_path: str,
        frame_paths: List[str],
        output_dir: str,
        collages_dir: str,
        timings: Optional[Dict[str, float]] = None,
        vertical_dimension: int = 2000,
    ) -> Dict[str, Any]:
        processor_name = product_info.get("processor")
        asset_id = product_info.get("asset", "")
        processor_desc = self._describe_processor(processor_name)

        def _file_size_mb(path):
            try:
                return round(os.path.getsize(path) / (1024 * 1024), 2)
            except (OSError, TypeError):
                return None

        frame_sizes = {}
        total_frames_mb = 0.0
        for fp in frame_paths:
            mb = _file_size_mb(fp)
            if mb is not None:
                frame_sizes[os.path.basename(fp)] = mb
                total_frames_mb += mb

        gif_size_mb = _file_size_mb(gif_path)

        collage_filename = f"{product_id}_{territory_id}_collage.png"
        collage_path = os.path.join(collages_dir, collage_filename)
        collage_size_mb = _file_size_mb(collage_path)

        margin_px = 30
        header_offset = 130  # padding_top(120) + gap(10)
        frame_layout = None
        if frame_paths:
            try:
                first_frame = PILImage.open(frame_paths[0])
                fw, fh = first_frame.size
                frame_layout = {
                    "map_offset_x": margin_px,
                    "map_offset_y": margin_px + header_offset,
                    "map_width": fw - 2 * margin_px,
                    "map_height": vertical_dimension,
                    "total_width": fw,
                    "total_height": fh,
                }
            except Exception:
                pass

        metadata = {
            "metadata_version": "1.1",
            "generated_at": datetime.now().isoformat(),
            "dataset": {
                "id": dataset_id,
                "description": product_info.get("dataset_description", ""),
                "source": product_info.get("dataset_source", ""),
            },
            "product": {
                "id": product_id,
                "name": product_info.get("name", product_id),
                "asset": asset_id,
                "asset_type": product_info.get("asset_type", "image"),
                "bands": product_info.get("bands", []),
                "bands_slice": product_info.get("bands_slice"),
                "temporal_range": product_info.get("temporal_range"),
            },
            "processor": {
                "name": processor_name,
                "description": processor_desc.get("processing") if processor_desc else None,
                "divide_by": processor_desc.get("divide_by") if processor_desc else None,
            },
            "visualization": {
                "key": viz_key,
                "name": viz_params.get("name", viz_key),
                "cmap_type": viz_params.get("cmap_type", "sequential"),
                "min": viz_params.get("min", 0),
                "max": viz_params.get("max", 1),
                "palette": viz_params.get("palette", []),
                "label": viz_params.get("label", ""),
                "unit": viz_params.get("unit", ""),
            },
            "territory": {
                "id": territory_id,
                "name": territory_info.get("name", territory_id),
            },
            "output": {
                "gif_path": gif_path,
                "gif_relative_path": os.path.relpath(gif_path, output_dir) if gif_path else None,
                "frames_count": len(frame_paths),
                "frames": [os.path.basename(f) for f in frame_paths],
                "frame_duration_ms": self.config.get_processing_config("gif_creation").get("frame_duration", 300),
            },
            "files": {
                "gif_size_mb": gif_size_mb,
                "collage_size_mb": collage_size_mb,
                "frames_total_mb": round(total_frames_mb, 2),
                "frames_count": len(frame_paths),
                "frames_sizes_mb": frame_sizes,
            },
        }

        if frame_layout:
            metadata["frame_layout"] = frame_layout

        if timings:
            metadata["timing"] = {
                "phases": {k: v for k, v in timings.items() if k != "total"},
                "total_seconds": timings.get("total"),
                "total_formatted": self._format_seconds(timings.get("total", 0)),
            }

        if frame_paths:
            try:
                first = PILImage.open(frame_paths[0])
                w, h = first.size
                pixels_per_frame = w * h
                total_pixels = pixels_per_frame * len(frame_paths)
                metadata["ee_estimate"] = {
                    "frame_dimensions": {"width": w, "height": h},
                    "pixels_per_frame": pixels_per_frame,
                    "total_pixels_processed": total_pixels,
                    "gee_thumbnail_requests": len(frame_paths),
                }

                tile_size = 256 * 256
                tile_equivalent = total_pixels / tile_size
                complexity_factor = self._processor_complexity(product_info.get("processor"))
                eecu = round(tile_equivalent * 0.001 * complexity_factor, 4)
                eecu_hours = round(eecu / 3600, 6)
                metadata["ee_estimate"]["tile_equivalent"] = round(tile_equivalent, 1)
                metadata["ee_estimate"]["complexity_factor"] = complexity_factor
                metadata["ee_estimate"]["estimated_eecu"] = eecu
                metadata["ee_estimate"]["estimated_eecu_hours"] = eecu_hours
            except Exception:
                pass

        return metadata

    @staticmethod
    def _processor_complexity(processor_name: Optional[str]) -> float:
        if not processor_name:
            return 1.0
        simple = {
            "decode_fire_frequency_col101",
            "decode_fire_age_col101",
            "decode_secondary_vegetation_age_col101",
            "build_fire_frequency_coverage_col101",
            "decode_edge_area_col101",
            "decode_edge_age_col101",
            "decode_morphology_col101",
            "decode_patch_id_col101",
            "decode_patch_size_col101",
            "decode_patch_size_fragments_col101",
            "decode_patch_size_massifs_col101",
        }
        moderate = {
            "build_secondary_vegetation_coverage_col101",
            "build_secondary_vegetation_coverage",
            "build_secondary_vegetation_age",
            "build_burned_natural_coverage_col101",
            "build_primary_natural_coverage_col101",
        }
        if processor_name in simple:
            return 1.0
        if processor_name in moderate:
            return 1.5
        return 2.0

    @staticmethod
    def _format_seconds(total_secs: float) -> str:
        mins = int(total_secs // 60)
        secs = int(total_secs % 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _load_existing_frames(output_dir: str, prefix: str) -> List[str]:
        import glob as glob_mod
        pattern = os.path.join(output_dir, f"{prefix}*.png")
        paths = sorted(glob_mod.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"Nenhum frame existente encontrado em {output_dir}/{prefix}*.png")
        return paths

    @staticmethod
    def _find_existing_gif(output_dir: str, product_id: str, territory_id: str) -> Optional[str]:
        import glob as glob_mod
        pattern = os.path.join(output_dir, f"{product_id}_{territory_id}_*.gif")
        gifs = sorted(glob_mod.glob(pattern))
        return gifs[-1] if gifs else None

    @staticmethod
    def _build_special_collages(
        collages_dir: str,
        product_id: str,
        territory_id: str,
        clean_paths: List[str],
        clean_dir: str,
        has_clean: bool,
        special_modes: Dict[str, Dict],
        cell_height: int,
        fs: Any,
    ) -> List[str]:
        from ..postprocessing.frame_selector import select_frames, extract_year
        created = []

        for mode, grid_cfg in special_modes.items():
            selected = select_frames(clean_paths, mode)
            if not selected:
                continue

            collage_src = [
                os.path.join(clean_dir, os.path.basename(p))
                for p in selected
            ]
            collage_src = [p for p in collage_src if os.path.isfile(p)]

            if not collage_src:
                continue

            cname = f"{product_id}_{territory_id}_collage_{mode}.png"
            try:
                from ..core.gif_generator import GIFGenerator
                gen = GIFGenerator()
                cpath = gen.create_collage(
                    image_paths=collage_src,
                    output_dir=collages_dir,
                    filename=cname,
                    cell_labels=None,
                    cell_height=cell_height,
                    grid_size=grid_cfg.get("grid_size"),
                    force_horizontal=grid_cfg.get("force_horizontal", False),
                )
                created.append(cpath)
                print(f"    {mode} grid: {os.path.basename(cpath)}")
            except Exception as e:
                print(f"    {mode} erro: {e}")
        return created

    @staticmethod
    def _save_metadata_json(metadata: Dict[str, Any], metadata_dir: str, product_id: str) -> str:
        filepath = os.path.join(metadata_dir, f"metadata_{product_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"  Metadata salvo: {filepath}")
        return filepath

    def _save_run_json(
        self, metadata_dir: str,
        product_id: str, territory_id: str, dataset_id: str,
        timings: Dict[str, float], t_start: float,
    ):
        run = {
            "dataset": dataset_id,
            "product": product_id,
            "territory": territory_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(time.perf_counter() - t_start, 1),
            "status": "success",
            "phases": {k: v for k, v in timings.items() if k not in ("total",)},
        }
        fp = os.path.join(metadata_dir, "run.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)

    def _save_csvs(
        self, csv_dir: str, output_dir: str,
        dataset_id: str, product_id: str, territory_id: str,
        metadata: Dict[str, Any], clean_paths: List[str],
        map_paths: List[str], gif_path: Optional[str],
        special_paths: List[str], collage_path: Optional[str],
    ):
        import csv
        from urllib.parse import quote

        base_url = "https://storage.googleapis.com/mapbiomas-fire/data-container"
        gcs_root = f"{base_url}/{quote(dataset_id)}/{quote(product_id)}/{quote(territory_id)}"
        prod_name = metadata.get("product", {}).get("name", product_id)
        terr_name = metadata.get("territory", {}).get("name", territory_id)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        frames_count = len(clean_paths)
        gif_rel = os.path.relpath(gif_path, output_dir) if gif_path else ""
        gif_url = f"{gcs_root}/{quote(gif_rel)}" if gif_rel else ""

        # --- product.csv (1 linha por produto) ---
        prod_row = {
            "dataset": dataset_id,
            "colecao": metadata.get("dataset", {}).get("description", ""),
            "produto_id": product_id,
            "nome_produto": prod_name,
            "territorio_id": territory_id,
            "nome_territorio": terr_name,
            "frames_count": frames_count,
            "gif_url": gif_url,
            "gif_tamanho_mb": metadata.get("files", {}).get("gif_size_mb", ""),
            "collage_url": f"{gcs_root}/{quote(product_id)}_{quote(territory_id)}_collage.png" if collage_path else "",
            "data_geracao": ts,
            "status": "success",
        }
        prod_csv = os.path.join(csv_dir, "product.csv")
        exists = os.path.isfile(prod_csv)
        with open(prod_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(prod_row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(prod_row)

        # --- frames.csv (1 linha por frame) ---
        frame_rows = []
        for cfp, mfp in zip(clean_paths, map_paths):
            year = ""
            m = re.findall(r"(\d{4})", os.path.basename(cfp))
            if m:
                year = m[-1]
            frame_rows.append({
                "dataset": dataset_id,
                "produto_id": product_id,
                "territorio_id": territory_id,
                "ano": year,
                "frame_clean": f"{gcs_root}/frames_clean/{quote(os.path.basename(cfp))}",
                "frame_map": f"{gcs_root}/frames_maps/{quote(os.path.basename(mfp))}",
            })
        if frame_rows:
            fcsv = os.path.join(csv_dir, "frames.csv")
            exists_f = os.path.isfile(fcsv)
            with open(fcsv, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
                if not exists_f:
                    w.writeheader()
                w.writerows(frame_rows)

        # --- collages.csv (1 linha por colagem) ---
        coll_rows = []
        if collage_path and os.path.isfile(collage_path):
            coll_rows.append({
                "dataset": dataset_id,
                "produto_id": product_id,
                "territorio_id": territory_id,
                "modo": "main",
                "arquivo": os.path.basename(collage_path),
                "url": f"{gcs_root}/collages_combined/{quote(os.path.basename(collage_path))}",
            })
        for sp in special_paths:
            mode = os.path.basename(sp).replace(f"{product_id}_{territory_id}_collage_", "").replace(".png", "")
            coll_rows.append({
                "dataset": dataset_id,
                "produto_id": product_id,
                "territorio_id": territory_id,
                "modo": mode,
                "arquivo": os.path.basename(sp),
                "url": f"{gcs_root}/collages_combined/{quote(os.path.basename(sp))}",
            })
        if coll_rows:
            ccsv = os.path.join(csv_dir, "collages.csv")
            exists_c = os.path.isfile(ccsv)
            with open(ccsv, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(coll_rows[0].keys()))
                if not exists_c:
                    w.writeheader()
                w.writerows(coll_rows)

    def run_batch(self, combinations: List[Tuple[str, str, str, Optional[str]]],
                  output_dir: Optional[str] = None,
                  cell_height: int = 300,
                  resume: bool = False,
                  font_scale: float = 1.0) -> List[Dict[str, Any]]:
        results = []
        for combo in combinations:
            dataset_id, product_id, territory_id = combo[:3]
            viz_key = combo[3] if len(combo) > 3 else None
            result = self.run(dataset_id, product_id, territory_id, viz_key, output_dir,
                            cell_height=cell_height, resume=resume, font_scale=font_scale)
            results.append(result)
        return results

    def list_available(self) -> Dict[str, List[str]]:
        available = {}
        for ds in self.datasets.list_datasets():
            ds_id = ds["id"]
            products = self.datasets.list_products(ds_id)
            available[ds_id] = [p["id"] for p in products]
        return available
