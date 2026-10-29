import os
import time
import requests
import ee
from typing import Any, Dict, List, Optional, Tuple
from ..config import ConfigLoader
from ..utils.file_utils import ensure_dir


class EEDownloader:
    """Download de imagens do Earth Engine com geração de thumbnails."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.vertical_dimension = (
            config.get_processing_config("image_download")
            .get("vertical_dimension", 2500)
        )
        self._ee_initialized = False

    def ensure_ee_initialized(self):
        if not self._ee_initialized:
            try:
                ee.Initialize(project=self.config.ee_project_id)
            except ee.EEException as e:
                print(f"EE não inicializado automaticamente: {e}")
                print("Execute: python main.py --auth")
                raise
            self._ee_initialized = True

    def load_image(self, asset_id: str, asset_type: str = "image",
                   mosaic: bool = False, mask_value: Optional[int] = None) -> ee.Image:
        self.ensure_ee_initialized()
        if asset_type == "image_collection" or mosaic:
            collection = ee.ImageCollection(asset_id)
            image = collection.mosaic()
        else:
            image = ee.Image(asset_id)
        if mask_value is not None:
            image = image.updateMask(image.neq(mask_value))
        return image

    def get_band_names(self, image: ee.Image) -> List[str]:
        return image.bandNames().getInfo()

    def download_frames(
        self,
        image: ee.Image,
        band_names: List[str],
        viz_params: Dict[str, Any],
        region_fc: ee.FeatureCollection,
        output_dir: str,
        prefix: str = "",
        add_labels: bool = False,
        overlay_fc: Optional[ee.FeatureCollection] = None,
    ) -> List[str]:
        ensure_dir(output_dir)
        downloaded = []

        def notify(kind, *args):
            callback = getattr(self, "progress_callback", None)
            if callback:
                try:
                    callback(kind, *args)
                except Exception as exc:
                    print(f"  [AVISO] Não foi possível atualizar progresso: {exc}")

        for index, band_name in enumerate(band_names, 1):
            notify("frame_start", index, len(band_names), band_name)
            filename = f"{prefix}{band_name}.png"
            filepath = os.path.join(output_dir, filename)
            max_attempts = 3
            frame_started = time.monotonic()

            for attempt in range(1, max_attempts + 1):
                try:
                    url = self._get_thumb_url(image, band_name, viz_params, region_fc, overlay_fc)

                    response = requests.get(url, stream=True, timeout=120)
                    response.raise_for_status()

                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    downloaded.append(filepath)
                    notify("frame_done", index, len(band_names), band_name, time.monotonic() - frame_started)
                    print(f"  [OK] {filename}")
                    break

                except Exception as e:
                    if attempt < max_attempts:
                        wait = 2 ** attempt
                        notify("frame_retry", index, len(band_names), band_name, attempt, max_attempts, wait, str(e))
                        print(f"  [RETRY {attempt}/{max_attempts}] {band_name}: {e} (esperando {wait}s)")
                        time.sleep(wait)
                    else:
                        notify("frame_failed", index, len(band_names), band_name, str(e))
                        print(f"  [FALHA] {band_name} apos {max_attempts} tentativas: {e}")

        return downloaded

    def _get_thumb_url(
        self,
        image: ee.Image,
        band_name: str,
        viz_params: Dict[str, Any],
        region: ee.FeatureCollection,
        overlay_fc: Optional[ee.FeatureCollection] = None,
    ) -> str:
        if viz_params.get("random_viz", False):
            band = image.select(band_name)
            rand = band.randomVisualizer().select([0, 1, 2])
            gray = ee.Image.constant([128, 128, 128]).toUint8()
            visualized = (
                rand.where(band.eq(0), gray)
                .updateMask(ee.Image().paint(region, 0).eq(0))
            )
        else:
            palette = list(viz_params.get("palette", ["ffffff", "000000"]))
            vmin = viz_params.get("min", 0)
            nodata_color = viz_params.get("nodata_color")
            nodata_value = viz_params.get("nodata_value")
            if nodata_color and nodata_value is None:
                # Reserva uma cor fixa para o valor 0 (fora da faixa da legenda)
                vmin = vmin - 1
                palette = [nodata_color] + palette
            vis = {
                "min": vmin,
                "max": viz_params.get("max", 1),
                "palette": palette,
                "bands": [band_name],
            }
            visualized = image.select(band_name).unmask().visualize(**vis)
            if nodata_color and nodata_value is not None:
                # Pinta exatamente o valor de nodata, sem afetar a faixa da legenda
                hexc = str(nodata_color).lstrip("#")
                fill = ee.Image.constant(
                    [int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)]
                )
                visualized = visualized.where(
                    image.select(band_name).unmask().eq(nodata_value), fill
                )
            visualized = visualized.updateMask(
                ee.Image().paint(region, 0).eq(0)
            ).blend(ee.Image().paint(region, "vazio", 1))

        if overlay_fc is not None:
            boundaries = ee.Image().byte().paint(featureCollection=overlay_fc, color=1, width=1)
            visualized = visualized.where(boundaries, 0)

        url = visualized.getThumbURL({
            "dimensions": str(self.vertical_dimension),
            "region": region.geometry().bounds(),
        })
        return url

    def download_band(
        self,
        image: ee.Image,
        band_name: str,
        viz_params: Dict[str, Any],
        region: ee.FeatureCollection,
        output_path: str,
    ) -> str:
        url = self._get_thumb_url(image, band_name, viz_params, region)
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return output_path

    def download_rgb_frames(
        self,
        image: ee.Image,
        band_groups: List[tuple],
        viz_params: Dict[str, Any],
        region_fc: ee.FeatureCollection,
        output_dir: str,
        prefix: str = "",
        overlay_fc: Optional[ee.FeatureCollection] = None,
    ) -> List[str]:
        ensure_dir(output_dir)
        downloaded = []
        for year_label, band_list in band_groups:
            filename = f"{prefix}{year_label}.png"
            filepath = os.path.join(output_dir, filename)
            max_attempts = 3

            for attempt in range(1, max_attempts + 1):
                try:
                    vis = {
                        "min": viz_params.get("min", 0),
                        "max": viz_params.get("max", 1),
                        "bands": band_list,
                    }
                    visualized = (
                        image.select(band_list)
                        .unmask()
                        .visualize(**vis)
                        .updateMask(ee.Image().paint(region_fc, 0).eq(0))
                        .blend(ee.Image().paint(region_fc, "vazio", 1))
                    )
                    if overlay_fc is not None:
                        boundaries = ee.Image().byte().paint(featureCollection=overlay_fc, color=1, width=1)
                        visualized = visualized.where(boundaries, 0)

                    url = visualized.getThumbURL({
                        "dimensions": str(self.vertical_dimension),
                        "region": region_fc.geometry().bounds(),
                    })
                    response = requests.get(url, stream=True, timeout=120)
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    downloaded.append(filepath)
                    print(f"  [OK] {filename}")
                    break

                except Exception as e:
                    if attempt < max_attempts:
                        wait = 2 ** attempt
                        print(f"  [RETRY {attempt}/{max_attempts}] {year_label}: {e} (esperando {wait}s)")
                        time.sleep(wait)
                    else:
                        print(f"  [FALHA] {year_label} apos {max_attempts} tentativas: {e}")

        return downloaded
