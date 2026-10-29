"""Seleção de frames por modos predefinidos (compartilhado entre catálogos e collages especiais)."""

import os
import re
from typing import List, Optional

FRAME_MODES = ("all", "decadal", "quinzenal", "first_last", "collage", "last_six")


def select_frames(frame_paths: List[str], mode: str) -> List[str]:
    sorted_paths = sorted(frame_paths)

    if mode == "all":
        return sorted_paths

    if mode == "decadal":
        targets = {1995, 2005, 2015, 2025}
        return _filter_by_year(sorted_paths, targets)

    if mode == "quinzenal":
        targets = {1995, 2010, 2025}
        return _filter_by_year(sorted_paths, targets)

    if mode == "first_last":
        if len(sorted_paths) <= 2:
            return sorted_paths
        return [sorted_paths[0], sorted_paths[-1]]

    if mode == "collage":
        return sorted_paths[:1] if sorted_paths else []

    if mode == "last_six":
        if len(sorted_paths) <= 6:
            return sorted_paths
        return sorted_paths[-6:]

    return sorted_paths


def _filter_by_year(sorted_paths: List[str], targets: set) -> List[str]:
    selected = []
    for fp in sorted_paths:
        basename = os.path.splitext(os.path.basename(fp))[0]
        matches = re.findall(r"(\d{4})", basename)
        if matches:
            year = int(matches[-1])
            if year in targets:
                selected.append(fp)
    return selected if selected else sorted_paths[:len(targets)]


def extract_year(filepath: str) -> Optional[int]:
    basename = os.path.splitext(os.path.basename(filepath))[0]
    matches = re.findall(r"(\d{4})", basename)
    return int(matches[-1]) if matches else None
