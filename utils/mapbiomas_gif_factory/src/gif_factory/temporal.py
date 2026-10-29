"""Resolução temporal de bandas sem depender de prefixos de produto."""
import re


def temporal_key(name):
    years = [int(x) for x in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name)]
    if not years or len(years) > 2:
        return None
    if len(years) == 2:
        if years[0] > years[1]:
            return None
        return (years[1], 0, years[0])
    year = years[0]
    match = re.search(rf"(?<!\d){year}[_-](0[1-9]|1[0-2])(?!\d)", name)
    return (year, int(match.group(1)) if match else 0, 0)


def sequence_labels(band_names, axis="time", year_offset=0):
    """Rótulos de frame sem presumir que todo produto é uma série de anos."""
    names = list(band_names)
    windows = [re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name) for name in names]
    # Todas as janelas começando no mesmo ano = série cumulativa.
    same_start = bool(windows) and all(len(w) == 2 for w in windows) and len({w[0] for w in windows}) == 1
    labels = []
    for name, years in zip(names, windows):
        depth = re.fullmatch(r"depth_(\d+)_(\d+)", name) if axis == "depth" else None
        if depth:
            start, end = int(depth.group(1)), int(depth.group(2))
            if start >= end:
                raise ValueError(f"Profundidade inválida: {name}")
            label = f"{start}–{end} cm"
        elif same_start and len(years) == 2:
            label = years[1]
        else:
            label = re.sub(r"^[a-z_]+", "", name).replace("_", "\u2192")
        if year_offset and re.fullmatch(r"\d{4}", label):
            label = str(int(label) + year_offset)
        labels.append(label)
    return labels


def resolve_bands(available, selected=None, start_year=None, end_year=None, axis="time"):
    available = list(available)
    if not available:
        raise ValueError("O asset não possui bandas disponíveis.")
    if axis == "depth":
        if start_year or end_year:
            raise ValueError("Filtro de ano não se aplica a uma sequência de profundidade.")
        names = list(selected or available)
        if len(set(names)) != len(names) or any(name not in available for name in names):
            raise ValueError("Profundidades repetidas ou não encontradas.")
        def depth_key(name):
            match = re.fullmatch(r"depth_(\d+)_(\d+)", name)
            if not match or int(match.group(1)) >= int(match.group(2)):
                raise ValueError(f"Profundidade inválida: {name}")
            return int(match.group(1)), int(match.group(2))
        return sorted(names, key=depth_key)
    if start_year and end_year and start_year > end_year:
        raise ValueError("Ano inicial maior que o ano final.")
    if selected:
        if len(set(selected)) != len(selected):
            raise ValueError("A seleção explícita contém bandas repetidas.")
        missing = [band for band in selected if band not in available]
        if missing:
            raise ValueError(f"Bandas não encontradas: {missing}")
        names = list(selected)
    else:
        names = available
    keyed = [(name, temporal_key(name)) for name in names]
    if any(key is None for _, key in keyed):
        if (len(names) > 1 and not selected) or start_year or end_year:
            raise ValueError("Não foi possível determinar a ordem temporal de todas as bandas. Informe 'bands' explicitamente e não use filtro de ano.")
        return names
    keys = [key for _, key in keyed]
    if len(set(keys)) != len(keys):
        raise ValueError("Há bandas com o mesmo período; configure uma seleção explícita sem períodos duplicados.")
    keyed.sort(key=lambda item: item[1])
    result = [name for name, key in keyed if (not start_year or key[0] >= start_year) and (not end_year or key[0] <= end_year)]
    if not result:
        raise ValueError("Nenhuma banda encontrada para o período selecionado.")
    return result
