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


def resolve_bands(available, selected=None, start_year=None, end_year=None):
    available = list(available)
    if not available:
        raise ValueError("O asset não possui bandas disponíveis.")
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
