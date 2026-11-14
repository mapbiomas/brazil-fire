"""Selecao multi-painel compartilhada pelas etapas do pipeline.

Cada etapa age sobre a selecao da grid, agregando itens de TODOS os
paineis (pais/tema/colecao/produto). Sem condicionais entre etapas:
a ordem de execucao e apenas uma sugestao de fluxo.
"""

from . import config


def collect_items(ui):
    """Itens selecionados em TODOS os paineis; fallback: painel ativo."""
    getter = getattr(ui, "get_selected_items", None)
    if callable(getter):
        items = getter() or []
        if items:
            return items
    units = ui.get_selected_units()
    ctx = config.processing_context()
    return [{"country": ctx["country"], "theme": ctx["theme"],
             "collection": ctx["collection"], "product": ctx["product"],
             "unit": u} for u in units]


def group_items(items):
    """Agrupa itens por chave de contexto (country, theme, collection, product)."""
    groups = {}
    for it in items:
        key = (it.get("country"), it.get("theme"),
               it.get("collection"), it.get("product"))
        groups.setdefault(key, []).append(it.get("unit"))
    return groups


def build_jobs(items):
    """Agrupa itens por contexto e devolve [(unit, ctx)] + ctxs por chave."""
    groups = group_items(items)
    seen_ctx = {key: config.processing_context(*key) for key in groups}
    jobs = []
    for key, units in groups.items():
        for unit in units:
            jobs.append((unit, seen_ctx[key]))
    return jobs, seen_ctx


def sync_affected(ui, contexts):
    """Sincroniza apenas os paineis afetados pelos contextos processados."""
    items = [{"country": c, "theme": t, "collection": co, "product": p}
             for (c, t, co, p) in contexts]
    syncer = getattr(ui, "sync_contexts", None)
    if callable(syncer):
        syncer(items)
    else:
        try:
            ui.sync()
        except Exception:
            pass
