"""Form para adicionar/ocultar colecoes no OBJ (qualquer tema, nao so fire).

Celuula opcional no notebook:
    from export_and_vectorization.collection_manager import add_collection_form
    add_collection_form()
"""

import ipywidgets as widgets
from IPython.display import display

from . import config

L = widgets.Layout


def add_collection_form():
    """Abre um form para inserir uma colecao (pais + tema + colecao + produtos)
    no `config.OBJ`. Depois de adicionar, re-execute a celula da UI para a
    navegacao refletir a nova colecao."""
    country = widgets.Dropdown(options=list(config.OBJ), value=config.COUNTRY,
                               description="País:")
    theme = widgets.Text(value=config.THEME, description="Tema:")
    collection = widgets.Text(value="collection1", description="Coleção:")

    p_name = widgets.Text(description="Produto:", layout=L(width="230px"))
    p_type = widgets.Dropdown(options=["byte", "int16", "float32"], value="byte",
                              description="Type:", layout=L(width="120px"))
    p_vec = widgets.Checkbox(description="vectorize", value=False, layout=L(width="100px"))
    p_vis = widgets.Checkbox(description="visible", value=True, layout=L(width="90px"))
    p_assetid = widgets.Text(description="AssetID:", layout=L(width="100%"))

    pending = []
    products_out = widgets.Output()

    btn_add_prod = widgets.Button(description="+ Produto", button_style="info")
    btn_commit = widgets.Button(description="Adicionar coleção", button_style="success")

    def _render_pending():
        with products_out:
            products_out.clear_output()
            if not pending:
                print("(nenhum produto pendente)")
                return
            for i, p in enumerate(pending, 1):
                print(f"  {i}. {p['product']} [{p['type']}] "
                      f"vec={p['vectorize']} vis={p['visible']}")
                print(f"     {p['assetid']}")

    def _add_prod(_):
        name = p_name.value.strip()
        assetid = p_assetid.value.strip()
        if not name or not assetid:
            print("Produto e AssetID são obrigatórios.")
            return
        pending.append({
            "product": name,
            "assetid": assetid,
            "type": p_type.value,
            "vectorize": bool(p_vec.value),
            "visible": bool(p_vis.value),
        })
        p_name.value = ""
        p_assetid.value = ""
        p_vec.value = False
        p_vis.value = True
        _render_pending()

    def _commit(_):
        if not pending:
            print("Adicione ao menos um produto antes de salvar.")
            return
        config.add_collection(country.value, theme.value.strip(),
                              collection.value.strip(), list(pending))
        n = len(pending)
        pending.clear()
        _render_pending()
        print(f"[OK] Coleção '{collection.value}' adicionada em "
              f"{country.value}/{theme.value} ({n} produtos).")
        print("Re-execute a célula da UI para a navegação refletir a nova coleção.")

    btn_add_prod.on_click(_add_prod)
    btn_commit.on_click(_commit)

    form = widgets.VBox([
        widgets.HBox([country, theme, collection], layout=L(gap="8px")),
        widgets.HBox([p_name, p_type, p_vec, p_vis], layout=L(gap="8px")),
        p_assetid,
        widgets.HBox([btn_add_prod, btn_commit], layout=L(gap="8px")),
        products_out,
    ])
    display(form)
    return form


def set_visible(country, theme, collection, product, visible):
    """Oculta (visible=False) ou mostra um produto no OBJ."""
    ok = config.set_product_visible(country, theme, collection, product, visible)
    if ok:
        print(f"[OK] {product} {'ocultado' if not visible else 'visível'} em "
              f"{country}/{theme}/{collection}. Re-execute a UI.")
    else:
        print(f"[WARN] Produto '{product}' não encontrado em {country}/{theme}/{collection}.")
    return ok
