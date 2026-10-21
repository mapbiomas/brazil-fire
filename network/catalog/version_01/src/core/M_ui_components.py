"""
Componentes de UI Base (Ipywidgets)
MapBiomas Fire Monitor Pipeline - ASCII Version
"""
import ipywidgets as widgets
import time
from IPython.display import display, clear_output
from M_lang import L

THEME = {
    'ERROR': '#d32f2f',
    'WARNING': '#e67e22',
    'SUCCESS': '#27ae60',
    'INFO': '#3498db',
    'DEFAULT': '#333333',
}

def cell_log(message, type="info"):
    """Exibe mensagem colorida na célula atual (não dentro de um PipelineStepUI)."""
    color = THEME.get(type.upper(), THEME['DEFAULT'])
    display(widgets.HTML(f"<span style='color:{color}'>[{type.upper()}] {message}</span>"))

class PipelineStepUI:
    """
    Componente base para englobar visualmente as etapas do pipeline.
    """
    
    def __init__(self, title, description=""):
        self.title = title
        self.description = description
        
        # Loader CSS
        self.loader_html = widgets.HTML(
            value=f'''
            <div id="mfm-loader" style="display:none; align-items:center; margin-left:15px;">
                <div class="spinner"></div>
                <span style="margin-left:8px; font-size:11px; color:#666;">{L.PROCESSING}</span>
            </div>
            <style>
            .spinner {{
                border: 3px solid #f3f3f3;
                border-top: 3px solid #3498db;
                border-radius: 50%;
                width: 16px;
                height: 16px;
                animation: mfm-spin 1s linear infinite;
                display: inline-block;
                vertical-align: middle;
            }}
            @keyframes mfm-spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            '''
        )
        
        self.header_title = widgets.HTML(
            value=f"<h3 style='margin-bottom:0; display:inline-block;'>{self.title}</h3>",
            layout=widgets.Layout(margin='0')
        )
        
        self.header_box = widgets.HBox([self.header_title, self.loader_html], layout=widgets.Layout(align_items='center'))
        
        self.header_desc = widgets.HTML(
            value=f"<p style='color:#666; margin-top:5px;'>{self.description}</p>"
        )
        
        self.main_area = widgets.VBox()
        self.log_output = widgets.Output() 
        
        self.container = widgets.VBox([
            self.header_box,
            self.header_desc,
            self.main_area,
            self.log_output
        ], layout=widgets.Layout(
            border='1px solid #ccc',
            padding='10px',
            border_radius='5px',
            margin='10px 0'
        ))
        
    def display(self):
        """Renderiza o componente no notebook."""
        display(self.container)
        
    def show_loader(self, message=None):
        """Mostra o spinner de carregamento."""
        if message is None:
            message = L.PROCESSING
        self.loader_html.value = self.loader_html.value.replace('display:none', 'display:flex')
        if message:
            import re
            self.loader_html.value = re.sub(r'<span.*?>.*?</span>', f'<span style="margin-left:8px; font-size:11px; color:#666;">{message}</span>', self.loader_html.value)

    def hide_loader(self):
        """Esconde o spinner de carregamento."""
        self.loader_html.value = self.loader_html.value.replace('display:flex', 'display:none')

    def update_status(self, message, type=None):
        """Atualiza a mensagem de status secundária no próprio loader."""
        if not message:
            self.hide_loader()
            return
            
        # Garante que o loader está visível
        if 'display:none' in self.loader_html.value:
            self.show_loader(message)
        else:
            # Apenas troca o texto
            import re
            self.loader_html.value = re.sub(r'<span.*?>.*?</span>', f'<span style="margin-left:8px; font-size:11px; color:#666;">{message}</span>', self.loader_html.value)

    def log(self, message, type="info"):
        """Adiciona uma mensagem textualmente formatada no Output de Logs."""
        color = THEME.get(type.upper(), THEME['DEFAULT'])
        with self.log_output:
            display(widgets.HTML(f"<span style='color:{color}'>[{type.upper()}] {message}</span>"))
            
    def clear_logs(self):
        """Limpa apenas o terminal de logs do componente."""
        self.log_output.clear_output()
        
    def clear_main(self):
        """Limpa apenas a area primaria de renderizacao de controles."""
        self.main_area.children = []

    @staticmethod
    def get_status_css():
        """Retorna o CSS padrao para as celulas de status da matriz."""
        return widgets.HTML('''<style>
            .mfm-ok   { background:#d4edda !important; border:1px solid #c3e6cb !important; }
            .mfm-run  { background:#fff3cd !important; border:1px solid #ffeaa8 !important; }
            .mfm-null { background:#f8f9fa !important; border:1px solid #dee2e6 !important; }
            .mfm-hdr  { font-size:11px; font-weight:700; color:#495057; text-align:center; }
        </style>''')

    @staticmethod
    def make_status_cell(chk, status_text, css_class, width='80px'):
        """Cria um HBox formatado para representar uma celula de status."""
        status_html = widgets.HTML(
            f'<span style="font-size:10px;font-weight:700;color:#155724">{status_text}</span>' 
            if css_class == 'mfm-ok' else
            f'<span style="font-size:10px;font-weight:700;color:#856404">{status_text}</span>'
            if css_class == 'mfm-run' else
            f'<span style="font-size:10px;color:#adb5bd">{status_text}</span>',
            layout=widgets.Layout(width='32px')
        )
        
        cell = widgets.HBox(
            [chk, status_html], 
            layout=widgets.Layout(
                width=width, 
                min_height='34px', 
                justify_content='center', 
                align_items='center', 
                padding='0', 
                overflow='hidden', 
                margin='1px'
            )
        )
        cell.add_class(css_class)
        return cell


# ---------------------------------------------------------------------------
# Helpers de confirmacao inline e spinner
# ---------------------------------------------------------------------------

_SPINNER_STYLE = """
<style>
.mfm-loader-mini {
    border: 2px solid #f3f3f3;
    border-top: 2px solid #3498db;
    border-radius: 50%;
    width: 14px;
    height: 14px;
    animation: mfm-spin 0.8s linear infinite;
    display: inline-block;
    vertical-align: middle;
}
@keyframes mfm-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>"""


def make_spinner(msg=None):
    """Retorna um widget HTML com spinner animado e mensagem."""
    if msg is None:
        msg = L.LOADING
    return widgets.HTML(f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            <div class="mfm-loader-mini"></div>
            <span style="color: #666; font-size: 11px; font-weight: bold;">{msg}</span>
        </div>
        {_SPINNER_STYLE}
    """)


def inline_confirm(btn, on_confirm, on_cancel=None, container=None):
    """Substitui o botao *btn* por [ Voltar ] [ OK ] no mesmo container.

    Parametros
    ----------
    btn : widgets.Button
        Botao a ser substituido.
    on_confirm : callable
        Executada quando o usuario clica OK (sem argumentos).
    on_cancel : callable, opcional
        Executada quando o usuario clica Voltar.
    container : widgets.HBox ou VBox, opcional
        Container que contem o botao. Omissao significa usar btn.parent.
    """
    if container is None:
        container = getattr(btn, 'parent', None)
    if container is None:
        return
    children = list(container.children)
    try:
        idx = children.index(btn)
    except ValueError:
        return

    btn_voltar = widgets.Button(
        description=L.BACK,
        button_style='',
        layout=widgets.Layout(width='70px', height='26px', padding='0', font_size='11px'))
    btn_ok = widgets.Button(
        description=L.OK,
        button_style='danger',
        layout=widgets.Layout(width='50px', height='26px', padding='0', font_size='11px'))

    confirm_box = widgets.HBox([btn_voltar, btn_ok],
                                layout=widgets.Layout(align_items='center', gap='3px'))

    spinner = make_spinner(msg=L.DELETING)

    def _restore(_):
        if on_cancel:
            on_cancel()
        container.children = tuple(children[:idx] + [btn] + children[idx + 1:])

    def _do_confirm(_):
        container.children = tuple(children[:idx] + [spinner] + children[idx + 1:])
        on_confirm()

    btn_voltar.on_click(_restore)
    btn_ok.on_click(_do_confirm)
    container.children = tuple(children[:idx] + [confirm_box] + children[idx + 1:])


# ---------------------------------------------------------------------------
# Atalho Layout (definir no modulo, nao localmente nos metodos)
# ---------------------------------------------------------------------------
Layout = widgets.Layout


# ---------------------------------------------------------------------------
# make_empty_state — placeholder padrao "sem dados"
# ---------------------------------------------------------------------------
def make_empty_state(message, padding="20px"):
    """Retorna widget HTML com mensagem de estado vazio."""
    return widgets.HTML(
        f"<div style='padding:{padding}; text-align:center; color:#999; border:1px dashed #ccc;'>"
        f"<i>{message}</i></div>"
    )


# ---------------------------------------------------------------------------
# flash_output — limpa e exibe conteudo em um Output
# ---------------------------------------------------------------------------
def flash_output(output_widget, content, as_html=True):
    """Limpa e exibe conteudo em um widget Output."""
    from IPython.display import HTML as _HTML
    with output_widget:
        clear_output()
        if as_html:
            display(_HTML(content))
        else:
            display(content)


# ---------------------------------------------------------------------------
# make_sync_button — botao com estado "Sincronizando..." + disable
# ---------------------------------------------------------------------------
def make_sync_button(description, icon, on_click_callback, width='220px', height='30px', button_style='success', ui=None):
    """Retorna (btn, output) onde btn desabilita durante callback.
    
    Se ui for um PipelineStepUI, chama ui.show_loader() antes e ui.hide_loader() depois.
    """
    btn = widgets.Button(description=description, icon=icon, button_style=button_style,
                         layout=Layout(width=width, height=height))
    out = widgets.Output()

    def _handler(b):
        if ui:
            ui.show_loader(L.SYNCING)
        btn.description = L.SYNCING
        btn.disabled = True
        try:
            with out:
                clear_output()
                on_click_callback()
        finally:
            btn.description = description
            btn.disabled = False
            if ui:
                ui.hide_loader()

    btn.on_click(_handler)
    return btn, out


# ---------------------------------------------------------------------------
# make_refresh_button — botao icone com spinner inline, nao precisa de PipelineStepUI
# ---------------------------------------------------------------------------
def make_refresh_button(icon, on_click_callback, description='', tooltip='', width='32px', button_style='info'):
    """Retorna (container_hbox, btn, spinner).

    O spinner aparece ao lado do botao durante a execucao do callback.
    Usar 'container' no lugar do btn no layout, e 'btn' se precisar referenciar o botao.

    description: texto do botao (opcional, para botoes maiores com label).
    """
    btn = widgets.Button(description=description, icon=icon, layout=Layout(width=width),
                         button_style=button_style, tooltip=tooltip)
    spinner = make_spinner()
    spinner.layout.display = 'none'
    container = widgets.HBox([btn, spinner], layout=Layout(align_items='center'))

    def _handler(b):
        btn.disabled = True
        spinner.layout.display = 'block'
        try:
            on_click_callback()
        finally:
            spinner.layout.display = 'none'
            btn.disabled = False

    btn.on_click(_handler)
    return container, btn, spinner


# ---------------------------------------------------------------------------
# make_select_all_none — par Todos / Limpiar
# ---------------------------------------------------------------------------
def make_select_all_none(on_all=None, on_none=None, width='70px'):
    """Retorna (btn_all, btn_none, hbox)."""
    btn_all = widgets.Button(description=L.ALL, icon='check-square',
                             layout=Layout(width=width), button_style='info')
    btn_none = widgets.Button(description=L.CLEAR, icon='square-o',
                              layout=Layout(width=width), button_style='warning')
    if on_all:
        btn_all.on_click(on_all)
    if on_none:
        btn_none.on_click(on_none)
    hbox = widgets.HBox([btn_all, btn_none])
    return btn_all, btn_none, hbox


# ---------------------------------------------------------------------------
# make_search_box — campo de texto com placeholder
# ---------------------------------------------------------------------------
def make_search_box(placeholder, on_change=None):
    """Retorna widgets.Text com placeholder. on_change recebe o valor."""
    text = widgets.Text(description=L.SEARCH + ':', placeholder=placeholder,
                        layout=Layout(width='100%'))
    if on_change:
        text.observe(lambda change: on_change(change['new']), names='value')
    return text


# ---------------------------------------------------------------------------
# build_thumbnail_column — coluna esquerda com thumb 128px + spacer
# ---------------------------------------------------------------------------
def build_thumbnail_column(thumb_b64, width='128px'):
    """Cria VBox com thumbnail (ou placeholder) e spacer vertical."""
    if thumb_b64:
        thumb_el = widgets.HTML(
            f'<img src="data:image/png;base64,{thumb_b64}" '
            f'style="width:128px;height:128px;object-fit:cover;border-radius:8px;border:1px solid #d1d5db;">',
            layout=Layout(width=width, margin='0'))
    else:
        thumb_el = widgets.HTML(
            '<div style="width:128px;height:128px;background:#f1f5f9;border-radius:8px;border:1px solid #d1d5db;"></div>',
            layout=Layout(width=width, margin='0'))
    return widgets.VBox([
        thumb_el,
        widgets.Box([], layout=Layout(flex='1')),
    ], layout=Layout(width=width, align_self='flex-start'))


# ---------------------------------------------------------------------------
# make_task_badges — badges roxos para nomes de tarefa
# ---------------------------------------------------------------------------
def make_task_badges(task_names):
    """Gera HTML com badges para nomes de tarefa."""
    names = sorted(set(n for n in task_names if n))
    if not names:
        return ''
    style = ('display:inline-block;background:#f3e8ff;color:#7c3aed;font-size:10px;'
             'padding:2px 8px;border-radius:10px;margin:2px 3px;border:1px solid #d8b4fe;')
    return ''.join(f'<span style="{style}">{t}</span>' for t in names)


# ---------------------------------------------------------------------------
# make_card_body — HBox padronizado para cards (thumb + conteudo)
# ---------------------------------------------------------------------------
def make_card_body(left_col, right_col, border_color='#e2e8f0', background='#ffffff'):
    """Envolve colunas esquerda/direita em um card HBox padronizado."""
    return widgets.HBox([left_col, right_col],
        layout=Layout(padding='0', border=f'1px solid {border_color}', border_radius='8px',
                      margin='6px 0', background=background,
                      box_shadow='0 1px 3px rgba(0,0,0,0.08)',
                      align_items='stretch'))


# ---------------------------------------------------------------------------
# _fmt_time — formata segundos para hh:mm:ss
# ---------------------------------------------------------------------------
def _fmt_time(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressTracker:
    """Tracks progress with estimated remaining time.

    Usage:
        p = ProgressTracker(total=10, description=\"Example\")
        for item in items:
            process(item)
            p.step()
            print(p.summary())
    """
    def __init__(self, total, description="Progress"):
        self.total = total
        self.completed = 0
        self.description = description
        self._start = time.time()

    def step(self, n=1):
        self.completed += n

    def passo(self, n=1):
        self.step(n)

    @property
    def remaining(self):
        if self.completed == 0:
            return 0
        elapsed = time.time() - self._start
        rate = self.completed / max(elapsed, 0.001)
        return (self.total - self.completed) / max(rate, 0.001)

    def summary(self):
        elapsed = _fmt_time(time.time() - self._start)
        remaining = _fmt_time(self.remaining)
        return f"[{self.completed}/{self.total}] elapsed {elapsed} remaining ~{remaining}"
