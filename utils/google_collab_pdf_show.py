import os
import subprocess
import sys
from IPython.display import clear_output, display, Image as IPyImage
from PIL import Image as PILImage
import ipywidgets as widgets
from IPython.display import HTML # Importing HTML from IPython.display


# Verificação e instalação da biblioteca fitz (PyMuPDF)
try:
    import fitz  # PyMuPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    clear_output(wait=True)  # Limpa a saída após a instalação
    import fitz  # Tentar importar novamente após a instalação

# Função para criar uma pasta específica, se não existir
def ensure_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def display_pdf_viewer(pdf_path, external_link=None):
    """
    Displays a PDF viewer with navigation controls inside a Jupyter/Colab notebook.

    Parameters:
    - pdf_path (str): Path to the PDF file to be displayed.
    - external_link (str, optional): An optional URL link for external access to the presentation.
    """
    
    # Definir pasta de saída para armazenar as imagens
    output_dir = "pdf_pages"
    ensure_output_directory(output_dir)

    # Abrir o PDF
    doc = fitz.open(pdf_path)

    # Número total de páginas no PDF
    total_pages = doc.page_count

    # Página atual (inicializa em 0)
    current_page = 0

    # Função para exibir uma página específica
    def display_page(page_number):
        page = doc.load_page(page_number)  # Carregar a página
        pix = page.get_pixmap(dpi=300)  # Renderizar a página com alta resolução (300 DPI)
        output = os.path.join(output_dir, f"page_{page_number}.png")
        pix.save(output)  # Salvar a imagem na pasta de saída

        # Exibir a imagem diretamente, com resolução maior
        with open(output, "rb") as file:
            img_data = file.read()
        display(IPyImage(img_data))  # Exibe a imagem na célula do notebook

    # Função para atualizar o status da página
    def update_status():
        page_info.value = f"Page {current_page + 1} of {total_pages}"

    # Função para ir para a próxima página
    def next_page(b):
        nonlocal current_page
        if current_page < total_pages - 1:
            current_page += 1
            clear_output(wait=True)  # Limpar a saída anterior
            display_controls()  # Exibir botões e informações da página
            display_page(current_page)  # Exibir a próxima página
            update_status()

    # Função para ir para a página anterior
    def prev_page(b):
        nonlocal current_page
        if current_page > 0:
            current_page -= 1
            clear_output(wait=True)  # Limpar a saída anterior
            display_controls()  # Exibir botões e informações da página
            display_page(current_page)  # Exibir a página anterior
            update_status()

    # Função para exibir botões de navegação e informações da página
    def display_controls():
        next_button = widgets.Button(description="Next", button_style='success', style={'font_weight': 'bold', 'font_size': '20px'})
        prev_button = widgets.Button(description="Previous", button_style='danger', style={'font_weight': 'bold', 'font_size': '20px'})

        # Conectar botões às suas respectivas funções
        next_button.on_click(next_page)
        prev_button.on_click(prev_page)

        # Exibir as informações da página e botões lado a lado
        controls = widgets.HBox([prev_button, next_button], layout=widgets.Layout(justify_content='center', margin='10px'))
        display(controls)

        # Mostrar informações da página atual
        display(page_info)

    # Caixa de texto para mostrar o status da página atual
    page_info = widgets.Label(value=f"Page {current_page + 1} of {total_pages}")

    # Exibir link externo, se fornecido
    if external_link:
        display(HTML(f'<a href="{external_link}" target="_blank">Access the presentation</a>'))

    # Exibir a primeira página, controles, e inicializar o status
    display_controls()
    display_page(current_page)
    update_status()
