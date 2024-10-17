import subprocess
import sys
from IPython.display import clear_output
from PIL import Image as PILImage


# Verificação e instalação da biblioteca fitz (PyMuPDF)
try:
    import fitz  # PyMuPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    clear_output(wait=True)  # Limpa a saída após a instalação
    import fitz  # Tentar importar novamente após a instalação

# Verificação e instalação da biblioteca PIL (Python Imaging Library)
try:
    from PIL import Image
    from IPython.display import Image, display
    from PIL import Image as PILImage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    clear_output(wait=True)  # Limpa a saída após a instalação
    from IPython.display import Image, display
    from PIL import Image as PILImage

def display_pdf_viewer(pdf_path, external_link=None):
    """
    Displays a PDF viewer with navigation controls inside a Jupyter/Colab notebook.

    Parameters:
    - pdf_path (str): Path to the PDF file to be displayed.
    - external_link (str, optional): An optional URL link for external access to the presentation.

    The viewer includes "Next" and "Previous" buttons for page navigation, a text box indicating the current page and 
    total number of pages, and optionally displays a link to access the PDF externally.
    """

    # Open the PDF
    doc = fitz.open(pdf_path)

    # Total number of pages in the PDF
    total_pages = doc.page_count

    # Initialize the current page (starts from 0)
    current_page = 0

    # Function to display a specific page
    def display_page(page_number):
        page = doc.load_page(page_number)  # Load the page
        pix = page.get_pixmap()  # Render the page as an image
        output = f"page_{page_number}.png"
        pix.save(output)  # Save the image
        
        # Redimensionar a imagem usando PIL
        img = PILImage.open(output)
        img_resized = img.resize((800, int(img.height * 800 / img.width)))  # Ajustar largura para 800px
        img_resized.save(output)  # Salva a imagem redimensionada

        # Exibir a imagem redimensionada
        with open(output, "rb") as file:
            img_data = file.read()
        display(IPyImage(img_data))


    # Function to update the page number and text box
    def update_status():
        page_info.value = f"Page {current_page + 1} of {total_pages}"

    # Function to go to the next page
    def next_page(b):
        nonlocal current_page
        if current_page < total_pages - 1:
            current_page += 1
            clear_output(wait=True)  # Clear the previous output
            display_controls()  # Display buttons and page info
            display_page(current_page)  # Display the next page
            update_status()

    # Function to go to the previous page
    def prev_page(b):
        nonlocal current_page
        if current_page > 0:
            current_page -= 1
            clear_output(wait=True)  # Clear the previous output
            display_controls()  # Display buttons and page info
            display_page(current_page)  # Display the previous page
            update_status()

    # Function to display navigation buttons and page info
    def display_controls():
        next_button = widgets.Button(description="Next", button_style='success', style={'font_weight': 'bold', 'font_size': '20px'})
        prev_button = widgets.Button(description="Previous", button_style='danger', style={'font_weight': 'bold', 'font_size': '20px'})
        
        # Connect buttons to their respective functions
        next_button.on_click(next_page)
        prev_button.on_click(prev_page)

        # Display the page info and buttons side by side, centered
        controls = widgets.HBox([prev_button, next_button], layout=widgets.Layout(justify_content='center', margin='10px'))
        display(controls)
        
        # Show current page info
        display(page_info)

    # Text box to show the current page status (e.g., "Page 1 of 10")
    page_info = widgets.Label(value=f"Page {current_page + 1} of {total_pages}")

    # Display an external link if provided
    if external_link:
        display(HTML(f'<a href="{external_link}" target="_blank">Access the presentation</a>'))

    # Display the first page, controls, and initialize the status
    display_controls()
    display_page(current_page)
    update_status()

# # Example usage:
# pdf_path = '/content/brazil-fire/network/Entrenamiento_de_monitoreo_de_cicatrices_de_fuego_en_regiones_de_la_red_MapBiomas.pdf'
# display_pdf_viewer(pdf_path, external_link="https://github.com/mapbiomas/brazil-fire/blob/main/network/Entrenamiento%20de%20monitoreo%20de%20cicatrices%20de%20fuego%20en%20regiones%20de%20la%20red%20MapBiomas.pdf")
