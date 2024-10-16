import gcsfs
import ipywidgets as widgets  # Library for creating interactive widgets in Jupyter notebooks
from IPython.display import display, HTML, clear_output  # Utilities for displaying content in Jupyter notebooks
from ipywidgets import VBox, HBox  # Containers for arranging widgets in vertical or horizontal boxes

# 3.1 Interface para explorar arquivos no Google Cloud Storage (opcional)
fs = gcsfs.GCSFileSystem(project=bucket_name)

# Definir o caminho inicial para os países
pasta_base = 'mapbiomas-fire/sudamerica/'

# Função para listar os países
def listar_paises(pasta_base):
    pastas = fs.ls(pasta_base)
    paises = [pasta.split('/')[-1] for pasta in pastas]
    return paises

# Função para listar subpastas ou arquivos de um país, ignorando diretórios
def listar_conteudo(pasta):
    pastas = fs.ls(pasta)
    conteudo = [pasta.split('/')[-1] for pasta in pastas if not pasta.endswith('/')]  # Ignorar diretórios
    return conteudo

# Função para contar o número de arquivos em uma subpasta
def contar_arquivos(pasta):
    arquivos = listar_conteudo(pasta)
    return len(arquivos)

# Widget para selecionar o país
dropdown_paises = widgets.Dropdown(
    options=listar_paises(pasta_base),
    description='Countries:',
    disabled=False,
)

# Layout horizontal para alinhar os painéis lado a lado
painel_layout = widgets.Layout(display='flex', flex_flow='row', justify_content='space-between')

# Função para listar arquivos e criar um painel dinâmico para cada subpasta
def criar_painel_arquivos(subpasta_selecionada):
    pasta_subpasta = f"{pasta_base}{dropdown_paises.value}/{subpasta_selecionada}/"
    arquivos = listar_conteudo(pasta_subpasta)
    painel = widgets.Output()
    with painel:
        print(f"Subpasta: {subpasta_selecionada}")
        print(f"Número de arquivos: {len(arquivos)}")
        for arquivo in arquivos:
            print(f'"{arquivo}",')
    return painel

# Função para gerenciar a atualização dos painéis de acordo com os checkboxes
def atualizar_paineis(change):
    # Limpar a área de exibição dos painéis
    clear_output(wait=True)
    display(dropdown_paises, caixa_selecao)

    # Lista de painéis ativos
    paineis_ativos = []

    # Adicionar painéis das subpastas selecionadas
    for checkbox in caixas_subpastas:
        if checkbox.value:
            painel = criar_painel_arquivos(checkbox.description)
            paineis_ativos.append(painel)

    # Exibir os painéis lado a lado
    display(HBox(paineis_ativos, layout=painel_layout))

# Função a ser chamada ao selecionar um país
def ao_selecionar_pais(change):
    pais_selecionado = change['new']
    pasta_pais = f"{pasta_base}{pais_selecionado}/"
    subpastas = listar_conteudo(pasta_pais)

    global caixas_subpastas
    caixas_subpastas = []

    # Criar checkboxes para cada subpasta
    for subpasta in subpastas:
        checkbox = widgets.Checkbox(value=False, description=subpasta)
        checkbox.observe(atualizar_paineis, names='value')
        caixas_subpastas.append(checkbox)

    # Atualizar a caixa de seleção
    caixa_selecao.children = caixas_subpastas
    clear_output(wait=True)
    display(dropdown_paises, caixa_selecao)

# Container para as checkboxes de subpastas
caixa_selecao = widgets.VBox()

# Inicialmente exibir a seleção de países
display(dropdown_paises, caixa_selecao)

# Vincular o evento de mudança de valor no dropdown dos países
dropdown_paises.observe(ao_selecionar_pais, names='value')
