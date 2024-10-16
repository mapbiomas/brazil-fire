import gcsfs
import ipywidgets as widgets
from IPython.display import display, clear_output

# Inicializa o sistema de arquivos do Google Cloud Storage
fs = gcsfs.GCSFileSystem(project=bucket_name)

# Função para listar os países (pastas principais)
def listar_paises(pasta_base):
    pastas = fs.ls(pasta_base)
    paises = [pasta.split('/')[-1] for pasta in pastas if pasta.split('/')[-1]]  # Remove itens vazios
    return paises

# Função para listar o conteúdo da subpasta "training_samples" de cada país
def listar_training_samples(pasta_pais):
    pasta_training = f"{pasta_base}{pasta_pais}/training_samples/"
    try:
        arquivos = fs.ls(pasta_training)
        return [arquivo.split('/')[-1] for arquivo in arquivos if arquivo.split('/')[-1]]  # Remove itens vazios
    except FileNotFoundError:
        return []  # Retorna uma lista vazia se a subpasta não existir

# Função para formatar os arquivos de forma personalizada
def formatar_arquivos(arquivos):
    formatted_list = []
    
    for arquivo in arquivos:
        split = arquivo.split('_')  # Divide o nome do arquivo em partes
        if len(split) >= 6:  # Garantir que há partes suficientes para a formatação
            formatted = f'trainings_{split[2]}_{split[3]}_{split[4]}_{split[5]}'  # Formatação personalizada
            if formatted not in formatted_list:
                formatted_list.append(formatted)  # Adicionar se ainda não estiver na lista
    
    return formatted_list

# Função para coletar amostras selecionadas
def coletar_amostras_selecionadas():
    amostras_selecionadas = []
    for checkbox in checkboxes:
        if checkbox.value:
            amostra = checkbox.description.split('_')
            versao = amostra[1]
            satelite = amostra[2]
            regiao = amostra[3]
            ano = 2023  # Exemplo de ano fixo, pode ajustar conforme necessário
            amostras_selecionadas.append((versao, satelite, regiao, ano))
    return amostras_selecionadas

# Função para gerenciar o clique no botão de simulação
def simular_processamento_click(b):
    amostras_selecionadas = coletar_amostras_selecionadas()
    if amostras_selecionadas:
        for amostra in amostras_selecionadas:
            versao, satelite, regiao, ano = amostra
            treinar_modelo(versao, satelite, regiao, ano, bucket_name, country, simulacao=True)
    else:
        print("Nenhuma amostra selecionada.")

# Função para gerenciar o clique no botão de treinamento
def treinar_modelos_click(b):
    amostras_selecionadas = coletar_amostras_selecionadas()
    if amostras_selecionadas:
        for amostra in amostras_selecionadas:
            versao, satelite, regiao, ano = amostra
            treinar_modelo(versao, satelite, regiao, ano, bucket_name, country, simulacao=False)
    else:
        print("Nenhuma amostra selecionada.")

# Função para exibir o conteúdo de "training_samples" ao selecionar um país
def ao_selecionar_pais(change):
    clear_output(wait=True)  # Limpar a saída anterior, mas manter o dropdown
    
    pais_selecionado = change['new']
    
    # Listar e exibir os arquivos na pasta "training_samples"
    arquivos_training = listar_training_samples(pais_selecionado)
    num_arquivos = len(arquivos_training)
    
    # Exibir o número total de arquivos e país selecionado no topo
    titulo_pais = widgets.HTML(value=f"<b>País selecionado: {pais_selecionado} ({num_arquivos} arquivos encontrados)</b>")
    display(titulo_pais)
    
    display(dropdown_paises)  # Reexibir o dropdown
    
    # Painel com rolagem para os arquivos
    painel_arquivos = widgets.Output(layout={'border': '1px solid black', 'height': '150px', 'overflow_y': 'scroll', 'margin': '10px 0'})
    
    with painel_arquivos:
        for arquivo in arquivos_training:
            print(f'  - {arquivo}')
    
    display(painel_arquivos)  # Exibe o painel com rolagem
    
    if arquivos_training:
        # Formatar os arquivos
        arquivos_formatados = formatar_arquivos(arquivos_training)
        
        # Título das amostras por sensor, região e versão
        num_amostras = len(arquivos_formatados)
        titulo_amostras = widgets.HTML(value=f"<b>Amostras por sensor, região e versão disponíveis para executar o treinamento ({num_amostras} amostras):</b>")
        display(titulo_amostras)
        
        # Exibir checkboxes para cada arquivo formatado
        global checkboxes  # Para acessar na coleta de amostras
        checkboxes = []
        for arquivo in arquivos_formatados:
            checkbox = widgets.Checkbox(value=False, description=arquivo, layout=widgets.Layout(width='auto'))
            checkboxes.append(checkbox)
        
        # Painel para organizar as checkboxes em colunas verticais, sem limitar a altura
        painel_checkboxes = widgets.VBox(checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0'))
        display(painel_checkboxes)
    
        # Botões para simulação e treinamento
        botao_simular = widgets.Button(description="Simular Processamento!", button_style='warning', layout=widgets.Layout(width='200px'))  # Botão amarelo
        botao_treinar = widgets.Button(description="Treinar Modelos", button_style='success', layout=widgets.Layout(width='200px'))  # Botão verde
        
        # Vincular os botões às funções de clique
        botao_simular.on_click(simular_processamento_click)
        botao_treinar.on_click(treinar_modelos_click)
        
        # Layout do rodapé com os dois botões, encostados
        rodape_layout = widgets.HBox([botao_simular, botao_treinar], layout=widgets.Layout(justify_content='flex-start', margin='20px 0'))
        display(rodape_layout)
    
    else:
        mensagem = widgets.HTML(value="<b style='color: red;'>Nenhum arquivo encontrado na pasta 'training_samples'.</b>")
        display(mensagem)

# Widget de dropdown para selecionar o país
dropdown_paises = widgets.Dropdown(
    options=listar_paises(pasta_base),
    description='<b>Países:</b>',
    disabled=False
)

# Exibir o dropdown inicialmente
display(dropdown_paises)

# Vincular o evento de mudança de valor ao dropdown
dropdown_paises.observe(ao_selecionar_pais, names='value')
