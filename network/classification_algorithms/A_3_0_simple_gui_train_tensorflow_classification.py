import gcsfs
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox

bucketName = 'mapbiomas-fire'
pastaBase = 'mapbiomas-fire/sudamerica/'

# Inicializa o sistema de arquivos do Google Cloud Storage
fs = gcsfs.GCSFileSystem(project=bucketName)

# Função para listar os países (pastas principais)
def listar_paises(pastaBase):
    pastas = fs.ls(pastaBase)
    paises = [pasta.split('/')[-1] for pasta in pastas]  # Mantém todos os itens, inclusive os vazios
    return paises

# Função para listar o conteúdo da subpasta "models" de cada país
def listar_training_samples(pasta_pais):
    pasta_training = f"{pastaBase}{pasta_pais}/models/"
    try:
        arquivos = fs.ls(pasta_training)
        return [arquivo.split('/')[-1] for arquivo in arquivos if arquivo.split('/')[-1]]  # Remove itens vazios
    except FileNotFoundError:
        return []  # Retorna uma lista vazia se a subpasta não existir

# Função para listar os arquivos da pasta "mosaics_col1_cog" de cada país, filtrados pela região "r"
def listar_mosaics(pasta_pais, regiao):
    pasta_mosaics = f"{pastaBase}{pasta_pais}/mosaics_col1_cog/"
    try:
        arquivos = fs.ls(pasta_mosaics)
        # Filtra apenas os arquivos que contenham a região "rX" no nome do arquivo
        return [arquivo.split('/')[-1] for arquivo in arquivos if f"_{regiao}_" in arquivo]
    except FileNotFoundError:
        return []  # Retorna uma lista vazia se a subpasta não existir

# Função para listar os arquivos classificados na pasta result_classified
def listar_classificados(pasta_pais):
    pasta_classificados = f"{pastaBase}{pasta_pais}/result_classified/"
    try:
        arquivos = fs.ls(pasta_classificados)
        return [arquivo.split('/')[-1] for arquivo in arquivos]
    except FileNotFoundError:
        return []  # Retorna uma lista vazia se a subpasta não existir

# Função para verificar se um arquivo de mosaico já foi classificado
def verificar_classificado(classificados, arquivo_mosaico, regiao, versao, ano):
    # Cria o padrão do nome do arquivo classificado
    padrao_classificado = f"burned_area_{arquivo_mosaico.split('_')[1]}_{versao}_{regiao}_{ano}.tif"
    return padrao_classificado in classificados

# Função para exibir os arquivos de mosaicos associados ao modelo selecionado
def exibir_mosaicos_selecionados(modelo, pais_selecionado, regiao):
    arquivos_mosaics = listar_mosaics(pais_selecionado, regiao)
    arquivos_classificados = listar_classificados(pais_selecionado)
    
    # Cria um painel com altura limitada e rolagem para os mosaicos
    painel_mosaics = widgets.Output(layout={'border': '1px solid black', 'height': '200px', 'overflow_y': 'scroll'})
    with painel_mosaics:
        if arquivos_mosaics:
            print(f"Mosaicos da região {regiao} do modelo: {modelo}")
            for arquivo in arquivos_mosaics:
                # Extraindo a versão e ano do nome do arquivo
                versao = arquivo.split('_')[1]
                ano = arquivo.split('_')[-1].split('.')[0]

                # Verifica se o arquivo já foi classificado
                classificado = verificar_classificado(arquivos_classificados, arquivo, regiao, versao, ano)
                
                # Se o arquivo já foi classificado, desliga o checkbox e adiciona "⚠️" ao nome
                checkbox_mosaico = widgets.Checkbox(value=not classificado, description=arquivo + (" ⚠️" if classificado else ""))
                display(checkbox_mosaico)
        else:
            print(f"Nenhum mosaico encontrado para a região {regiao}")
    
    # Painel de legenda
    painel_legenda = widgets.Output(layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'})
    with painel_legenda:
        print("⚠️ Arquivo já existe e será sobrescrito se o checkbox for mantido ligado.")

    return widgets.VBox([painel_mosaics, painel_legenda])

# Função para atualizar a interface e exibir os painéis corretamente
def atualizar_interface():
    # Limpa a saída e reexibe o dropdown, checkboxes e painéis
    clear_output(wait=True)
    display(dropdown_paises)
    display(VBox(checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0')))
    display(HBox(painel_mosaicos, layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')))  # Exibe os painéis lado a lado
    
    # Botões
    rodape_layout = widgets.HBox([botao_simular, botao_classificar], layout=widgets.Layout(justify_content='flex-start', margin='20px 0'))
    display(rodape_layout)

# Função para exibir o conteúdo da pasta "models" ao selecionar um país
def ao_selecionar_pais(change):
    pais_selecionado = change['new']
    
    # Listar os arquivos na pasta "models"
    arquivos_training = listar_training_samples(pais_selecionado)
    
    # Se há arquivos na pasta "models", cria a interface de checkboxes
    if arquivos_training:
        global checkboxes, painel_mosaicos
        checkboxes = []
        painel_mosaicos = []  # Armazena os painéis de mosaicos

        # Função interna para exibir e remover painéis dinamicamente
        def atualizar_paineis(change, arquivo, regiao):
            global painel_mosaicos  # Certifica-se de que estamos usando a variável global

            if change['new']:  # Se checkbox ativado
                painel = exibir_mosaicos_selecionados(arquivo, pais_selecionado, regiao)
                painel_mosaicos.append(painel)
            else:  # Se checkbox desativado
                # Remove o painel da região desativada
                painel_mosaicos = [p for p in painel_mosaicos if f"_{regiao}_" not in p.outputs[0]['text']]  # Corrigido o acesso ao atributo `outputs`

            # Atualiza a interface após a modificação
            atualizar_interface()

        # Criar checkboxes para cada arquivo dentro de "models"
        for arquivo in arquivos_training:
            # Identifica a região do modelo a partir do nome (ex: r1, r2, etc.)
            regiao = arquivo.split('_')[-1].split('.')[0]
            checkbox = widgets.Checkbox(value=False, description=arquivo)
            checkbox.observe(lambda change, arq=arquivo, reg=regiao: atualizar_paineis(change, arq, reg), names='value')
            checkboxes.append(checkbox)
        
        # Atualiza a interface inicial
        atualizar_interface()
    
    else:
        # Exibir mensagem de erro se não houver arquivos
        mensagem = widgets.HTML(value="<b style='color: red;'>Nenhum arquivo encontrado na pasta 'models'.</b>")
        clear_output(wait=True)
        display(dropdown_paises)
        display(mensagem)

# Função para gerenciar o clique no botão de simulação de processamento
def simular_processamento_click(b):
    modelos_selecionados = coletar_modelos_selecionados()
    if modelos_selecionados:
        for amostra in modelos_selecionados:
            print(f"Simulando classificação para: {amostra}")
    else:
        print("Nenhum arquivo selecionado.")

# Função para gerenciar o clique no botão de classificação de área queimada
def classificar_area_queimada_click(b):
    modelos_selecionados = coletar_modelos_selecionados()
    if modelos_selecionados:
        for modelo in modelos_selecionados:
            print(f"Classificando área queimada para o modelo: {modelo}")
            # Aqui chamamos a função render_classify para processar as imagens
            dataset_classify = load_image(modelo)  # Função que você usa para carregar a imagem (GDAL)
            output_image = render_classify(dataset_classify, model_path='path_to_model', country='guyana', region='region1', version='v1', year=2020)
            # Salvar ou processar a imagem classificada conforme necessário
    else:
        print("Nenhum modelo selecionado.")


# Coletar os modelos selecionados (apenas o nome completo do arquivo)
def coletar_modelos_selecionados():
    modelos_selecionados = [checkbox.description for checkbox in checkboxes if checkbox.value]
    return modelos_selecionados

# Widget de dropdown para selecionar o país
dropdown_paises = widgets.Dropdown(
    options=listar_paises(pastaBase),
    description='<b>Países:</b>',
    disabled=False
)

# Botões no final da interface
botao_simular = widgets.Button(description="Simular Processamento!", button_style='warning', layout=widgets.Layout(width='200px'))  # Botão amarelo
botao_classificar = widgets.Button(description="Classificar área queimada", button_style='success', layout=widgets.Layout(width='200px'))  # Botão verde

botao_classificar.on_click(classificar_area_queimada_click)
# Exibir o dropdown inicialmente
display(dropdown_paises)

# Vincular o evento de mudança de valor ao dropdown
dropdown_paises.observe(ao_selecionar_pais, names='value')
