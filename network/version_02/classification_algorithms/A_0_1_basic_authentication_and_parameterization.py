# last_update: '2024/10/22', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_0_1_basic_authentication_and_parameterization.py 
### Step A_0_1 - Basic authentication and parameterization

# Country options: ['bolivia', 'colombia', 'chile', 'peru', 'paraguay', 'guyana']
# Specify the country for processing. Options: ['bolivia', 'colombia', 'chile', 'peru', 'paraguay', 'guyana']
country = 'guyana'  # Set the country from the available options
# Define a name for the collection to generate log messages and build GCS/GEE paths
collection = 'collection_1'
collection_ee = collection.upper().replace('_', '') # e.g., COLLECTION1

# Import and authenticate libraries for Google cloud services
ee_project = f'mapbiomas-{country}'  # Set the project name based on the selected country
bucket_name = 'mapbiomas-fire'

def authenticates(ee_project,bucketName):
    import ee
    ee.Authenticate()
    ee.Initialize(project=ee_project)

    # Authenticate with Google Cloud (necessary when using Colab)
    from google.colab import auth
    auth.authenticate_user()

    # Initialize Google Cloud Storage client and define the bucket name
    from google.cloud import storage
    client = storage.Client()
    bucket = client.get_bucket(bucketName)

# Call the authentication function with the correct project and bucket names
authenticates(ee_project, bucket_name)

# Define the path to the classification algorithms scripts
algorithms = f'/content/brazil-fire/network/fire_landsat_30m/version_02/classification_algorithms'

exec(open(f'/content/brazil-fire/utils/google_collab_pdf_show.py').read())
pdf_path = '/content/brazil-fire/network/fire_landsat_30m/version_02/entrenamiento_de_monitoreo_de_cicatrices_de_fuego_en_regiones_de_la_red_mapBiomas.pdf'
display_pdf_viewer(pdf_path, external_link="https://docs.google.com/presentation/d/1MPoqHWHLw-jJqKUStikJ0Cc-8oLuMuKZ4_c_kQA3DKQ")
