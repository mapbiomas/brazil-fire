# last_update: '2024/10/22', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_1_0_gee_gui_collect_samples_burned_area_classification.py 
### Step A_1_0 - Simple script for share acess toolkt in Google Earth Engine toolkit interface to collect samples and avaliable results

# Definindo os links
presentation_link = "https://docs.google.com/presentation/d/1iMRXRH4xoWTFPSSzDOJkB6c7KJLPggxrnvYzVQ3BzP0/edit#slide=id.g220825c6698_0_546"
gee_toolkit_link = "https://code.earthengine.google.com/?scriptPath=users%2Fmapbiomasworkspace1%2Fmapbiomas-fire%3A1-Toolkit_Collection1%2FToolkit_samples_collection"

# Função que imprime os labels e os links || Function that prints the labels and links
def display_links():
    print("### How it works and how to use the toolkit ###")
    print(f"Presentation: {presentation_link}")
    print("\n### Access the Toolkit on GEE ###")
    print(f"Toolkit Link: {gee_toolkit_link}")

# Execução principal
if __name__ == "__main__":
    display_links()  
