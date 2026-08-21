# MapBiomas Fogo - QGIS Plugin

![MapBiomas Logo](logo_mapbiomas.png)

## Overview

The **MapBiomas Fogo** plugin provides seamless integration with MapBiomas Fire data directly in QGIS. This plugin allows users to access, visualize, and analyze fire occurrence data throughout Brazil, displaying both accumulated and annual fire patterns from the MapBiomas Project.

MapBiomas is a collaborative network that monitors land use and land cover changes in Brazil, providing essential data for environmental analysis and decision-making.

## Features

- **Direct Access to Fire Data**: Connect to MapBiomas Fire data API directly from within QGIS
- **Accumulated Fire Visualization**: View total burned areas over time with cumulative frequency analysis
- **Annual Fire Layers**: Access fire occurrence data for specific years (1985-2022)
- **Land Use Integration**: Visualize fire data classified by land use and land cover classes
- **Custom Legends**: Automatic legend generation for different data categories and visualization modes
- **Multi-temporal Analysis**: Compare fire patterns across different time periods

## Installation

### Via QGIS Plugin Manager (Recommended)
1. Open QGIS
2. Go to `Plugins` → `Manage and Install Plugins`
3. Search for "MapBiomas Fogo"
4. Click `Install Plugin`

### Manual Installation
1. Download or clone this repository
2. Copy the plugin folder to your QGIS plugins directory:
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS
4. Enable the plugin via `Plugins` → `Manage and Install Plugins` → `Installed`

## Requirements

- **QGIS Version**: 3.22 or higher (for optimal Qt6 compatibility)
- **Internet Connection**: Required to access the MapBiomas API
- **Qt Framework**: Qt5 (5.12+) or Qt6 (6.2+) - automatically detected by QGIS
- **Python Libraries**: All dependencies are included with standard QGIS installation

## Usage

### Basic Workflow

1. **Open the Plugin**
   - Click the MapBiomas Fogo icon ![Plugin Icon](icon.png) in the QGIS toolbar
   - Or access via `Plugins` → `MapBiomas Fogo`

2. **Select Data Type**
   - **Accumulated Fire**: Total burned areas over selected time period
   - **Annual Fire**: Fire occurrence for a specific year

3. **Configure Parameters**
   - Choose time period or specific year
   - Select visualization mode:
     - By land use/land cover classes
     - By total burn frequency
     - By fire recurrence patterns

4. **Territory Analysis** (Optional)
   - Select administrative boundaries (states, municipalities, biomes, RPPNs) 
   - Generate statistical reports
   - Export data as vector layers with attributes

5. **Load Data**
   - Click "Load" to add the selected layers to your QGIS project
   - Layers will be automatically styled with appropriate legends

### Advanced Features

#### Custom Visualization
- Combine multiple years of data
- Create custom fire frequency classifications
- Export high-quality maps for reports

## Data Source

All fire data is provided by the **MapBiomas Project**, a collaborative initiative that monitors land use and land cover changes in Brazil using satellite imagery and advanced classification algorithms.

- **Spatial Resolution**: 30 meters
- **Temporal Coverage**: 2000-2024 (updated annually)
- **Geographic Coverage**: Complete Brazilian territory
- **Data Source**: Landsat satellite imagery processed through Google Earth Engine

Learn more about MapBiomas: [https://mapbiomas.org](https://mapbiomas.org)

## Technical Details

### Architecture
- Built with PyQt5/PyQt6 for cross-platform compatibility
- Integrates with QGIS Processing framework
- Uses QGIS native rendering engine for optimal performance
- RESTful API integration for real-time data access

### Supported Formats
- **Input**: MapBiomas API (GeoTIFF/Cloud Optimized GeoTIFF)
- **Output**: QGIS raster 
- **Export**: GeoPackage

## Contributing

We welcome contributions to improve the MapBiomas Fogo plugin! Please feel free to:

- Report bugs or request features via GitHub Issues
- Submit pull requests with improvements
- Share feedback and suggestions
- Help with translations

## Credits

**Developed by IPAM Amazônia (Instituto de Pesquisa Ambiental da Amazônia)**

### Development Team
- **Newton Monteiro** - Lead Developer
- **Wallace Silva** - Developer
- **Felipe Martenexen** - Developer  
- **Vera Laísa** - Developer
- **João Ribeiro** - Developer

### Contact
- **Email**: newton.monteiro@ipam.org.br
- **Organization**: [IPAM Amazônia](https://ipam.org.br)

## License

This plugin is licensed under the **GNU General Public License v2.0 (GPL-2.0)**

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MapBiomas Project** for providing the fire data and API
- **QGIS Community** for the excellent GIS platform
- **Google Earth Engine** for satellite data processing infrastructure
- **IPAM Amazônia** for supporting the development and maintenance

---

**MapBiomas Fogo Plugin** - Bringing Brazil's fire data to your QGIS workspace
