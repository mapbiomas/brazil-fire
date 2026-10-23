# MapBiomas Fire Network
Developed by [Amazon Environmental Research Institute - IPAM](https://ipam.org.br/pt/)

## About
**This is the root directory for the MapBiomas Fire Network project.** 

This initiative represents the expansion of the MapBiomas Fire network to other countries in South America. The content in this directory focuses on the effort of sharing our methodology and facilitating collaboration internationally.

The methodology is designed to be replicable across different regions of the continent, using Deep Neural Network (DNN) algorithms for model training and classification.

## Versions and Directory Structure

The network algorithms now live under [`catalog/`](./catalog/), split into
development versions and a production release:

### [catalog/development/version_00/](./catalog/development/version_00/) (Legacy)
Original scripts and methods used for the first collection (Collection 1). Moved
from `network/version_01`; kept for historical reference and compatibility.
**Removal is planned after a public notice.**

### [catalog/development/version_01/](./catalog/development/version_01/) (Active)
Current standard for the network (Collection 2 onwards): enhanced models,
automated parameterization, expanded scopes, and multi-country flows. Moved from
`network/version_02`; kept for compatibility. **Removal is planned after a public
notice.** New work should target `catalog/development/version_03/`.

- **Quick Start via Colab**: The [`mapbiomas_fire_classification_for_collab.ipynb`](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/development/version_01/classification_algorithms/mapbiomas_fire_classification_for_collab.ipynb) notebook inside `catalog/development/version_01/classification_algorithms/` is the entry point for the Collection 2 scripts.

### [catalog/development/version_03/](./catalog/development/version_03/) (Active development + stable reference)
Campaign/target catalog: 9 countries, `MONITOR_01` (monthly) and annual
collections. It is both the active development copy and the source for
production (`version_04` was removed). See
[`catalog/README.md`](./catalog/README.md) and
[`version_03/COLAB_LINKS.csv`](./catalog/development/version_03/COLAB_LINKS.csv).

### [catalog/production/](./catalog/production/) (Production)
Shipped release, copied from `catalog/development/version_03`. Public Colab
links are in [`production/COLAB_LINKS.csv`](./catalog/production/COLAB_LINKS.csv).

## Legacy root locations

### [version_01/](./version_01/) (Legacy mirror, deprecated)
The previous root copy of the Collection 1 scripts is kept temporarily so
existing notebooks and links keep working **unchanged**. It is **deprecated**
and will be removed after a public notice. New access must use
[`catalog/development/version_00/`](./catalog/development/version_00/).

> **Reference version in the catalog:** `catalog/development/version_00/`.

### [version_02/](./version_02/) (Legacy mirror, deprecated)
The previous root copy of the Collection 2 scripts is kept temporarily so
existing notebooks and links keep working **unchanged**. It is **deprecated**
and will be removed after a public notice. New access must use
[`catalog/development/version_01/`](./catalog/development/version_01/).

> **Reference version in the catalog:** `catalog/development/version_01/`.

> Catalog `version_02` is reserved (separate from this legacy mirror).

See [`catalog/VALIDATION.md`](./catalog/VALIDATION.md) for the validated entry
points of every version.

---
## Contact
For clarifications or to report issues/bugs, please contact:  
[Vera Arruda](mailto:vera.arruda@ipam.org.br)  
[Wallace Silva](mailto:wallace.silva@ipam.org.br)
