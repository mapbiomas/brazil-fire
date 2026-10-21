# MapBiomas Fire Network
Developed by [Amazon Environmental Research Institute - IPAM](https://ipam.org.br/pt/)

## About
**This is the root directory for the MapBiomas Fire Network project.** 

This initiative represents the expansion of the MapBiomas Fire network to other countries in South America. The content in this directory focuses on the effort of sharing our methodology and facilitating collaboration internationally.

The methodology is designed to be replicable across different regions of the continent, using Deep Neural Network (DNN) algorithms for model training and classification.

## Versions and Directory Structure

The network algorithms now live under [`catalog/`](./catalog/), the single umbrella
for the classification algorithm versions and the product catalog:

### [catalog/version_00/](./catalog/version_00/) (Legacy)
Original scripts and methods used for the first collection (Collection 1). Moved
from `network/version_01`; kept for historical reference and compatibility.
**Removal is planned after a public notice.**

### [catalog/version_01/](./catalog/version_01/) (Active)
Current standard for the network (Collection 2 onwards): enhanced models,
automated parameterization, expanded scopes, and multi-country flows. Moved from
`network/version_02`; kept for compatibility. **Removal is planned after a public
notice.** New work should target `catalog/version_04/`.

- **Quick Start via Colab**: The [`mapbiomas_fire_classification_for_collab.ipynb`](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/version_01/classification_algorithms/mapbiomas_fire_classification_for_collab.ipynb) notebook inside `catalog/version_01/classification_algorithms/` is the entry point for the Collection 2 scripts.

### [catalog/version_03/](./catalog/version_03/) (Baseline)
Frozen snapshot of the Peru monitor pipeline (`peru-fire@58897ca`). Reference
only; do not edit.

### [catalog/version_04/](./catalog/version_04/) (Catalog)
Campaign/target catalog under active development: 7 countries, `MONITOR_01`
(monthly) and annual collections. See [`catalog/README.md`](./catalog/README.md).

> `version_02` is reserved.

---
## Contact
For clarifications or to report issues/bugs, please contact:  
[Vera Arruda](mailto:vera.arruda@ipam.org.br)  
[Wallace Silva](mailto:wallace.silva@ipam.org.br)
