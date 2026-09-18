# notebooks (version_02)

Alvo único desta v02: `mapbiomas_directlink_all_initiatives.ipynb` — wrapper fino
que instala o pacote, autentica no Colab e abre `mapbiomas_cog.ui.run_ui`.

O notebook específico de país (`mapbiomas_fire_monitor_brazil.ipynb`) **não**
faz parte desta versão.

Fluxo na UI: país → tema → coleção → produto → **Load units** → **Export** →
**Sync** → **Assemble**.
