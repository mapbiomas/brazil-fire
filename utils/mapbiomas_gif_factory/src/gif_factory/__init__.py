"""Fábrica de GIFs MapBiomas: mapas animados de qualquer território, direto do Colab.

Uso no Colab:

    from gif_factory.app import show_app
    show_app("meu-projeto-gee")
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mapbiomas-gif-factory")
except PackageNotFoundError:  # executando direto do código-fonte
    __version__ = "0+local"
