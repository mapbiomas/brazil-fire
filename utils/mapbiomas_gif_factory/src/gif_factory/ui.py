"""Compatibilidade com notebooks antigos: a interface agora está em `gif_factory.app`."""
from .app import show_app


def show_ui(layers=(), territories=(), output_root="/content/mapbiomas_gif_factory_output",
            project=None, temp_root="/content/temp"):
    """Abre a nova interface em 4 passos; camadas e territórios próprios entram pré-carregados."""
    return show_app(project, layers, territories=territories, output_root=output_root, temp_root=temp_root)
