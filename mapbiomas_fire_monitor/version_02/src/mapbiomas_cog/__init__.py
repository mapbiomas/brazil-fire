"""mapbiomas_cog — engine de export/resume de COGs multi-iniciativa (version_02).

Nucleo library-first:

    catalogo -> engine (runner plugavel) -> assemble COG -> publish

A UI/notebook chama apenas o facade publico (``Engine``), mantendo a fabrica
de download desacoplada da interface.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
