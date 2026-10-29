__all__ = ["GeoPDFBuilder", "CatalogPDFBuilder"]


def __getattr__(name):
    if name == "GeoPDFBuilder":
        from .geo_pdf import GeoPDFBuilder
        return GeoPDFBuilder
    if name == "CatalogPDFBuilder":
        from .catalog_pdf import CatalogPDFBuilder
        return CatalogPDFBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
