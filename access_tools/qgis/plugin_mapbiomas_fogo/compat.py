# -*- coding: utf-8 -*-
"""
Compatibilidade de enums Qt5 / Qt6 para plugins QGIS.

No PyQt5 (QGIS 3) os enums do Qt são acessados como atributos planos:
    Qt.UserRole, Qt.Checked, Qt.AlignRight …

No PyQt6 (QGIS 4) os enums são aninhados nas suas classes:
    Qt.ItemDataRole.UserRole, Qt.CheckState.Checked, Qt.AlignmentFlag.AlignRight …

Esta classe proxy intercepta o acesso de atributos e tenta os dois caminhos,
garantindo compatibilidade sem alterar o restante do código.
"""

from qgis.PyQt.QtCore import Qt as _RealQt

# Mapeamento: nome_plano → classe_de_enum_no_PyQt6
_SCOPED = {
    # ItemDataRole
    "UserRole":             "ItemDataRole",
    "DisplayRole":          "ItemDataRole",
    "DecorationRole":       "ItemDataRole",
    "EditRole":             "ItemDataRole",
    "ToolTipRole":          "ItemDataRole",
    "StatusTipRole":        "ItemDataRole",
    "CheckStateRole":       "ItemDataRole",
    "SizeHintRole":         "ItemDataRole",
    # ItemFlag
    "ItemIsEditable":       "ItemFlag",
    "ItemIsEnabled":        "ItemFlag",
    "ItemIsSelectable":     "ItemFlag",
    "ItemIsUserCheckable":  "ItemFlag",
    "ItemIsUserTristate":   "ItemFlag",
    "ItemNeverHasChildren": "ItemFlag",
    "ItemIsDragEnabled":    "ItemFlag",
    "ItemIsDropEnabled":    "ItemFlag",
    "NoItemFlags":          "ItemFlag",
    # CheckState
    "Checked":              "CheckState",
    "Unchecked":            "CheckState",
    "PartiallyChecked":     "CheckState",
    # AlignmentFlag
    "AlignLeft":            "AlignmentFlag",
    "AlignRight":           "AlignmentFlag",
    "AlignCenter":          "AlignmentFlag",
    "AlignHCenter":         "AlignmentFlag",
    "AlignVCenter":         "AlignmentFlag",
    "AlignTop":             "AlignmentFlag",
    "AlignBottom":          "AlignmentFlag",
    "AlignJustify":         "AlignmentFlag",
    "AlignAbsolute":        "AlignmentFlag",
    # SortOrder
    "AscendingOrder":       "SortOrder",
    "DescendingOrder":      "SortOrder",
    # ContextMenuPolicy
    "CustomContextMenu":    "ContextMenuPolicy",
    "NoContextMenu":        "ContextMenuPolicy",
    "DefaultContextMenu":   "ContextMenuPolicy",
    "ActionsContextMenu":   "ContextMenuPolicy",
    "PreventContextMenu":   "ContextMenuPolicy",
    # WidgetAttribute
    "WA_DeleteOnClose":     "WidgetAttribute",
    "WA_TranslucentBackground": "WidgetAttribute",
    "WA_NoSystemBackground": "WidgetAttribute",
    # DockWidgetArea
    "RightDockWidgetArea":  "DockWidgetArea",
    "LeftDockWidgetArea":   "DockWidgetArea",
    "TopDockWidgetArea":    "DockWidgetArea",
    "BottomDockWidgetArea": "DockWidgetArea",
    "NoDockWidgetArea":     "DockWidgetArea",
    # FocusReason
    "OtherFocusReason":     "FocusReason",
    "TabFocusReason":       "FocusReason",
    "MouseFocusReason":     "FocusReason",
    # KeyboardModifier
    "NoModifier":           "KeyboardModifier",
    "ControlModifier":      "KeyboardModifier",
    "ShiftModifier":        "KeyboardModifier",
    "AltModifier":          "KeyboardModifier",
    "MetaModifier":         "KeyboardModifier",
    # Orientation
    "Horizontal":           "Orientation",
    "Vertical":             "Orientation",
    # WindowType
    "Window":               "WindowType",
    "Dialog":               "WindowType",
    "Tool":                 "WindowType",
    "Popup":                "WindowType",
    "FramelessWindowHint":  "WindowType",
    "WindowStaysOnTopHint": "WindowType",
    # ScrollBarPolicy
    "ScrollBarAlwaysOff":   "ScrollBarPolicy",
    "ScrollBarAlwaysOn":    "ScrollBarPolicy",
    "ScrollBarAsNeeded":    "ScrollBarPolicy",
    # CursorShape
    "ArrowCursor":          "CursorShape",
    "WaitCursor":           "CursorShape",
    "BusyCursor":           "CursorShape",
    "PointingHandCursor":   "CursorShape",
    # AspectRatioMode
    "KeepAspectRatio":      "AspectRatioMode",
    "IgnoreAspectRatio":    "AspectRatioMode",
    # TransformationMode
    "SmoothTransformation": "TransformationMode",
    "FastTransformation":   "TransformationMode",
    # MatchFlag
    "MatchExactly":         "MatchFlag",
    "MatchContains":        "MatchFlag",
    "MatchStartsWith":      "MatchFlag",
    "MatchEndsWith":        "MatchFlag",
    "MatchCaseSensitive":   "MatchFlag",
    "MatchRegularExpression": "MatchFlag",
    # MouseButton
    "LeftButton":           "MouseButton",
    "RightButton":          "MouseButton",
    "MiddleButton":         "MouseButton",
    "NoButton":             "MouseButton",
    # DropAction
    "CopyAction":           "DropAction",
    "MoveAction":           "DropAction",
    "LinkAction":           "DropAction",
    "IgnoreAction":         "DropAction",
}


class _QtProxy:
    """
    Proxy transparente para o namespace Qt.
    Tenta primeiro acesso plano (PyQt5) e depois acesso aninhado (PyQt6).
    Delega todos os outros atributos diretamente ao Qt real.
    """

    __slots__ = ()

    def __getattr__(self, name):
        # 1. Acesso direto (funciona no PyQt5 e para atributos que não são enums)
        val = getattr(_RealQt, name, _MISSING)
        if val is not _MISSING:
            return val

        # 2. Acesso aninhado via mapa (PyQt6)
        scope = _SCOPED.get(name)
        if scope:
            enum_class = getattr(_RealQt, scope, None)
            if enum_class is not None:
                val = getattr(enum_class, name, _MISSING)
                if val is not _MISSING:
                    return val

        raise AttributeError(f"Qt has no attribute '{name}'")


_MISSING = object()

# Instância única usada como substituto do Qt importado
Qt = _QtProxy()


# ============================================================================
# Proxy para QMessageBox compatível com PyQt5/Qt5 e PyQt6/Qt6
# ============================================================================

from qgis.PyQt.QtWidgets import QMessageBox as _RealQMessageBox

# Mapeamento de constantes de QMessageBox que mudaram de nome entre PyQt5 e PyQt6
_QMESSAGEBOX_SCOPED = {
    "Ok":       "StandardButton",
    "Cancel":   "StandardButton",
    "Save":     "StandardButton",
    "Discard":  "StandardButton",
    "Yes":      "StandardButton",
    "No":       "StandardButton",
    "Retry":    "StandardButton",
    "Ignore":   "StandardButton",
    "NoButton": "StandardButton",
}


class _QMessageBoxProxy:
    """
    Proxy para QMessageBox que fornece compatibilidade entre PyQt5 e PyQt6.
    
    No PyQt5, você usa: QMessageBox.Ok, QMessageBox.Cancel
    No PyQt6, você usa: QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Cancel
    
    Este proxy detecta automaticamente qual versão está sendo usada.
    """

    __slots__ = ()

    def __getattr__(self, name):
        # Tenta acesso direto primeiro (funciona no PyQt5)
        if hasattr(_RealQMessageBox, name) and name[0].isupper():
            val = getattr(_RealQMessageBox, name, _MISSING)
            if val is not _MISSING:
                return val

        # Tenta acesso aninhado para PyQt6
        scope = _QMESSAGEBOX_SCOPED.get(name)
        if scope:
            enum_class = getattr(_RealQMessageBox, scope, None)
            if enum_class is not None:
                val = getattr(enum_class, name, _MISSING)
                if val is not _MISSING:
                    return val

        # Fallback para atributos/métodos normais
        return getattr(_RealQMessageBox, name)

    def __call__(self, *args, **kwargs):
        return _RealQMessageBox(*args, **kwargs)


# Instância do proxy de QMessageBox
QMessageBox = _QMessageBoxProxy()
