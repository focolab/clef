"""
Centralized styling constants and widget factories for CLEF demo GUIs.

Both BrainalyzerWorker and LimitCycleVisualizer pull shared colors, stylesheet
fragments, and pre-configured widget builders from this module so that
visual appearance stays consistent across demos.
"""

from pathlib import Path

CSS_PATH = Path(__file__).resolve().parent / "css" / "Ubuntu.qss"


class DemoStyle:
    """Shared styling constants and widget factory methods for CLEF demo GUIs."""

    # ── Semantic color palette ──────────────────────────────────────────
    COLOR_SUCCESS = "#009900"       # green — active/ready pulse buttons
    COLOR_DANGER = "#DC143C"        # red — refractory/cooldown
    COLOR_WARNING = "#FFA500"       # orange — selected/active highlight
    COLOR_NEUTRAL = "#A9A9A9"       # grey — inactive/deselected borders

    COLOR_MUTED_TEXT = "#555555"
    COLOR_HELP_TEXT = "#666666"
    COLOR_SEPARATOR = "#cccccc"

    # ── Stylesheet fragments ────────────────────────────────────────────
    HEADING_STYLE = "font-weight: bold; font-size: 13px; padding-bottom: 2px;"
    SUBHEADING_STYLE = "font-weight: bold; font-size: 12px;"
    MUTED_TEXT_STYLE = "color: #555; padding-bottom: 4px;"
    HELP_TEXT_STYLE = "color: #666;"
    SEPARATOR_STYLE = "color: #ccc;"

    INFO_BOX_STYLE = (
        "QLabel { background: #f8f8f8; border: 1px solid #ddd; "
        "border-radius: 4px; padding: 8px; font-size: 11px; }"
    )
    INSTRUCTIONS_BOX_STYLE = (
        "QLabel { background: #f0f4f8; border: 1px solid #c8d0d8; "
        "border-radius: 4px; padding: 8px; font-size: 11px; }"
    )
    CALLOUT_BOX_STYLE = (
        "QLabel { background: #fff8e7; border: 1px solid #f0c040; "
        "border-radius: 4px; padding: 8px; font-size: 11px; }"
    )
    GROUP_BOX_STYLE = (
        "QGroupBox { font-weight: bold; font-size: 12px; "
        "border: 1px solid #c8d0d8; border-radius: 4px; "
        "margin-top: 8px; padding-top: 14px; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
    )

    # ── Static factory methods ──────────────────────────────────────────

    @staticmethod
    def load_qss(app):
        """Load the shared Ubuntu.qss stylesheet onto a QApplication."""
        if CSS_PATH.exists():
            with open(CSS_PATH, "r") as f:
                app.setStyleSheet(f.read())

    @staticmethod
    def make_heading(text, QtWidgets, style=None):
        """Create a styled QLabel heading."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(style or DemoStyle.HEADING_STYLE)
        return label

    @staticmethod
    def make_separator(QtWidgets):
        """Create a styled horizontal line separator."""
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(DemoStyle.SEPARATOR_STYLE)
        return sep

    @staticmethod
    def make_info_box(text, QtWidgets, style=None):
        """Create a styled info box QLabel with word wrap."""
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(style or DemoStyle.INFO_BOX_STYLE)
        return label

    @staticmethod
    def make_action_button(text, QtWidgets, color=None, height=28):
        """Create a QPushButton with optional accent background and fixed height."""
        btn = QtWidgets.QPushButton(text)
        btn.setFixedHeight(height)
        if color:
            btn.setStyleSheet(f"background-color: {color}")
        return btn

    @staticmethod
    def set_button_state_color(button, active: bool):
        """Set a stimulus button to success (green) or danger (red) state."""
        color = DemoStyle.COLOR_SUCCESS if active else DemoStyle.COLOR_DANGER
        button.setStyleSheet(f"background-color: {color}")
