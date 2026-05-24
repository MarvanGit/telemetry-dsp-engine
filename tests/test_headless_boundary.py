from __future__ import annotations

import sys

FORBIDDEN_IMPORT_PREFIXES = (
    "PySide6",
    "matplotlib",
    "pyqtgraph",
    "src.main",
    "src.ui",
)


def test_dsp_test_session_does_not_import_ui_or_plotting_modules():
    imported_forbidden_modules = sorted(
        module_name
        for module_name in sys.modules
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    )

    assert imported_forbidden_modules == []
