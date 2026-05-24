from __future__ import annotations

import importlib.abc
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORBIDDEN_IMPORT_PREFIXES = (
    "PySide6",
    "matplotlib",
    "pyqtgraph",
    "src.main",
    "src.ui",
)


def is_forbidden_import(fullname: str) -> bool:
    return any(
        fullname == prefix or fullname.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


class HeadlessImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if is_forbidden_import(fullname):
            raise ImportError(
                f"{fullname!r} is forbidden during headless DSP validation. "
                "Keep CI tests focused on DSP modules and away from UI imports."
            )
        return None


if not any(isinstance(finder, HeadlessImportBlocker) for finder in sys.meta_path):
    sys.meta_path.insert(0, HeadlessImportBlocker())


def pytest_sessionfinish(session, exitstatus):
    violations = sorted(name for name in sys.modules if is_forbidden_import(name))
    if not violations:
        return

    session.exitstatus = pytest.ExitCode.TESTS_FAILED
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_sep(
            "=",
            "Forbidden UI imports detected in headless DSP validation: "
            + ", ".join(violations),
        )
