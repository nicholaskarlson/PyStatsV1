from __future__ import annotations

import importlib.resources
import webbrowser
from pathlib import Path


def get_local_docs_path() -> Path:
    """Return the path to the bundled PDF documentation.

    Raises FileNotFoundError if the PDF is missing.
    """
    # importlib.resources.files returns an abstract path-like object
    pdf_resource = importlib.resources.files("pystatsv1.docs") / "pystatsv1.pdf"
    pdf_path = Path(pdf_resource)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Documentation PDF not found: {pdf_path}")

    return pdf_path


def open_local_docs() -> None:
    """Open the bundled PDF documentation in the default PDF viewer."""
    pdf_path = get_local_docs_path()
    webbrowser.open_new(pdf_path.as_uri())
