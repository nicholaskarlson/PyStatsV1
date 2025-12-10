from __future__ import annotations

from pathlib import Path

import pystatsv1


def test_get_local_docs_path_points_to_existing_pdf() -> None:
    pdf_path = pystatsv1.get_local_docs_path()
    assert isinstance(pdf_path, Path)
    assert pdf_path.name == "pystatsv1.pdf"
    assert pdf_path.is_file()