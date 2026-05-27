"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_file(file: BinaryIO | str | Path) -> pd.DataFrame:
    """Load a CSV or XLSX file into a DataFrame.

    Parameters
    ----------
    file:
        A Streamlit uploaded file object or a filesystem path.

    Returns
    -------
    pandas.DataFrame
        Parsed dataset.

    Raises
    ------
    ValueError
        If the file type is unsupported or parsing fails.
    """
    filename = getattr(file, "name", str(file))
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    if hasattr(file, "seek"):
        file.seek(0)

    try:
        if suffix == ".csv":
            try:
                return pd.read_csv(file)
            except UnicodeDecodeError:
                if hasattr(file, "seek"):
                    file.seek(0)
                return pd.read_csv(file, encoding="latin1")

        return pd.read_excel(file, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 - Streamlit should show the original parser issue.
        raise ValueError(f"Could not load dataset: {exc}") from exc
