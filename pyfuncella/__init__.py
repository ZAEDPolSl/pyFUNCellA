import importlib.resources
import pandas as pd
import json


def load_pathways(kind: str):
    """
    Load pathway information or data.

    Parameters
    ----------
    kind : str
        Either "info" (returns CSV as DataFrame) or "data" (returns JSON as dict)

    Returns
    -------
    pd.DataFrame | dict
    """
    if kind == "info":
        filename = "pathway_info.csv"
        with importlib.resources.files(__package__ + ".data").joinpath(filename).open(
            "r", encoding="utf-8"
        ) as f:
            return pd.read_csv(f)

    elif kind == "data":
        filename = "pathways.json"
        with importlib.resources.files(__package__ + ".data").joinpath(filename).open(
            "r", encoding="utf-8"
        ) as f:
            return json.load(f)

    else:
        raise ValueError("kind must be either 'info' or 'data'")
