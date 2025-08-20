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
    import tempfile
    import urllib.request
    import os

    if kind == "info":
        filename = "pathway_info.csv"
        github_url = "https://raw.githubusercontent.com/ZAEDPolSl/pyFUNCellA/main/pyfuncella/data/pathway_info.csv"
        try:
            with importlib.resources.files("pyfuncella.data").joinpath(filename).open(
                "r", encoding="utf-8"
            ) as f:
                return pd.read_csv(f)
        except FileNotFoundError:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                urllib.request.urlretrieve(github_url, tmp.name)
                tmp.close()
                df = pd.read_csv(tmp.name)
            os.unlink(tmp.name)
            return df

    elif kind == "data":
        filename = "pathways.json"
        github_url = "https://raw.githubusercontent.com/ZAEDPolSl/pyFUNCellA/main/pyfuncella/data/pathways.json"
        try:
            with importlib.resources.files("pyfuncella.data").joinpath(filename).open(
                "r", encoding="utf-8"
            ) as f:
                return json.load(f)
        except FileNotFoundError:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                urllib.request.urlretrieve(github_url, tmp.name)
                tmp.close()
                with open(tmp.name, "r", encoding="utf-8") as f:
                    data = json.load(f)
            os.unlink(tmp.name)
            return data

    else:
        raise ValueError("kind must be either 'info' or 'data'")
