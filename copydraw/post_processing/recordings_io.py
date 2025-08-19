import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.progress import track

from copydraw.utils.logging import logger

# Fields to transform to numpy arrays
TO_NUMPY = [
    "cursor_t",
    "scaling_matrix",
    "trace_let",
    "traces_pix",
    "template",
    "template_pix",
    "template_pos",
    "template_size",
]


def load_copydraw_record_yaml(pth: Path) -> dict:
    """Load a single recording"""
    data = yaml.safe_load(open(pth, "r"))

    for k, v in data.items():
        if k in TO_NUMPY:
            data[k] = np.asarray(v)

    return data


def files_to_dataframe(folder: Path) -> pd.DataFrame:
    """Read all yaml files to a data frame"""
    files = list(folder.rglob("*trial*.yaml"))
    dicts = []
    for fl in track(files):
        blk = int(re.findall(r"block_(\d*)", str(fl))[0])
        trial = int(re.findall(r"trial(\d*)", str(fl))[0])
        data = load_copydraw_record_yaml(fl)

        if data["ix_block"] != blk or data["ix_trial"] != trial:
            logger.warning(
                f"Inconsistency between stored block and trial index found - "
                f"{data['ix_block']=}, {data['ix_trial']=}, {blk=}, {trial=}"
            )

        dicts.append(data)

    df = pd.DataFrame(dicts)
    return df


if __name__ == "__main__":
    folder = Path("./data/VPtest").resolve()

    df = files_to_dataframe(folder)
