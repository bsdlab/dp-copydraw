import os
from pathlib import Path

import numpy as np
import yaml

from copydraw.utils.logging import logger


def load_block_config(script_root=None):
    block_cfg = load_paradigm_config(script_root=script_root)["block"]
    return block_cfg


def load_paradigm_config(script_root=None, data_root=None):
    cfg = {}
    if script_root is None:
        script_root = Path(__file__).parents[2]

    paradigm_config_path = script_root.joinpath("configs", "paradigm_config.yaml")
    cfg = yaml.load(open(paradigm_config_path), Loader=yaml.FullLoader)

    # use data root passed to this function
    if data_root is not None:
        cfg.update({"data_root": Path(data_root)})

    cfg.update({"data_root": Path(cfg["data_root"])})
    cfg.update({"script_root": Path(script_root)})

    return cfg


def get_nextblock_metadata(save_dir=None, paradigm_name="copyDraw"):
    if save_dir is None:
        logger.info(
            "no save directory passed to the function looking for existing save files - get_nextblock_metadata()"
        )
        logger.info("looking for existing data in default location...")
        save_dir = Path(load_paradigm_config()["data_root"], paradigm_name, "results")
        logger.info("Default Location for data:", save_dir)

    # example save location:
    # dataroot / copyDraw / results / copyDraw_block03 / datafile.extension
    block_name = f"{paradigm_name}_block"

    if not os.path.exists(save_dir):
        ix_block = 1
    else:
        all_files = [
            file for file in save_dir.iterdir() if file.match("*%s*" % block_name)
        ]

        ids_block = []
        for file in all_files:
            file = file.stem
            ids_block.append(int(file[-2:]))

        if not ids_block:
            logger.info("no blocks found, starting with block 1")
            ix_block = 1
        else:
            ix_block = np.max(ids_block) + 1
            logger.info("Last block executed: %d" % (ix_block - 1))

    next_block_name = f"{block_name}{ix_block:02.0f}"

    return ix_block, next_block_name


# if __name__=='__main__':
# print('testing load_paradigm_config()[\'script_root\']\noutput:')
# print(load_paradigm_config()['script_root'])
# print('testing load_block_config\noutput:')
# print(load_paradigm_config())
# print('testing get_nextblock_metadata\noutput:')
# print(get_nextblock_metadata())
