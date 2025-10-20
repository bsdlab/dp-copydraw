from fire import Fire

from copydraw.copydraw import CopyDraw
from copydraw.utils.config_loading import load_paradigm_config
from copydraw.utils.logging import logger

logger.setLevel(10)


def init_paradigm(
    data_root: str | None = None,
    script_root: str | None = None,
    serial_nr: str | None = "COM4",
):
    # load configs/paradigm_config.yaml
    cfg = load_paradigm_config(data_root=data_root, script_root=script_root)

    paradigm = CopyDraw(
        cfg["data_root"],
        cfg["script_root"],
        # nr of screen, stored in configs/paradigm_configs.yaml
        screen_ix=cfg["screen_ix"],
        serial_nr=serial_nr,
    )

    # Session folder needs to be initialized for copydraw separately
    paradigm.init_session(cfg["session"])

    return paradigm


def main(
    for_calibration: bool = False,
    stim: str | None = None,
    block_nr: int | None = None,
    serial_nr: str | None = "COM4",
):
    serial_nr = None if serial_nr == "None" else serial_nr
    paradigm = init_paradigm(serial_nr=serial_nr)
    logger.debug(f"{paradigm}")

    if for_calibration:
        paradigm.for_calibration = for_calibration
        paradigm.block_settings["n_trials"] = 1
        paradigm.block_settings["letter_time"] = 0.2

    paradigm.exec_block(stim=stim, block_nr=block_nr)


if __name__ == "__main__":
    Fire(main)
