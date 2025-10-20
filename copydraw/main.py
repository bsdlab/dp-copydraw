from fire import Fire

from copydraw.copydraw import CopyDraw
from copydraw.utils.config_loading import load_paradigm_config
from copydraw.utils.logging import logger

logger.setLevel(10)


def init_paradigm(data_root: str = None, script_root: str = None):
    """
    Initialize a CopyDraw paradigm instance with configuration settings.

    Loads the paradigm configuration from YAML file and creates a CopyDraw instance.

    Parameters
    ----------
    data_root : str, optional
        Root directory for data storage. If None, uses value from config file.
    script_root : str, optional
        Root directory for scripts and templates. If None, it defaults to the copydraw folder.

    Returns
    -------
    CopyDraw
        Initialized CopyDraw paradigm instance
    """
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


def main(for_calibration: bool = False,
         stim: str = None,
         block_nr: int = None
         # stim: None | str = None,
         # block_nr: None | int = None
         ):
    """
    Main function to run the CopyDraw experiment.

    Initializes the paradigm and executes a block of trials. Can be configured
    for calibration mode with reduced trial duration and single trial.

    Parameters
    ----------
    for_calibration : bool, optional
        If True, runs in calibration mode with 1 trial and 0.2s letter time.
    stim : str, optional
        Stimulus identifier for the block.
    block_nr : int, optional
        Block number for the experiment. If None, the next block number is determined automatically.
    """
    paradigm = init_paradigm()
    logger.debug(f"{paradigm}")

    if for_calibration:
        paradigm.for_calibration = for_calibration
        paradigm.block_settings["n_trials"] = 1
        paradigm.block_settings["letter_time"] = 0.2

    paradigm.exec_block(stim=stim, block_nr=block_nr)


if __name__ == "__main__":
    Fire(main)
