from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from rich.progress import track
from sklearn.metrics import roc_auc_score

from copydraw.post_processing.behavioral_scores import dtw_py
from copydraw.post_processing.behavioral_scores.projection import \
    projection_performance_metrics
from copydraw.post_processing.recordings_io import load_copydraw_record_yaml
from copydraw.utils.logging import logger


def kin_scores(var_pos, delta_t, sub_sampled=False):
    kin_res = {"pos_t": var_pos}

    velocity, velocity_mag = deriv_and_norm(var_pos, delta_t)
    accel, accel_mag = deriv_and_norm(velocity, delta_t)
    jerk, jerk_mag = deriv_and_norm(accel, delta_t)

    N = len(var_pos)

    # average
    # divided by number of timepoints,
    # because delta t was used to calc instead of total t
    kin_res["speed"] = np.sum(velocity_mag) / N
    kin_res["acceleration"] = np.sum(accel_mag) / N
    kin_res["velocity_x"] = np.sum(np.abs(velocity[:, 0])) / N
    kin_res["velocity_y"] = np.sum(np.abs(velocity[:, 1])) / N
    kin_res["acceleration_x"] = np.sum(np.abs(accel[:, 0])) / N
    # matlab code does not compute y values, incorrect indexing
    kin_res["acceleration_y"] = np.sum(np.abs(accel[:, 1])) / N

    # isj
    # in matlab this variable is overwritten
    isj_ = np.sum((jerk * delta_t**3) ** 2, axis=0)
    kin_res["isj_x"], kin_res["isj_y"] = isj_[0], isj_[1]
    kin_res["isj"] = np.mean(isj_)

    kin_res["speed_t"] = velocity * delta_t
    kin_res["accel_t"] = accel * delta_t**2
    kin_res["jerk_t"] = jerk * delta_t**3

    if sub_sampled:
        kin_res = {f"{k}_sub": v for k, v in kin_res.items()}

    return kin_res


def computeScoreSingleTrial(traceLet, template, trialTime):
    trial_results = {}

    # compute avg delta_t
    delta_t = trialTime / traceLet.shape[0]
    trial_results["delta_t"] = delta_t

    # Kinematic scores
    kin_res = kin_scores(traceLet, delta_t)
    trial_results = {**trial_results, **kin_res}

    # sub sample
    traceLet_sub = movingmean(traceLet, 5)
    traceLet_sub = traceLet_sub[::3, :]  # take every third point
    kin_res_sub = kin_scores(traceLet_sub, delta_t * 3, sub_sampled=True)
    trial_results = {**trial_results, **kin_res_sub}

    # dtw
    dtw_res = dtw_py.dtw_features(traceLet, template)
    trial_results = {**trial_results, **dtw_res}

    # misc
    trial_results["dist_t"] = _w_to_dist_t(
        trial_results["w"].astype(int),
        trial_results["pos_t"],
        template,
        trial_results["pathlen"],
    )

    # normalize distance dt by length of copied template (in samples)
    trial_results["dt_norm"] = trial_results["dt_l"] / (trial_results["pathlen"] + 1)

    # get length of copied part of the template (in samples)
    trial_results["len"] = (trial_results["pathlen"] + 1) / len(template)

    return trial_results


def _w_to_dist_t(w, trace, template, pathlen, template_idx_in_w: int = 0):
    """This is a copy of how dist_t is computed in matlab.

    with the added feature of informing it as to whether w is indexed the other way ie [trace_idxs, template_idxs]   # noqa
    """

    tmp1 = template[w[:pathlen, template_idx_in_w], :]
    tmp2 = trace[w[:pathlen, int(not template_idx_in_w)]]
    dist_t = np.sqrt(np.sum((tmp1 - tmp2) ** 2, axis=1))
    return dist_t


def process_trial(trial_file: Path, use_longest_only: bool = True):
    """Actual trial post processing takes place here."""

    logger.info(f"Loading trial data for {trial_file=}")
    res = load_copydraw_record_yaml(trial_file)
    res["stim"] = derive_stim(trial_file)

    # scale the template to how it would be on the screen in real pixels
    # as the trace_let is recorded in screen pixel coords
    temp = res["template_pix"] * res["template_scaling"]
    scaled_template = temp - (
        res["template_pos"] / res["scaling_matrix"][0, 0] / res["template_scaling"]
    )

    res["scaled_template"] = scaled_template

    trace = (
        [
            tr
            for tr in res["traces_pix"]
            if len(tr) == max([len(e) for e in res["traces_pix"]])
        ][0]
        if use_longest_only
        else res["trace_let"]
    )

    # do dtw etc
    scores = computeScoreSingleTrial(
        np.asarray(trace), res["scaled_template"], res["trial_time"]
    )

    return {**res, **scores}


def test_plot_template_vs_tracelet(temp: np.ndarray, trace: np.ndarray):
    """Debugging function to check if the scaling is applied correctly"""

    plt.plot(temp[:, 0], temp[:, 1], color="#5555ff", label="template")
    plt.plot(trace[:, 0], trace[:, 1], color="#ff5555", label="trace")
    plt.legend()


def derive_stim(fpath: Path) -> str:
    """For a given session dir get the stim value for a given block"""
    if fpath.stem.startswith("STIM_OFF_"):
        return "off"
    elif fpath.stem.startswith("STIM_ON_"):
        return "on"
    else:
        logger.warning(f"Cannot derive stim state from {fpath.stem=}")
        return "unknown"


def deriv_and_norm(var, delta_t):
    """
    Given an array (var) and timestep (delta_t), computes the derivative
    for each timepoint and returns it (along with the magnitudes)

    """
    # This is not the same as the kinematic scores in the matlab code!
    deriv_var = np.diff(var, axis=0) / delta_t
    deriv_var_norm = np.linalg.norm(deriv_var, axis=1)
    return deriv_var, deriv_var_norm


# TODO: find an implementation for this
def movingmean(arr, w_size):
    """This is trying to mimic some of the functionality from:
    https://uk.mathworks.com/matlabcentral/fileexchange/41859-moving-average-function
    which (I think) is the function used in compute_scoreSingleTrial.m
    (not in matlab by default). Returns an array of the same size by shrinking
    the window for the start and end points."""

    # round down even window sizes
    if w_size % 2 == 0:
        w_size -= 1

    w_tail = np.floor(w_size / 2)

    arr_sub = np.zeros_like(arr)

    for j, col in enumerate(arr.T):  # easier to work with columns like this
        for i, val in enumerate(col):
            # truncate window if needed
            start = i - w_tail if i > w_tail else 0
            stop = i + w_tail + 1 if i + w_tail < len(col) else len(col)
            s = slice(int(start), int(stop))

            # idxs reversed bc .T
            arr_sub[i, j] = np.mean(col[s])

    return arr_sub


def test_plot_last_test_trial_for_calibration():
    # Load last trial of last run if VPtest
    pth = list(Path("./data/VPtest/copyDraw/raw_behavioral/").rglob("*_trial*.yaml"))[
        -1
    ]
    res = load_copydraw_record_yaml(pth)

    temp = res["template_pix"] * res["template_scaling"]
    trace = res["trace_let"]

    # pos shift
    sm = res["scaling_matrix"]
    ttemp = temp - res["template_pos"] / sm[0, 0] / res["template_scaling"]

    test_plot_template_vs_tracelet(ttemp, trace)


def create_copydraw_results_data(
    copydraw_folder: Path,
    feature_set: str = "standard_performance_metrics",
    overwrite: bool = False,
    session: str = "",
) -> tuple[pd.DataFrame, Any]:
    """
    For a given folder, collect all data under raw_behavioral and compose
    into a single copydraw data frame

    Steps include:
    - calculation of kinematic scores (velocity, acceleration, jitter)
    - dtw for calculation of a distance metric
    - fitting of the LDA model for projection
    """

    cfg = yaml.safe_load(open("./configs/paradigm_config.yaml"))

    # The results folder
    res_folder = copydraw_folder.resolve().joinpath("projection_results")
    if res_folder.exists():
        if overwrite:
            res_folder.unlink()
        else:
            q = ""
            while q not in ["y", "n"]:
                q = input(
                    f"There is alread a results folder at {res_folder},"
                    f"do you want to overwrite? [y, n]"
                )

            if q == "y":
                res_folder.unlink()
            else:
                return res_folder

    res_folder.mkdir()

    # kinematics and dtw
    trial_files = list(
        copydraw_folder.joinpath("raw_behavioral").rglob("*_block*_trial*.yaml")
    )
    assert trial_files, f"No trial files found at {copydraw_folder}"
    trial_data = [process_trial(f) for f in track(trial_files)]

    # only get the non iterables, to keep the frame lean
    df_scores = pd.concat(
        [
            pd.DataFrame(
                {
                    k: v
                    for k, v in d.items()
                    if isinstance(v, str) or not isinstance(v, Iterable)
                },
                index=[0],
            )
            for d in trial_data
        ]
    ).reset_index(drop=True)

    df_scores["startTStamp"] = (
        df_scores["start_t_stamp"] - df_scores["start_t_stamp"].min()
    )
    df_scores["startTStamp"] /= 60  # in minutes

    # Get the labels :)
    y, ix_clean, model = projection_performance_metrics(
        scores=df_scores[cfg[feature_set]],
        # scores=df_scores[cfg['EXTENDED_PERFORMANCE_METRICS']],
        block_labels=df_scores["stim"],
        block_indices=df_scores["ix_block"],
        trial_indices=df_scores["ix_trial"],
        detrend=True,
        t_stamps=df_scores["startTStamp"].values,
        reject_outliers=True,
        session=session,
    )

    df_scores["final_label"] = y
    df_scores["final_clean"] = ix_clean

    # store model and
    df_scores.to_hdf(
        res_folder.joinpath("motoric_scores.hdf"), key="joined_motoric_scores"
    )
    joblib.dump(model, res_folder.joinpath("proj_model.joblib"))

    return df_scores, model


def plot_projection_plotly(df: pd.DataFrame):
    ix_clean = df.final_clean
    fig = px.scatter(
        df[ix_clean],
        x="startTStamp",
        y="final_label",
        color="stim",
        marginal_y="box",
    )

    fig.show()


if __name__ == "__main__":
    sessions = [
        "VPtest",
    ]
    for session in sessions:
        df, model = create_copydraw_results_data(
            Path(f"../../data/dareplane/{session}/behavioral/copydraw/"),
            feature_set="standard_performance_metrics",
            session=session,
        )

    import plotly.express as px

    px.scatter(df[df.final_clean], x="startTStamp", y="final_label", color="stim").show(
        renderer="browser"
    )

    # quick feature importance plot
    cfg = yaml.safe_load(open("./configs/paradigm_config.yaml"))
    ex_feat = cfg["extended_performance_metrics"]

    dfeat = pd.DataFrame(
        {"feature": ex_feat[: len(model.coef_[0])], "weight": model.coef_[0]}
    )
    px.bar(dfeat, x="feature", y="weight").show()
