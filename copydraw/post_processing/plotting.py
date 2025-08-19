from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copydraw.post_processing.recordings_io import files_to_dataframe


def plot_trace_vs_template(
    trace: np.ndarray, template: np.ndarray, ax: plt.Axes | None
) -> plt.Figure:
    """Plot the trace over the template draw to a given ax if provided"""

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(*trace.T, label="trace", color="#ff5555")
    ax.plot(*template.T, label="template", color="#5555ff")

    fig = ax.get_figure()
    return fig


def facet_plot_trace_and_templates(
    df: pd.DataFrame,
    trace_col: str,
    template_col: str,
    n_facet_cols: int = 4,
) -> plt.Figure:
    if df.shape[0] < n_facet_cols:
        nrows = 1
        ncols = df.shape[0]
    else:
        ncols = n_facet_cols
        nrows = int(df.shape[0] // ncols + 0.5)

    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, figsize=(4 * ncols, 2 * nrows)
    )

    stim_map = {1: "ON", 0: "OFF"}

    for ir, ax in zip(range(df.shape[0]), axs.flatten()):
        trace = df.iloc[ir, :][trace_col]
        template = df.iloc[ir, :][template_col]
        block = df.iloc[ir, :]["ix_block"]
        trial = df.iloc[ir, :]["ix_trial"].flatten()[0]

        stim = stim_map[df.iloc[ir, :]["stim"]] if "stim" in df.columns else "OFF"
        tcolor = "#ff5555" if stim == "ON" else "#3333ff"

        ax.set_title(f"{block=}, {trial=}, stim={stim}", color=tcolor)
        fig = plot_trace_vs_template(trace, template, ax=ax)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    files = list(Path("../../data/").rglob("scores_*.pickel"))
    folders = ["../../data/VPtest"]

    for pth in folders:  # files:
        pth = Path(pth)
        print(f"Processing: {pth=}")
        df = files_to_dataframe(pth)
        # df = pickle.load(open(pth, 'rb'))
        fig = facet_plot_trace_and_templates(
            df, trace_col="trace_let", template_col="template_pix"
        )
        # fig.suptitle(f"{pth.stem}")
        fig.savefig(pth.parent.joinpath(f"{pth.stem}.svg"))
