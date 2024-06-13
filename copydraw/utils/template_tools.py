import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from copydraw.utils.logging import logger
from pathlib import Path


def scale_to_norm_units(array, scaling_matrix=None):
    """
    for scaling matrix to monitor norm units: scales input array between 0 and 1
    and then translates to origin in 1,-1 coord system

    """

    # translation require an extra col of 1s
    temp_array = np.ones([array.shape[0],array.shape[1]+1])

    # is this copying or referencing? Check it!
    temp_array[:, 0] = array[:, 0]
    temp_array[:, 1] = array[:, 1]

    if scaling_matrix is None:
        scaling_matrix = np.identity(3)

        # minmax scaling
        # scaling
        S_x = 1/(np.max(array[:, 0]) - np.min(array[:, 0]))
        S_y = 1/(np.max(array[:, 1]) - np.min(array[:, 1]))

        # pre scaling translation
        T_x = -np.min(array[:, 0])
        T_y = -np.min(array[:, 1])

        # translate to origin in norm units
        # post scaling translation
        t_x = -0.5
        t_y = -0.5

        scaling_matrix[0, 0] = S_x
        scaling_matrix[1, 1] = S_y

        scaling_matrix[0, 2] = T_x * S_x + t_x
        scaling_matrix[1, 2] = T_y * S_y + t_y

    scaled_matrix = np.matmul(scaling_matrix, temp_array.T).T[:, :-1]
    return scaled_matrix, scaling_matrix


def smooth(shape, return_df=False):
    # create img
    df = pd.DataFrame(shape, columns=['x', 'y'])
    pd.options.mode.chained_assignment = None
    df['dx'] = df.x.diff()
    df['dy'] = df.y.diff()
    df['dxma'] = df.dx.rolling(2).mean()
    df['dyma'] = df.dy.rolling(2).mean()
    df['dxs'] = 0
    df['dys'] = 0
    df['dxs'][:-1] = (df.dx - df.dxma)[1:]
    df['dys'][:-1] = (df.dy - df.dyma)[1:]
    df['dxm'] = df.x + df.dxs
    df['dym'] = df.y + df.dys

    # first row will be nans, need to replace with og starting points?
    df['dxm'][0] = df['x'][0]
    df['dym'][0] = df['y'][0]

    pd.options.mode.chained_assignment = 'warn'
    return df if return_df else df[['dxm','dym']].to_numpy()


def template_to_image(template, fname, path, for_calibration: bool = False,
                       **kwargs):

    # if template images dir doesn't exists make it
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    fullpath = path.joinpath(f'{fname}.png')

    if not fullpath.exists():
        fig = plt.figure(figsize=(16, 10))
        # fig = plt.figure(figsize=(16, 9), frameon=False)

        logger.debug(f"Creating new template png for {fname=} {kwargs=}")

        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.plot(template.T[0], template.T[1], **kwargs)

        if for_calibration:
            # Add a point grid
            xx, yy = np.meshgrid(
                np.linspace(template.T[0].min(), template.T[0].max(), 10),
                np.linspace(template.T[1].min(), template.T[1].max(), 10),
            )
            ax.plot(xx.flatten(), yy.flatten(), 'og')

            # grid corners
            ax.plot(template.T[0].min(), template.T[1].min(), '+y', markersize=44)
            ax.plot(template.T[0].min(), template.T[1].max(), '+y', markersize=44)
            ax.plot(template.T[0].max(), template.T[1].min(), '+y', markersize=44)
            ax.plot(template.T[0].max(), template.T[1].max(), '+y', markersize=44)

            ax.plot(0, 0, '+g', markersize=44)

        plt.savefig(fullpath, format='png', bbox_inches='tight',
                    transparent=True, dpi=300)

    return fullpath


def create_template_order(stimuli_dict, block_settings_dict):
    if block_settings_dict['n_trials'] % stimuli_dict['n_templates'] != 0:
        logger.warning(f'WARNING: {block_settings_dict["n_trials"]} trials means that '
                       f'there will be an uneven number of templates')

    order = [(i % stimuli_dict['n_templates'])
                             for i in range(block_settings_dict['n_trials'])]

    if block_settings_dict['shuffle']:
        random.shuffle(order)

        # reshuffle to remove repeated trials showing
        while 0 in np.diff(np.array(order)):
            random.shuffle(order)

    return order


