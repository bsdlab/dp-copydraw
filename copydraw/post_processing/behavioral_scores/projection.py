import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from copydraw.post_processing.behavioral_scores.chrono_split import (
    ChronoGroupsSplit,
)
from copydraw.utils.logging import logger


def projection_performance_metrics(
    scores,
    block_labels: np.ndarray,
    block_indices: np.ndarray,
    trial_indices: np.ndarray,
    detrend=False,
    t_stamps=None,
    reject_outliers=False,
    session: str = "",
):
    """
    trains LDA (if stim labels are available) or PCA (if labels not available)
    and projects the copyDraw trace features onto the generated subspace

    Parameters
    ----------
    scores : np.array NxD
        array containing the copyDraw behavioral features
    block_labels : np.array Nx1
        stim condition for each of the observations in scores
    block_indices : np.array Nx1
        the block numbers used for the cross validation results
    detrend : bool, optional
        detrend the features in scores across the session. The default is False
    t_stamps : np.array Nx1, optional
        time stamps of the execution time of each of the observations in
        scores. The default is None.
    reject_outliers : bool, optional
        Perform outlier rejection using DBSCAN. After clustering, observations
        not belonging to a cluster are marked as outliers.
        The default is False.
    return_performance : bool, optional
        return decoding performance of the LDA model (AUC).
        The default is False.
    return_proj_model : TYPE, optional
        trained LDA model. The default is False.

    Returns
    -------
    ret : tuple
        [labels,
         onehot_accepted (outliers),
         (decoding_auc,pval)
         trained LDA model
         ].

    """
    if block_labels is not None:
        assert scores.shape[0] == block_labels.shape[0]
        projector = LDA(solver="eigen", shrinkage="auto")
    else:
        logger.info("no block labels provided, performing label projection with PCA")
        projector = PCA(n_components=1)
    # detect outliers

    scores_normalized = StandardScaler().fit_transform(scores)
    if reject_outliers:
        clustering = DBSCAN(eps=4).fit(scores_normalized)
        onehot_accepted = clustering.labels_ != -1
        ix_accepted = np.where(onehot_accepted)[0]
    else:
        onehot_accepted = np.ones((len(scores_normalized)), dtype=int)
        ix_accepted = np.where(onehot_accepted)[0]

    # Note the labels will be used as text, i.e. 'off', 'on'. LDA will sort
    # alphabetically so that positive decission function values belong to
    # 'on' and negative values to 'off'
    l_block_labels = block_labels[onehot_accepted]
    l_scores = scores_normalized[ix_accepted]

    if detrend:
        if t_stamps is None:
            t_stamps = np.arange(scores.shape[0])
        scores_detrended = scores_normalized
        for ix_feat, c_feat in enumerate(l_scores.T):
            c_model_detrend = LinearRegression().fit(
                t_stamps[ix_accepted].reshape(-1, 1), c_feat
            )
            scores_detrended[:, ix_feat] -= c_model_detrend.predict(
                t_stamps.reshape(-1, 1)
            )
    else:
        scores_detrended = scores_normalized

    # np.save("block_indices.npy", block_indices)
    # np.save("scores_detrended.npy", scores_detrended)
    # np.save("l_scores.npy", l_scores)
    # np.save("ix_accepted.npy", ix_accepted)
    # np.save("scores_normalized.npy", scores_normalized)
    # np.save("t_stamps.npy", t_stamps)
    # np.save("l_block_labels.npy", l_block_labels)
    # np.save("onehot_accepted.npy", onehot_accepted)

    # This was the old projection, now I use a more uniform approach, both training and projecting
    # on the same piece of data
    # projector.fit(l_scores, l_block_labels)
    # labels = projector.transform(
    #     StandardScaler().fit_transform(scores_detrended)
    # )
    projector.fit(scores_detrended[ix_accepted], l_block_labels)
    labels = projector.decision_function(
        StandardScaler().fit_transform(scores_detrended)
    )
    create_and_store_fit_details(
        l_scores,
        scores_detrended,
        l_block_labels,
        block_indices,
        trial_indices,
        ix_accepted,
        t_stamps,
        session=session,
    )
    # px.scatter(df, x="t", y="y", color="stim", facet_col="proj").show()

    return labels, onehot_accepted, projector


def create_and_store_fit_details(
    l_scores,
    scores_detrended,
    l_block_labels,
    block_indices,
    trial_indices,
    ix_accepted,
    t_stamps,
    session="",
):
    projector = LDA(solver="eigen", shrinkage="auto")

    dfs = []

    l_scores_dt = scores_detrended[ix_accepted]

    selected_bi = block_indices.to_numpy()[ix_accepted]
    selected_ti = trial_indices.to_numpy()[ix_accepted]
    sel_t_stamps = t_stamps[ix_accepted]
    l_block_labels = l_block_labels.to_numpy()

    # also add a cross validation
    cv = ChronoGroupsSplit()
    splits = cv.split(l_scores, l_block_labels, selected_bi)
    for i, (_, st) in enumerate(splits):
        try:
            print(
                f"split {i} - test blocks {np.unique(selected_bi[st])} with "
                f"stim {np.unique(l_block_labels[st])}"
            )
        except:
            import pdb

            pdb.set_trace()

    for train_data_str, train_data in zip(
        ("normal", "normal_detrended"), (l_scores, l_scores_dt)
    ):
        projector = LDA(solver="eigen", shrinkage="auto")
        projector.fit(train_data, l_block_labels)

        for src_data_str, src_data in zip(
            ("normal", "normal_detrended"), (l_scores, l_scores_dt)
        ):
            # Get the projections on detrended data for reference
            for proj_str in ("transform", "decision_function"):
                if proj_str == "transform":
                    transformer = projector.transform

                elif proj_str == "decision_function":
                    transformer = projector.decision_function

                print(f"projecting {src_data_str=} with {proj_str=}, {train_data_str=}")
                dw = proj_to_data_frame(
                    transformer(src_data),
                    l_block_labels,
                    sel_t_stamps,
                    proj_str=proj_str,
                    train_data_str=train_data_str,
                    src_data_str=src_data_str,
                    split_str="all",
                )
                dw["block"] = selected_bi
                dw["ix_trial"] = selected_ti
                dfs.append(dw)

                for i, (ix_train, ix_test) in enumerate(splits):
                    projector = LDA(solver="eigen", shrinkage="auto")
                    projector.fit(train_data[ix_train], l_block_labels[ix_train])
                    if proj_str == "transform":
                        transformer = projector.transform

                    elif proj_str == "decision_function":
                        transformer = projector.decision_function

                    dw = proj_to_data_frame(
                        transformer(src_data[ix_test]),
                        l_block_labels[ix_test],
                        sel_t_stamps[ix_test],
                        proj_str=proj_str,
                        train_data_str=train_data_str,
                        src_data_str=src_data_str,
                        split_str=str(i),
                    )
                    dw["block"] = selected_bi[ix_test]
                    dw["ix_trial"] = selected_ti[ix_test]
                    dfs.append(dw)

    df = pd.concat(dfs)

    df.to_csv(f"{session}_projection_scores.csv", index=False)


def proj_to_data_frame(
    yproj,
    l_block_labels,
    t_stamps,
    proj_str="",
    train_data_str="",
    src_data_str="",
    split_str="",
) -> pd.DataFrame:
    y = yproj.flatten()
    return pd.DataFrame(
        dict(
            y=y,
            stim=l_block_labels,
            t=t_stamps,
            proj=[proj_str] * len(y),
            train_data=[train_data_str] * len(y),
            src_data=[src_data_str] * len(y),
            split=split_str,
        )
    )


# block_indices = np.load("block_indices.npy")
# scores_detrended = np.load("scores_detrended.npy")
# l_scores = np.load("l_scores.npy")
# ix_accepted = np.load("ix_accepted.npy")
# scores_normalized = np.load("scores_normalized.npy")
# t_stamps = np.load("t_stamps.npy")
# l_block_labels = np.load("l_block_labels.npy", allow_pickle=True)
# onehot_accepted = np.load("onehot_accepted.npy")
