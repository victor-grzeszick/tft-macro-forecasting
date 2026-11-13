"""Datasets for the Temporal Fusion Transformer (pytorch-forecasting TimeSeriesDataSet).

One dataset is built per target and horizon. Each sample consists of an encoder window of 24 months
ending at the forecast origin t and a decoder window covering months t+1 .. t+h.
"""
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

import config


def to_long(panel):
    """Convert the wide monthly panel into the long format the library expects.

    Adds an integer `time_idx` (0 = first month of the panel) and a constant group id, because the
    whole panel is treated as one multivariate series.
    """
    long = panel.reset_index(drop=True).copy()
    long["time_idx"] = np.arange(len(long))
    long["series"] = "macro"
    return long


def training_dataset(train_panel, target, horizon):
    """Training samples for one target and horizon; every sample lies entirely inside `train_panel`.

    All series (including the target's own past) enter as past-observed inputs. The only input known in
    the decoder is the position relative to the forecast origin. Input scalers are fitted on these rows.
    """
    return TimeSeriesDataSet(
        to_long(train_panel),
        time_idx="time_idx",
        target=target,
        group_ids=["series"],
        max_encoder_length=config.ENCODER_LENGTH,
        max_prediction_length=horizon,
        time_varying_unknown_reals=list(train_panel.columns),
        add_relative_time_idx=True,
    )


def evaluation_dataset(train_ds, panel, first_origin):
    """Samples with forecast origins at positions >= `first_origin` of `panel`.

    The dataset inherits every setting and every fitted scaler from `train_ds`, so nothing is
    re-estimated on the evaluation data.
    """
    return TimeSeriesDataSet.from_dataset(train_ds, to_long(panel), stop_randomization=True,
                                          min_prediction_idx=first_origin + 1)
