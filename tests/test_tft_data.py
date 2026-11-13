"""TFT datasets: sample timing and scalers fitted on training data only."""
import numpy as np
import pandas as pd

import config
from data_pipeline import split
from tft_data import evaluation_dataset, to_long, training_dataset


def test_long_format_adds_consecutive_time_index(synthetic_panel):
    long = to_long(synthetic_panel)
    assert long["time_idx"].tolist() == list(range(len(synthetic_panel)))
    assert long["series"].nunique() == 1


def test_sample_counts_and_origins(synthetic_panel):
    """Validation samples start at the last training month; one sample per origin with a full h-step decoder."""
    train = split(synthetic_panel, "train")
    through_val = synthetic_panel.loc[: config.PERIODS["val"][1]]
    for horizon, expected_train, expected_val in [(3, 310, 22), (12, 301, 13)]:
        train_ds = training_dataset(train, "UNRATE", horizon)
        val_ds = evaluation_dataset(train_ds, through_val, first_origin=len(train) - 1)
        assert len(train_ds) == expected_train
        assert len(val_ds) == expected_val
        first_decoder_step = val_ds.decoded_index["time_idx_first_prediction"].min()
        assert through_val.index[first_decoder_step - 1] == pd.Timestamp("2017-12-01")


def test_encoder_ends_at_origin_and_target_is_h_steps_ahead(synthetic_panel):
    train = split(synthetic_panel, "train")
    through_val = synthetic_panel.loc[: config.PERIODS["val"][1]]
    train_ds = training_dataset(train, "UNRATE", 3)
    val_ds = evaluation_dataset(train_ds, through_val, first_origin=len(train) - 1)
    x, (y, _) = next(iter(val_ds.to_dataloader(train=False, batch_size=64)))
    assert x["encoder_cont"].shape[1] == config.ENCODER_LENGTH and y.shape[1] == 3
    np.testing.assert_allclose(y[0].numpy(), through_val["UNRATE"].iloc[336:339].to_numpy(), rtol=1e-5)


def test_scalers_ignore_evaluation_data(synthetic_panel):
    """Input scalers come from the training dataset and are identical whatever follows the training period."""
    train = split(synthetic_panel, "train")
    train_ds = training_dataset(train, "UNRATE", 3)
    shifted = synthetic_panel.copy()
    shifted.loc["2018-01":] += 100.0
    fitted = train_ds.get_parameters()["scalers"]
    for panel in (synthetic_panel, shifted):
        inherited = evaluation_dataset(train_ds, panel, first_origin=len(train) - 1).get_parameters()["scalers"]
        for name in ["CPIAUCSL", "FEDFUNDS"]:
            assert inherited[name].mean_[0] == fitted[name].mean_[0]
            assert abs(fitted[name].mean_[0] - train[name].mean()) < 1e-9
