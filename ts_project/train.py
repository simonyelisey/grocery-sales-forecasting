import os
import warnings

import feature_generation
import hydra
import pandas as pd
import target_generation
from catboost import CatBoostRegressor
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Функция реализует обучения модели.
    """
    warnings.filterwarnings("ignore")

    drop_features = cfg["modeling"]["drop_columns"]
    target = cfg["modeling"]["target"]

    data = pd.read_parquet(cfg["paths"]["sells"])

    # generate features
    features = feature_generation.apply_feature_generation(
        data=data,
        target=cfg["constants"]["raw_target"],
        predicting_unit=cfg["constants"]["predicting_unit"],
        date_col=cfg["constants"]["date_col"],
        rolling_windows=cfg["feature_generation"]["rolling_windows"],
    )

    # create target
    features = target_generation.create_target(
        data=features,
        horizont=cfg["constants"]["horizont"],
        raw_target=cfg["constants"]["raw_target"],
        predicting_unit=cfg["constants"]["predicting_unit"],
    )

    features_list = list(
        features.loc[:, ~features.columns.isin([target] + drop_features)].columns
    )

    train_data = features[~features[target].isna()][features_list]
    train_target = features[~features[target].isna()][target]

    model = CatBoostRegressor(**cfg["catboost_params"])

    model.fit(train_data, train_target)

    model.save_model(os.path.join(cfg["paths"]["models"], "catboost.cbm"), format="cbm")

    print("SUCCES")


if __name__ == "__main__":
    main()
