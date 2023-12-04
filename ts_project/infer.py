import datetime
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
    Функция реализует прогнозирование предобученной моделью.
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

    next_period_data = features[features[target].isna()]

    model = CatBoostRegressor()

    model.load_model(os.path.join(cfg["paths"]["models"], "catboost.cbm"))

    next_period_prediction = model.predict(next_period_data[features_list])

    next_period_prediction = pd.DataFrame(
        {
            cfg["constants"]["predicting_unit"]: next_period_data[
                cfg["constants"]["predicting_unit"]
            ],
            cfg["constants"]["date_col"]: next_period_data[
                cfg["constants"]["date_col"]
            ],
            "prediction": next_period_prediction,
        }
    ).sort_values(
        by=[cfg["constants"]["predicting_unit"], cfg["constants"]["date_col"]]
    )

    next_period_prediction[cfg["constants"]["date_col"]] = next_period_prediction[
        cfg["constants"]["date_col"]
    ] + pd.DateOffset(days=7)

    today = datetime.datetime.now().date()

    next_period_prediction.to_csv(
        os.path.join(cfg["paths"]["predictions"], f"prediction_{today}"), index=False
    )

    print("SUCCES")


if __name__ == "__main__":
    main()
