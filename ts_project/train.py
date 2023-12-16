import os
import warnings

import feature_generation
import hydra
import metrics
import mlflow
import pandas as pd
import target_generation
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Функция реализует обучения модели & логгирование метрик.
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

    if cfg["mlflow"]["logging"]:
        # set tracking server uri for logging
        mlflow.set_tracking_uri(uri=cfg["mlflow"]["logging_uri"])

        # create a new MLflow Experiment
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        # start an MLflow run
        with mlflow.start_run():
            # log the hyperparameters
            mlflow.log_params(cfg["catboost_params"])

            # calculate metrics
            all_metrics = metrics.Metrics(
                actual=train_target, prediction=model.predict(train_data)
            )

            # Log the loss metric
            mlflow.log_metric("WAPE", all_metrics.wape())
            mlflow.log_metric("MedianApe", all_metrics.median_ape())

            # set a tag
            mlflow.set_tag(cfg["mlflow"]["tag_name"], cfg["mlflow"]["tag_value"])

            # Infer the model signature
            signature = infer_signature(train_data, model.predict(train_data))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=cfg["mlflow"]["artifact_path"],
                signature=signature,
                input_example=train_data,
                registered_model_name=cfg["mlflow"]["registered_model_name"],
            )

    print("SUCCES")


if __name__ == "__main__":
    main()
