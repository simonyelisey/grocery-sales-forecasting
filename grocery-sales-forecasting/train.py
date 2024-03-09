import datetime
import os

import feature_generation
import metrics
import mlflow
import target_generation
from catboost import CatBoostRegressor
from hydra import compose, initialize
from mlflow.models import infer_signature

from sql import database_connection


def main():
    """
    Функция реализует обучения модели & логгирование метрик.
    """
    initialize(version_base=None, config_path="../configs")
    cfg = compose(config_name="config.yaml")
    cfg_mlflow = compose(config_name="mlflow.yaml")
    cfg_catboost = compose(config_name="catboost_params.yaml")

    drop_features = cfg["modeling"]["drop_columns"]
    target = cfg["modeling"]["target"]

    mydb = database_connection.SoccerDatabase(
        host=os.environ["POSTGRES_HOST"],
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        port=os.environ["POSTGRES_PORT"],
    )

    data = mydb.query("select * from sales")
    mydb.close()
    # data = pd.read_parquet('data/sells.parquet')

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

    model = CatBoostRegressor(**cfg_catboost["catboost_params"])

    model.fit(train_data, train_target)

    model.save_model(os.path.join(cfg["paths"]["models"], "catboost.cbm"), format="cbm")

    print(
        f"{datetime.datetime.now()}, success train. Model is saved to {cfg['paths']['models']} folder."
    )

    if cfg_mlflow["mlflow"]["logging"]:
        # set tracking server uri for logging
        mlflow.set_tracking_uri(uri=cfg_mlflow["mlflow"]["logging_uri"])

        # create a new MLflow Experiment
        mlflow.set_experiment(cfg_mlflow["mlflow"]["experiment_name"])

        # start an MLflow run
        with mlflow.start_run():
            # log the hyperparameters
            mlflow.log_params(cfg_catboost["catboost_params"])

            # calculate metrics
            all_metrics = metrics.Metrics(
                actual=train_target, prediction=model.predict(train_data)
            )

            # Log the loss metric
            mlflow.log_metric("WAPE", all_metrics.wape())
            mlflow.log_metric("MedianApe", all_metrics.median_ape())
            mlflow.log_metric("MAE", all_metrics.mae())

            # set a tag
            mlflow.set_tag(
                cfg_mlflow["mlflow"]["tag_name"], cfg_mlflow["mlflow"]["tag_value"]
            )

            # Infer the model signature
            signature = infer_signature(train_data, model.predict(train_data))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=cfg_mlflow["mlflow"]["artifact_path"],
                signature=signature,
                input_example=train_data,
                registered_model_name=cfg_mlflow["mlflow"]["registered_model_name"],
            )


if __name__ == "__main__":
    main()
