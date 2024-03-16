import datetime
import os

import feature_generation
import pandas as pd
import target_generation
from catboost import CatBoostRegressor
from hydra import compose, initialize

from sql import database_connection


def check_create_table(db_connection, table_name: str, queries_path: str) -> None:
    """
    Проверка присутсвия таблицы в БД и ее создание в случае отсутсвия.

    :param db_connection: соединение с БД
    :param table_name: str название таблицы
    :param queries_path: str путь к запросам по созданию таблиц

    :return: None
    """
    tables = db_connection.show_tables()

    if table_name not in tables:
        with open(os.path.join(queries_path, f"{table_name}.sql"), "r") as f:
            query = f.read()

            db_connection.create_table(query=query)


def main(store_number: int):
    """
    Функция реализует прогнозирование предобученной моделью заданного магазина.

    :param store_number: int код прогнозируемого магазина):
    """
    initialize(version_base=None, config_path="../configs")
    cfg = compose(config_name="config.yaml")

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

    # select only provided store
    data = data[data["store_nbr"] == store_number].reset_index(drop=True)

    if data.shape[0] == 0:
        return f"Data doesn't contain store: {store_number}. Select other store."

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

    next_period_prediction["predicted_from"] = today

    check_create_table(
        db_connection=mydb,
        table_name="predictions",
        queries_path=cfg["sql"]["tables_creation_queries_path"],
    )

    mydb.write_dataframe(table_name="predictions", df=next_period_prediction)

    mydb.close()

    status = f"""
        {datetime.datetime.now()}, success inference.
        \nStore: {store_number}
        \nNumber of items: {data['item_nbr'].nunique()}

        \nPrediction is saved to predictions table in DB.
    """

    return status


if __name__ == "__main__":
    main()
