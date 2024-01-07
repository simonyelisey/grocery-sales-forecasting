import pandas as pd


def create_target(
    data: pd.DataFrame, horizont: int, raw_target: str, predicting_unit: str
) -> pd.DataFrame:
    """
    Функция создает таргет для обучения и прогнозирования (лаг соответсвующий горизонту прогнозирвоания).

    :param data: pd.DataFrame с историе продаж
    :param horizont: int горизонт прогнозирования
    :param raw_target: str название целевой переменной
    :param predicting_unit: str уровень агрегации продукта

    """
    df = data.copy()

    if horizont < 10:
        df[f"target_0{horizont}"] = df.groupby(predicting_unit)[raw_target].shift(
            -horizont
        )
    else:
        df[f"target_{horizont}"] = df.groupby(predicting_unit)[raw_target].shift(
            -horizont
        )

    return df
