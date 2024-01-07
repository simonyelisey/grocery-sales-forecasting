import gc

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def transform_cyclical_features(data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Функция производит sin/cos преобразование циклических признаков.

    :param data: pd.DataFrame с историей продаж
    :param date_col: str название колонки с датой

    :return: pd.DataFrame с трансформированными циклическими признаками
    """
    cyclical_features = ["dayofweek", "day", "dayofyear", "week", "month"]

    n_unique_values_map = {
        "dayofweek": 7,
        "day": 31,
        "dayofyear": 365,
        "week": 52,
        "month": 12,
    }

    data[date_col] = pd.to_datetime(data[date_col])

    data["year"] = data[date_col].dt.year
    data["month"] = data[date_col].dt.month
    data["dayofweek"] = data[date_col].dt.day_of_week
    data["day"] = data[date_col].dt.day
    data["dayofyear"] = data[date_col].dt.day_of_year
    data["week"] = data[date_col].dt.isocalendar().week

    for feature in cyclical_features:
        data[f"{feature}_sin"] = np.sin(
            (data[feature] * 2 * np.pi) / n_unique_values_map[feature]
        )
        data[f"{feature}_cos"] = np.cos(
            (data[feature] * 2 * np.pi) / n_unique_values_map[feature]
        )

    return data


def create_roll_features(
    data: pd.DataFrame, target: str, gb_cols: list, windows: list
) -> pd.DataFrame:
    """
    Функция создает признаки аггрегации по целевой переменной в скользящем окне.

    :param data: pd.DataFrame с историей продаж
    :param target: str название целевой переменной
    :param gb_cols: list колонок для группировки
    :param windows: list размеров скользящего окна в зависимости от уровня аггрегации

    :return: pd.DataFrame с новыми признаками
    """
    # абсолютное изменение продаж
    data.loc[:, "abs_change"] = data.groupby(gb_cols)[target].transform(
        lambda x: np.abs(x - x.shift())
    )

    for window in windows:
        for quantile in [0.1, 0.5, 0.9]:
            # квантили
            data.loc[:, f"{target}_roll_q{quantile}_w{window}"] = data.groupby(gb_cols)[
                target
            ].transform(
                lambda x: x.rolling(window=window, min_periods=window).quantile(
                    quantile
                )
            )
        # среднее counts
        data.loc[:, f"{target}_roll_mean_w{window}"] = data.groupby(gb_cols)[
            target
        ].transform(lambda x: x.rolling(window=window, min_periods=window).mean())
        # среднее не нулевых counts
        data.loc[:, f"non_zero_{target}_roll_mean_w{window}"] = data.groupby(gb_cols)[
            target
        ].transform(
            lambda x: x.rolling(window=window, min_periods=window).apply(
                lambda y: y.replace(0, np.nan).mean()
            )
        )
        # std counts
        data.loc[:, f"{target}_roll_std_w{window}"] = data.groupby(gb_cols)[
            target
        ].transform(lambda x: x.rolling(window=window, min_periods=window).std())
        # сумма квадратов
        data.loc[:, f"{target}_roll_squared_sum_w{window}"] = data.groupby(gb_cols)[
            target
        ].transform(
            lambda x: np.square(x).rolling(window=window, min_periods=window).sum()
        )
        # доля не нулей
        data.loc[:, f"{target}_roll_nonzero_prop_w{window}"] = data.groupby(gb_cols)[
            target
        ].transform(
            lambda x: (x != 0).rolling(window=window, min_periods=window).mean()
        )
        # доля нулей
        data.loc[:, f"{target}_roll_zero_prop_w{window}"] = (
            1 - data.loc[:, f"{target}_roll_nonzero_prop_w{window}"]
        )

    return data


def create_previous_year_locality_features(
    data: pd.DataFrame, target: str, predicting_unit: str
) -> pd.DataFrame:
    """
    Функция считает значения продаж в окрестности предыдущего года (7 дней до/после).

    :param data: pd.DataFrame с историе продаж
    :param target: str название целевой переменной
    :param predicting_unit: str уровень агрегации продукта

    :return: pd.DataFrame с новыми признаками
    """
    # точки в окрестности прошлого года
    lags = range(358, 373)

    # считаем значения в окрестности прошлого года
    for lag in lags:
        data[f"{target}_y_ago_lag{lag}"] = (
            data.groupby(predicting_unit)[target]
            .transform(lambda x: x.shift(lag))
            .values
        )

    return data


def create_lag_features(
    data: pd.DataFrame, target: str, predicting_unit: str
) -> pd.DataFrame:
    """
    Функция считает лаг признаки в диапазоне min_lag - max_lag
    и значения y_{lag}/y_{lag+1}.

    :param data: pd.DataFrame с историей продаж
    :param target: str целевая переменная
    :param predicting_unit: str уровень агрегации продукта

    :return: pd.DataFrame с новыми признаками
    """

    # считаем лаги
    for lag in range(0, 15):
        data[f"{target}_lag{lag}"] = data.groupby(predicting_unit)[target].transform(
            lambda x: x.shift(lag)
        )

        if lag < 14:
            data[f"{target}_lag{lag}/lag{lag + 1}"] = data.groupby(predicting_unit)[
                target
            ].transform(lambda x: x.shift(lag) / x.shift(lag + 1))

    # исправляем inf (вызванные x/0, где x > 0) и лишние NaN (вызванные 0/0)
    for lag in range(0, 14):
        # если оба значения == 0, в {target}_lag{lag}/lag{lag+1} будет NaN, поэтому заполняем их единицей
        data.loc[
            (data[f"{target}_lag{lag}"] == 0) & (data[f"{target}_lag{lag + 1}"] == 0),
            f"{target}_lag{lag}/lag{lag + 1}",
        ] = 1
        # если числитель > 0 и знаменатель == 0, в {target}_lag{lag}/lag{lag+1} inf меняем на 2
        data.loc[
            (data[f"{target}_lag{lag}"] > 0) & (data[f"{target}_lag{lag + 1}"] == 0),
            f"{target}_lag{lag}/lag{lag + 1}",
        ] = 2

    return data


def calculate_nearest_holidays(
    date, holidays_data: pd.DataFrame, date_col: str, future=True
):
    """
    Функция счтает количество периодов до ближайшего праздника/нового года.

    :param date: pd.datetime дата
    :param holidays_data: pd.DataFrame с датами праздников в России
    :param date_col: str название колонки с датой в holidays_data
    :param future: bool - True(default) - до ближайшего праздника в будущем
                        - False - до ближайшего празднкика в прошлом

    :return: pd.Series с дистанциями до ближайшего праздника
    """

    # дней до всех праздников
    days_to_holiday = (date - holidays_data[date_col]).dt.days

    # периодов до ближайшего праздника
    periods_to_hol = np.min(
        np.abs(days_to_holiday[(days_to_holiday == 0) | (days_to_holiday < 0)])
    )
    if not future:
        periods_to_hol = np.min(
            np.abs(days_to_holiday[(days_to_holiday == 0) | (days_to_holiday > 0)])
        )

    return periods_to_hol


def create_holidays_features(
    data: pd.DataFrame, holidays_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Функция создает признаки на основе праздников:
        - дистанция до ближайшего праздника в будущем/в прошлом

    :param data: pd.DataFrame с истрией продаж
    :param holidays_data: pd.DataFrame с датами праздников в России

    :return: pd.DataFrame с новыми признаками holidays
    """
    # date -> pd.to_datetime
    data["date"] = pd.to_datetime(data["date"])

    # отложим только уникальные даты
    unique_dates = data[["date"]].drop_duplicates()

    # признаки до праздников
    unique_dates["d_to_holiday_future"] = unique_dates["date"].transform(
        lambda x: calculate_nearest_holidays(
            date=x, holidays_data=holidays_data, date_col="date"
        )
    )
    unique_dates["d_to_holiday_past"] = unique_dates["date"].transform(
        lambda x: calculate_nearest_holidays(
            date=x, holidays_data=holidays_data, date_col="date", future=False
        )
    )

    # date -> str
    unique_dates["date"] = unique_dates["date"].astype(str)
    data["date"] = data["date"].astype(str)

    # соединяем признаки к основном датафрейму
    data = data.merge(unique_dates, how="left")

    return data


def apply_feature_generation(
    data: pd.DataFrame,
    target: str,
    predicting_unit: str,
    date_col: str,
    rolling_windows: list,
) -> pd.DataFrame:
    """"""
    data = (
        data.groupby([predicting_unit, date_col])[target]
        .sum()
        .unstack(fill_value=0)
        .stack()
        .reset_index()
        .rename(columns={0: target})
    )

    # sells
    sells_data = data[[predicting_unit, date_col, target]]
    sells_data = sells_data.sort_values(by=[predicting_unit, date_col]).reset_index(
        drop=True
    )

    # transform cyclical features
    cyclycal_data = transform_cyclical_features(
        data=sells_data[[date_col]].drop_duplicates(), date_col=date_col
    )

    # create rolling aggregations
    rolling_features = Parallel(n_jobs=3)(
        delayed(create_roll_features)(
            data=sells_data.copy(),
            target=target,
            gb_cols=[predicting_unit],
            windows=[w],
        )
        for w in rolling_windows
    )

    rolling_features_df = pd.concat(rolling_features, axis=1)
    rolling_features_df = rolling_features_df.loc[
        :, ~rolling_features_df.columns.duplicated()
    ].copy()

    # previous year locality feature
    prev_year_locality = create_previous_year_locality_features(
        data=sells_data.copy(), target=target, predicting_unit=predicting_unit
    )

    # lag features
    lag_features = create_lag_features(
        data=sells_data.copy(), target=target, predicting_unit=predicting_unit
    )

    # merge all features

    # lags
    lag_features = lag_features.sort_values(by=[predicting_unit, date_col]).reset_index(
        drop=True
    )
    lag_features_columns = lag_features.columns[
        ~lag_features.columns.isin(sells_data.columns)
    ]
    sells_data = pd.concat([sells_data, lag_features[lag_features_columns]], axis=1)

    del lag_features
    gc.collect()

    # previous year locality
    prev_year_locality = prev_year_locality.sort_values(
        by=[predicting_unit, date_col]
    ).reset_index(drop=True)
    prev_year_locality_columns = prev_year_locality.columns[
        ~prev_year_locality.columns.isin(sells_data.columns)
    ]
    sells_data = pd.concat(
        [sells_data, prev_year_locality[prev_year_locality_columns]], axis=1
    )

    del prev_year_locality
    gc.collect()

    # rollings
    rolling_features_df = rolling_features_df.sort_values(
        by=[predicting_unit, date_col]
    ).reset_index(drop=True)
    rolling_features_df_columns = rolling_features_df.columns[
        ~rolling_features_df.columns.isin(sells_data.columns)
    ]
    sells_data = pd.concat(
        [
            sells_data.reset_index(drop=True),
            rolling_features_df[rolling_features_df_columns],
        ],
        axis=1,
    )

    del rolling_features_df
    gc.collect()

    # cyclical features
    sells_data = sells_data.merge(cyclycal_data, on=[date_col])

    return sells_data
