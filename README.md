# Corporación Favorita Grocery Sales Forecasting

В данном проекте реализовано прогнозирование временных рядов из соревнования на kaggle [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/overview).

**Для ускорения инференса, отобраны 100 прогнозируемых объектов (Store+Item) и датасет с историей продаж. Промо, цена на нефть и тд не используются.*

## Инструкция по запуску
```

git clone --branch checkpoint_2 https://github.com/simonyelisey/grocery-sales-forecasting.git           # clone repo
cd grocery-sales-forecasting
make get_data                                                                     # download sql dump and pretrained model
make build                                                                        # build/rebuild servces
make up                                                                           # create and start containers
```

## Основная схема
1. Генерация признаков;
2. Создание таргета;
3. Прогноз.

## Генерация признаков
`feature_generation.py`

Генерируеются следующие типы признаков:
1. Производится sin/cos трансформация циклических признаков на основе даты (день, месяц и тд)
2. Агрегации в скользящем окне
3. Значения в окрестности прошлого года
4. Лаговые признаки
5. Бинарный признак наличия праздника + число периодов до ближайшего праздника

## Создание таргета
`target_generation.py`

Сдвигаем значения сырого таргета на n-значений в прошлое, где n - горизонт прогнозирования.
Каждому значению таргета должны соответсвовать признаки известные на n периодов назад.

## Валидация модели
Для валидации модели использовался последний год тренировочного периода по стратегии **TimeSeriesKFold**

## Финальная модель и гиперпараметры
Для прогнозирования используется модель CatBoostRegressor со следующими гиперпараметрами:
- iterations = 500,
- best_model_min_trees = 20,
- loss_function = RMSE,
- max_depth = 4.

## Метрики
MedianAPE = 0.4

WAPE = 0.47
