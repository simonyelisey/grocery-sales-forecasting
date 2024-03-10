# Corporación Favorita Grocery Sales Forecasting

## Описание проекта

WEB-сервис по прогнозированию спроса на основе данных из соревнования на kaggle [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/overview).

Сервис состоит из двух разделов:
1. Forecasting: в качестве параметра подается код магазина, обученная модель прогнозирует спрос на все товары в этом
магазине и кладет прогнозы в БД;
2. Training: запуск обучения модели. На выходе в папку `/models` созраняется обученная модель.

## Инструкция по запуску
```

git clone --branch checkpoint_2 https://github.com/simonyelisey/grocery-sales-forecasting.git           # clone repo
cd grocery-sales-forecasting
pip install dvc==3.31.1 && pip install dvc_gdrive==2.20.0                       # download sql dump and pretrained model
dvc fetch data/grocery.sql models/catboost.cbm
dvc pull
make build                                                                        # build/rebuild servces
make up                                                                           # create and start containers
make migrate                                                                      # run migrations
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
