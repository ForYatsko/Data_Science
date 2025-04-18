##### Финальная работа
  
    1. Взять набор данных исходя из ваших интересов.
    
    2. Не используйте датасеты, которые вы уже брали.
    
    3. Описать колонки, какие характеристики.
    
    4. Проведите анализ EDA.
    
    5. Провести предварительную обработку данных, если это необходимо (сделать данные понятными для модели машинного обучения: заполнить пропущенные значения, заменить категориальные признаки и т.д.)
    
    6. Решить задачу сегментации или анализа временного ряда при помощи не менее 5-ти подходов ML. Составьте ансамбль моделей.
    
    7. Решить задачу поиска аномалий.
    
    8. Визуализация. Создать графики ошибок прогнозирования, метрик качества обученной модели и важности признаков.
  
##### Структура проекта

hw6/

📦 crypto_forecasting_project/

├── 📂 data/                       # Данные

│   └── coin_Bitcoin.csv           # Исторические данные о криптовалюте

├── my_log.log                     # Лог-файл с информацией о выполнении программы

├── 📂 output/                     # Вывод результатов

│   ├── 📂 visualizations/         # Графики и визуализации

│   ├── 📂 processed_data/         # Обработанные данные

│   └── metrics.json               # Метрики моделей

├── anomaly_detection.py           # Обнаружение аномалий

├── data_processing.py             # Обработка данных

├── finale.ipynb                   # Реализация в юпитер ноутбуке

├── logging_config.py              # Настройка логирования

├── model_training.py              # Обучение моделей

├── visualization.py               # Функции для визуализации

├── README.md                      # Документация проекта

└── improvements.py                # Улучшения модели (взвешенный ансамбль и др.)


#### Датасет Cryptocurrency Historical Prices: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory/data?select=coin_Bitcoin.csv

#### Этот проект реализует систему анализа исторических данных по криптовалюте Bitcoin, использует различные алгоритмы машинного обучения для прогнозирования цены закрытия, а также выполняет поиск аномалий в данных. 

#### Основные этапы работы:
1. Импорт библиотек и настройка окружения

2. Загрузка и предварительная обработка данных: Используется датасет Исторические данные по криптовалюте Bitcoin загружаются из CSV-файла.

Предварительная обработка: Удаление пропусков и аномальных значений. Добавление новых признаков, таких как лаги (например, Close_shift_1 — значение закрытия за предыдущий день). Масштабирование данных (при необходимости).

3. Анализ данных (EDA)
   
Корреляционный анализ: Оценивается связь между признаками (например, цена открытия, закрытия, объём торгов и др.).
Визуализация распределения данных: Построение гистограмм для анализа распределения цен и других признаков.
Анализ сезонности: Исследование изменений средней цены закрытия по месяцам или другим временным интервалам.

4. Разделение данных
Данные делятся на обучающую и тестовую выборки.

5. Обучение моделей
   
Градиентные бустинги:
Обучение моделей XGBoost, LightGBM, CatBoost.
Вычисление метрики MSE для каждой модели.

ARIMA:
Обучение модели временного ряда для предсказания цен.
Вычисление MSE предсказаний.

LSTM:
Обучение рекуррентной нейронной сети (LSTM) для предсказания цен.
Оценка качества модели на тестовой выборке.

Ансамбль:
Усреднение предсказаний всех моделей для создания ансамбля.
Вычисление метрики MSE для ансамбля.

6. Визуализация прогнозов
   
Построение графика, сравнивающего фактические значения цен закрытия и предсказания ансамбля моделей.

Вывод метрик (например, MSE для каждой модели и ансамбля) на графике.

7. Поиск аномалий:
   
Вычисляется разница между фактическими значениями и предсказаниями ансамбля.
Значения, превышающие заданный порог отклонения, определяются как аномалии.

Визуализация аномалий:
Построение графика, где аномалии выделяются красными точками.
Нормальные значения и предсказания отображаются для сравнения.

8. Сохранение результатов
   
Метрики моделей сохраняются в текстовый файл (metrics.txt).
Построенные графики сохраняются в папке output/visualizations.

##### Резюме этапов

Программа проходит все ключевые шаги анализа данных: от загрузки и подготовки данных, до построения моделей и визуализации результатов. Она позволяет:

Понять структуру данных.

Построить прогнозы на основе нескольких моделей.

Выявить аномалии в данных.

Оценить качество прогнозов.

Этот подход может быть использован для анализа других временных рядов, таких как акции, товарные рынки и экономические показатели.
