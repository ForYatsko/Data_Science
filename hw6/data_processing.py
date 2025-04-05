import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from anomaly_detection import remove_outliers


def ensure_directories():
    """
    Создание необходимых директорий для сохранения обработанных данных и графиков.
    """
    os.makedirs("output/processed_data", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)


def load_data(file_path):
    """
    Загрузка данных из CSV и преобразование столбца 'Date' в datetime.
    """
    logging.info("Загрузка данных...")
    try:
        df = pd.read_csv(file_path)

        # Преобразование столбца 'Date' в datetime
        if 'Date' in df.columns:
            logging.info("Преобразование столбца 'Date' в datetime...")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                logging.warning(f"Обнаружено {invalid_dates} некорректных дат. Они будут удалены.")
                df = df.dropna(subset=['Date'])

        logging.info(f"Данные успешно загружены: {df.shape[0]} строк, {df.shape[1]} колонок.")
        logging.info(f"Первые строки датасета:\n{df.head().to_string()}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise

def describe_data(df):
  """
  Вывод описательной статистики и пропущенных значений.
  """
  logging.info("Описание данных...")
  try:
      describe_stats = df.describe()
      logging.info("Описательная статистика:\n%s", describe_stats)

      # Вывод конкретных значений
      logging.info(f"Среднее значение столбца 'Close': {describe_stats.loc['mean', 'Close']}")
      logging.info(f"Максимальное значение столбца 'Close': {describe_stats.loc['max', 'Close']}")
      logging.info(f"Минимальное значение столбца 'Close': {describe_stats.loc['min', 'Close']}")

      # Пропущенные значения
      logging.info(f"Пропущенные значения:\n{df.isnull().sum()}")
  except Exception as e:
      logging.error(f"Ошибка при описании данных: {e}")
      raise       

def preprocess_data(df):
    """
    Предварительная обработка данных: удаление выбросов, масштабирование и создание лагов.
    """
    logging.info("Начало предварительной обработки данных...")
    try:
        if df.empty:
            raise ValueError("Передан пустой DataFrame.")

        # Удаление строк с некорректными датами и сортировка по времени
        logging.info("Удаление строк с некорректными датами и сортировка...")
        df = df.dropna(subset=['Date']).sort_values('Date')

        # Удаление строк с пропущенными значениями в критических столбцах
        critical_columns = ['Close', 'Volume']
        logging.info(f"Удаление строк с пропущенными значениями в столбцах: {critical_columns}...")
        df = df.dropna(subset=critical_columns)

        # Удаление выбросов
        logging.info("Удаление выбросов в столбце 'Volume'...")
        df = remove_outliers(df, 'Volume')

        # Удаление строк с нулевым значением 'Volume'
        logging.info(f"Тип данных столбца 'Volume' перед фильтрацией: {df['Volume'].dtype}")

        # Преобразование в числовой формат
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # Логирование количества строк с нулевым значением
        logging.info(f"Количество строк с нулевым 'Volume' до удаления: {(df['Volume'] == 0).sum()}")
        logging.info(f"Количество NaN значений в 'Volume': {df['Volume'].isna().sum()}")

        # Удаление строк с NaN и нулевыми значениями
        df = df.dropna(subset=['Volume'])
        df = df[df['Volume'] > 0]

        # Логирование после удаления
        logging.info(f"Количество строк с нулевым 'Volume' после удаления: {(df['Volume'] == 0).sum()}")
        logging.info(f"Размер данных после удаления строк с нулевым 'Volume': {df.shape}")
        logging.info(f"Минимум и максимум 'Volume' после удаления: {df['Volume'].min()}, {df['Volume'].max()}")

        # Масштабирование объёма торгов
        logging.info("Масштабирование данных в столбце 'Volume'...")
        df['Volume'] = df['Volume'] / 1e9  # Приведение к миллиардам
        logging.info("Данные в 'Volume' масштабированы до миллиарда единиц.")

        # Создание лаговых признаков
        logging.info("Создание лаговых признаков...")
        for lag in range(1, 4):
            df[f'Close_shift_{lag}'] = df['Close'].shift(lag)

        # Удаление строк с NaN после создания лагов
        df = df.dropna()
        logging.info("Предварительная обработка завершена.")

        # Сохранение обработанных данных
        processed_dir = "output/processed_data"
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, "processed_data.csv")
        df.to_csv(processed_path, index=False)
        logging.info(f"Обработанные данные сохранены в {processed_path}.")

        return df
    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")
        raise

def eda(df):
    """
    Расширенный разведочный анализ данных (EDA): распределение, корреляции, сезонные тренды и аномалии.
    """
    logging.info("Начало расширенного EDA...")

    try:
        if df.empty:
            raise ValueError("Передан пустой DataFrame.")

        sns.set(style="whitegrid")
        visualization_dir = "output/visualizations"
        os.makedirs(visualization_dir, exist_ok=True)

        # 1. Распределение цены закрытия
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Close'], kde=True, bins=50, color="blue")
            plt.title("Распределение цены закрытия Bitcoin", fontsize=16)
            plt.xlabel("Цена закрытия ($)", fontsize=14)
            plt.ylabel("Частота", fontsize=14)
            save_path = os.path.join(visualization_dir, "eda_close_distribution.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
            plt.close()
            logging.info(f"График распределения цены закрытия сохранён в {save_path}.")
        except Exception as e:
            logging.error(f"Ошибка при построении графика распределения цены закрытия: {e}")
            raise

        # 2. Распределение объёма торгов
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['Volume'], kde=True, bins=50, color="orange")
            plt.title("Распределение объёма торгов Bitcoin", fontsize=16)
            plt.xlabel("Объём торгов (в миллиардах $)", fontsize=14)
            plt.ylabel("Частота", fontsize=14)
            save_path = os.path.join(visualization_dir, "eda_volume_distribution.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
            plt.close()
            logging.info(f"График распределения объёма торгов сохранён в {save_path}.")
        except Exception as e:
            logging.error(f"Ошибка при построении графика распределения объёма торгов: {e}")
            raise

        # 3. Корреляционная матрица
        try:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[['Close', 'Volume', 'Open', 'High', 'Low']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Корреляционная матрица", fontsize=16)
            save_path = os.path.join(visualization_dir, "eda_correlation_matrix.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
            plt.close()
            logging.info(f"Корреляционная матрица сохранена в {save_path}.")
        except Exception as e:
            logging.error(f"Ошибка при построении корреляционной матрицы: {e}")
            raise

        # График объёма торгов
        try:
            logging.info("Агрегация данных по месяцам для графика объёма торгов...")
            df['Month'] = df['Date'].dt.to_period('M')  # Преобразование даты в месяцы
            monthly_data = df.groupby('Month')['Volume'].sum().reset_index()
            monthly_data['Month'] = monthly_data['Month'].astype(str)  # Преобразование в строку

            # Логирование данных для проверки
            logging.info(f"Первые строки monthly_data:\n{monthly_data.head()}")
            logging.info(f"Количество строк в monthly_data: {len(monthly_data)}")
            logging.info(f"Минимум, максимум и среднее 'Volume': {monthly_data['Volume'].min()}, {monthly_data['Volume'].max()}, {monthly_data['Volume'].mean()}")

            # Логирование перед удалением
            logging.info(f"Количество строк с нулевым 'Volume': {(df['Volume'] == 0).sum()}")
            logging.info(f"Примеры строк с нулевым 'Volume':\n{df[df['Volume'] == 0].head()}")

            # Удаление строк с нулевым 'Volume'
            df = df[df['Volume'] > 0]

            # Логирование после удаления
            logging.info(f"Количество строк с нулевым 'Volume' после фильтрации: {(df['Volume'] == 0).sum()}")

            # Проверка на наличие данных
            if monthly_data.empty or monthly_data['Volume'].sum() == 0:
                logging.warning("Данные для графика объёма торгов отсутствуют или равны нулю.")
            else:
                # Построение графика
                plt.figure(figsize=(12, 6))
                plt.bar(monthly_data['Month'], monthly_data['Volume'], color='green', label='Объём торгов (в миллиардах)')
                plt.title('Объём торгов Bitcoin (по месяцам)')
                plt.xlabel('Месяц')
                plt.ylabel('Объём торгов (в миллиардах)')
                plt.xticks(ticks=range(0, len(monthly_data['Month']), 6), rotation=45)
                plt.legend()
                plt.grid()

                # Установка диапазона оси Y
                plt.ylim(0, max(1, monthly_data['Volume'].max() * 1.1))  # Минимум 1 для видимости

                # Сохранение графика
                save_path = os.path.join("output/visualizations", "eda_volume_monthly.png")
                plt.tight_layout()
                plt.savefig(save_path)
                plt.show()
                plt.close()
                logging.info(f"График объёма торгов сохранён в {save_path}.")
        except Exception as e:
            logging.error(f"Ошибка при построении графика объёма торгов: {e}")
            raise

        # 4. Сезонный анализ: распределение по месяцам
        try:
            logging.info("Построение сезонного анализа по месяцам...")
            df['Month'] = df['Date'].dt.month
            monthly_avg_close = df.groupby('Month')['Close'].mean()

            plt.figure(figsize=(10, 6))
            monthly_avg_close.plot(kind='bar', color='skyblue')
            plt.title("Средняя цена закрытия по месяцам", fontsize=16)
            plt.xlabel("Месяц", fontsize=14)
            plt.ylabel("Средняя цена закрытия ($)", fontsize=14)
            save_path = os.path.join(visualization_dir, "eda_monthly_avg_close.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
            plt.close()
            logging.info(f"График сезонного анализа по месяцам сохранён в {save_path}.")
        except Exception as e:
            logging.error(f"Ошибка при построении сезонного анализа по месяцам: {e}")
            raise

        # 5. Аномалии: выбросы в цене закрытия
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=df['Close'], color='skyblue')
            plt.title("Выбросы в цене закрытия Bitcoin", fontsize=16)
            plt.xlabel("Цена закрытия ($)", fontsize=14)
            save_path = os.path.join(visualization_dir, "eda_close_outliers.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
            plt.close()
            logging.info(f"График выбросов в цене закрытия сохранён в {save_path}.")
        except Exception as e:
            logging.error(f"Ошибка при построении графика выбросов: {e}")
            raise

        logging.info("Расширенный EDA успешно завершён.")
    except Exception as e:
        logging.error(f"Ошибка при выполнении расширенного EDA: {e}")
        raise