import logging

logger = logging.getLogger(__name__)

def analyze_data(df):
    """Анализ данных: структура, типы, пропущенные значения."""
    logger.info("Начат анализ данных.")
    print("Первые 5 строк данных:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nПропущенные значения:")
    print(df.isnull().sum())
    print("\nСтатистика данных:")
    print(df.describe())
    logger.info("Анализ данных завершён.")

def check_missing_values(df):
    """Проверка на пропущенные значения."""
    logger.info("Проверка на пропущенные значения.")
    if df.isnull().sum().sum() > 0:
        logger.warning("Пропущенные значения обнаружены.")
        print("Есть пропущенные значения.")
    else:
        logger.info("Пропущенных значений нет.")
        print("Пропущенных значений нет.")