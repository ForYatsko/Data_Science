import logging
import os

def setup_logging(log_file="project.log", log_level=logging.INFO):
    """
    Настройка логирования с поддержкой вывода в консоль и файл.

    Parameters:
    - log_file (str): Имя файла для записи логов.
    - log_level (int): Уровень логирования (например, logging.INFO, logging.DEBUG).

    Returns:
    - logger (logging.Logger): Объект логгера.
    """
    # Создание логгера
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Очистка старых обработчиков, чтобы избежать дублирующихся записей
    if logger.hasHandlers():
        logger.handlers.clear()

    # Создание обработчика для записи в файл
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Создание обработчика для вывода в консоль (только для уровней INFO и выше)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    console_handler.setLevel(logging.INFO)  # Устанавливаем уровень логирования для консоли
    logger.addHandler(console_handler)

    logger.info("Логгирование настроено. Логи записываются в файл: %s", os.path.abspath(log_file))
    return logger