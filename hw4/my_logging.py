import logging
import os

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Настраивает логгирование для программы.

    :param log_file: Путь к файлу, куда будут записываться логи (опционально).
    :param level: Уровень логгирования (по умолчанию logging.INFO).
    :return: Настроенный объект логгера.
    """
    # Формат логов
    log_format = "%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s"
    
    # Настройка логгирования
    handlers = [logging.StreamHandler()]  # Вывод в консоль
    if log_file:
        # Если указан путь к файлу, добавляем запись в файл
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
    )
    
    logger = logging.getLogger()
    logger.info("Логгирование настроено успешно.")
    if log_file:
        logger.info(f"Логи записываются в файл: {os.path.abspath(log_file)}")
    return logger