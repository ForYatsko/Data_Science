import logging

def setup_logging(log_file='app.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Перезаписывать файл лога при каждом запуске
    )
    logging.info("Logging is set up.")