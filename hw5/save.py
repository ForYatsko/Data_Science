import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def save_to_csv(data, file_path):
    """Сохранение данных в CSV."""
    logger.info(f"Сохранение данных в файл: {file_path}")
    pd.DataFrame(data).to_csv(file_path, index=False)
    logger.info("Данные успешно сохранены.")

def save_plot(plt, file_path):
    """Сохранение графика."""
    logger.info(f"Сохранение графика в файл: {file_path}")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    logger.info("График успешно сохранён.")