#This is main.py
from data_loader import DataLoader
from data_analyzer import DataFrameHandler
from Visualizer import Visualizer
import pandas as pd

if __name__ == "__main__":
    loader = DataLoader()
    
    # Загрузка из CSV и первый просмотр данных
    csv_data = loader.load_csv('2015.csv')
display(csv_data)

if __name__ == "__main__":
    df = pd.DataFrame(csv_data)
    handler = DataFrameHandler(df)
  
    print("Пропущенные значения:\n", handler.count_missing_values())   # Подсчет пропущенных значений
    
    print("\nОтчет о пропущенных значениях:\n", handler.report_missing_values())  # Отчет о пропущенных значениях

    handler.fill_missing_values(method='mean')  #Заполнение пропущенных значений: средним, медианой или наиболее частым значением
    print("\nDataFrame после заполнения пропущенных значений:\n", handler.df)

if __name__ == "__main__":
    visualizer = Visualizer(df)

    # Добавление визуализаций
    visualizer.add_histogram('Happiness Score', bins=20)
    visualizer.add_line_plot('Happiness Rank', 'Economy (GDP per Capita)')
    visualizer.add_scatter_plot('Happiness Rank', 'Economy (GDP per Capita)')