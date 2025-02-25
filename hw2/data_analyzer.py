#This is data_analyzer.py
import pandas as pd
class DataFrameHandler:  #Создаем класс DataFrameHandler
    def __init__(self, df):
        self.df = df
        
    def count_missing_values(self):  #Подсчет пустых или пропущенных значений в каждом столбце.
        return self.df.isnull().sum()
       
    def report_missing_values(self):  #Вывод отчета с информацией о пропущенных значениях.
        missing_values = self.count_missing_values()
        total = self.df.shape[0]
        report = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': (missing_values / total) * 100
        })
        return report[report['Missing Values'] > 0]

    def fill_missing_values(self, method='mean'):  #Заполнение пропущенных значений: средним, медианой или наиболее частым значением.
        if method == 'mean':
            self.df.fillna(self.df.mean(numeric_only=True), inplace=True)
        elif method == 'median':
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        elif method == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        else:
            raise ValueError("Метод заполнения должен быть 'mean', 'median' или 'mode'.")