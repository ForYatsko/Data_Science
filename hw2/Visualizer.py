#This is Visualizer.py

import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, df):
        self.df = df
        self.plots = []

    def add_histogram(self, column, bins=20):  #Визуализация гистограммы для указанного столбца.
        plt.figure(figsize=(7, 6))
        self.df[column].hist(bins=bins, color='skyblue', edgecolor='black')
        plt.title(f'Гистограмма: {column}')
        plt.xlabel(column)
        plt.ylabel('Частота')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    def add_line_plot(self, x_column, y_column):  #Добавление линейного графика.
        plt.figure()
        plt.plot(self.df[x_column], self.df[y_column], marker='o')
        plt.title(f'Линейный график: {y_column} по {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.show()
        self.plots.append(f'Линейный график: {y_column} по {x_column}')

    def add_scatter_plot(self, x_column, y_column):  #Добавление диаграммы рассеяния.
        plt.figure()
        plt.scatter(self.df[x_column], self.df[y_column])
        plt.title(f'Диаграмма рассеяния: {y_column} по {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.show()
        self.plots.append(f'Диаграмма рассеяния: {y_column} по {x_column}')