import matplotlib.pyplot as plt
import numpy as np

class DataVisualzation:

    def __init__(self, df) -> None:
        self.df = df.copy()

    def _plot_graph(self):
        plt.bar(self.df['shipping'], self.df['price'])
        plt.show()
