from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

R_earth = 6378137.0

class PlotWindow(QWidget):
    def __init__(self, df, pp_xyz):
        super().__init__()
        self.setWindowTitle("Траектория ИСЗ")

        layout = QVBoxLayout()

        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)

        self.save_button = QPushButton("Сохранить график в PNG")
        self.save_button.clicked.connect(self.save_plot)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.df = df
        self.pp_xyz = pp_xyz
        self.plot()

    def plot(self):
        fig = self.canvas.figure
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.df['x'], self.df['y'], self.df['z'], color='gray', alpha=0.5, label='Траектория ИСЗ')

        visible_df = self.df[self.df['visible']]
        ax.scatter(visible_df['x'], visible_df['y'], visible_df['z'], color='red', s=10, label='Видимость')

        pp_x, pp_y, pp_z = self.pp_xyz
        ax.scatter(pp_x, pp_y, pp_z, color='blue', s=50, label='Приёмный пункт')

        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xe = R_earth * np.outer(np.cos(u), np.sin(v))
        ye = R_earth * np.outer(np.sin(u), np.sin(v))
        ze = R_earth * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xe, ye, ze, color='lightblue', alpha=0.3)

        ax.legend()
        ax.set_title("3D траектория ИСЗ")
        self.canvas.draw()

    def save_plot(self):
        self.canvas.figure.savefig("trajectory.png")
