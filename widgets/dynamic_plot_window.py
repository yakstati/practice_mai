# widgets/dynamic_plot_window.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

R_earth = 6378137.0

class DynamicPlotWindow(QWidget):
    def __init__(self, df, pp_xyz):
        super().__init__()
        self.setWindowTitle("Эволюция системы")
        self.dynamic_artists = []

        self.df = df
        self.pp_xyz = pp_xyz

        layout = QVBoxLayout()

        # matplotlib холст
        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)

        # ползунок времени
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(df) - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.update_plot)
        layout.addWidget(self.slider)

        # метка текущего времени
        self.time_label = QLabel("Время: 0 с")
        layout.addWidget(self.time_label)

        # кнопка сохранения
        self.save_button = QPushButton("Сохранить PNG")
        self.save_button.clicked.connect(self.save_plot)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.plot_background()
        self.update_plot(0)

    def plot_background(self):
        """
        Рисует землю и серую траекторию один раз.
        """
        fig = self.canvas.figure
        fig.clf()
        self.ax = fig.add_subplot(111, projection='3d')

        # Земля
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xe = R_earth * np.outer(np.cos(u), np.sin(v))
        ye = R_earth * np.outer(np.sin(u), np.sin(v))
        ze = R_earth * np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(xe, ye, ze, color='lightblue', alpha=0.3)

        # серая траектория
        self.ax.plot(
            self.df['x'],
            self.df['y'],
            self.df['z'],
            color='gray',
            alpha=0.3,
            label='Траектория ИСЗ'
        )

        # приёмный пункт
        pp_x, pp_y, pp_z = self.pp_xyz
        self.ax.scatter(pp_x, pp_y, pp_z, color='blue', s=50, label='Приёмный пункт')

        # выставить лимиты одинаковыми
        max_range = np.array([
            self.df['x'].max() - self.df['x'].min(),
            self.df['y'].max() - self.df['y'].min(),
            self.df['z'].max() - self.df['z'].min()
        ]).max() / 2.0

        mid_x = (self.df['x'].max() + self.df['x'].min()) / 2.0
        mid_y = (self.df['y'].max() + self.df['y'].min()) / 2.0
        mid_z = (self.df['z'].max() + self.df['z'].min()) / 2.0

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax.legend()

    def update_plot(self, index):
        """
        Обновляет график при смене позиции ползунка.
        """
        # удалить все dynamic объекты предыдущего кадра
        for artist in self.dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.dynamic_artists.clear()

        row = self.df.iloc[index]

        sat_pos = np.array([row['x'], row['y'], row['z']])
        pp = np.array(self.pp_xyz)
        visible = row['visible']

        color = 'green' if visible else 'red'

        # точка спутника
        point = self.ax.scatter(
            sat_pos[0], sat_pos[1], sat_pos[2],
            color=color, s=40, label='Положение спутника'
        )
        self.dynamic_artists.append(point)

        # конус видимости
        cone_surf = self.plot_cone(pp, sat_pos, np.deg2rad(5), color=color)
        if cone_surf is not None:
            self.dynamic_artists.append(cone_surf)

        # обновить текст времени
        self.time_label.setText(f"Время: {int(row['time_sec'])} с")

        self.canvas.draw()


    def plot_cone(self, base, tip, angle, color='green'):
        """
        Строит коническую поверхность между base и tip.
        """
        v = tip - base
        h = np.linalg.norm(v)
        if h == 0:
            return None

        v_unit = v / h

        # радиус конуса на верхушке
        radius = h * np.tan(angle)

        # ортогональные вектора для построения круга
        not_v = np.array([1, 0, 0])
        if np.allclose(v_unit, not_v):
            not_v = np.array([0, 1, 0])

        n1 = np.cross(v_unit, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v_unit, n1)

        # сетка круга
        theta = np.linspace(0, 2 * np.pi, 50)
        circle_points = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta)
        ])

        # вращение в плоскость конуса
        R = np.column_stack((n1, n2, v_unit))

        # переместить circle к tip
        cone_surface = []
        X = []
        Y = []
        Z = []

        for i in np.linspace(0, h, 30):
            scale = (i / h)
            scaled_circle = circle_points * scale
            rotated = R @ scaled_circle
            layer = base.reshape(3, 1) + rotated + v_unit.reshape(3,1)*i
            X.append(layer[0])
            Y.append(layer[1])
            Z.append(layer[2])

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        # отрисовать поверхность
        surf = self.ax.plot_surface(
            X, Y, Z, color=color, alpha=0.2, linewidth=0
        )
        return surf

    def save_plot(self):
        self.canvas.figure.savefig("dynamic_trajectory.png")
