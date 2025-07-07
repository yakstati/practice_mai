from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from widgets.plot_window import PlotWindow
from widgets.intervals_window import IntervalsWindow
from satellite_model import calculate_satellite_trajectory
from db import DBHandler

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Tracker")
        self.init_ui()
        self.db = DBHandler()

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Широта приёмного пункта (°):"))
        self.lat_edit = QLineEdit("55.812843")
        layout.addWidget(self.lat_edit)

        layout.addWidget(QLabel("Долгота приёмного пункта (°):"))
        self.lon_edit = QLineEdit("37.494503")
        layout.addWidget(self.lon_edit)

        self.calc_button = QPushButton("Рассчитать траекторию")
        self.calc_button.clicked.connect(self.on_calculate)
        layout.addWidget(self.calc_button)

        self.setLayout(layout)

    def on_calculate(self):
        try:
            lat = float(self.lat_edit.text())
            lon = float(self.lon_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректные числа.")
            return

        # расчет орбиты
        df, intervals_dt, pp_xyz = calculate_satellite_trajectory(lat, lon)

        # запись в БД
        self.db.save_to_db(df, intervals_dt, lat, lon, pp_xyz)

        # открываем окна
        self.plot_window = PlotWindow(df, pp_xyz)
        self.plot_window.show()

        self.intervals_window = IntervalsWindow(intervals_dt)
        self.intervals_window.show()
