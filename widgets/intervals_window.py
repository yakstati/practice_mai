from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton

class IntervalsWindow(QWidget):
    def __init__(self, intervals_dt):
        super().__init__()
        self.setWindowTitle("Интервалы видимости")

        layout = QVBoxLayout()

        self.table = QTableWidget(len(intervals_dt), 2)
        self.table.setHorizontalHeaderLabels(["Начало", "Конец"])

        for i, (start, end) in enumerate(intervals_dt):
            self.table.setItem(i, 0, QTableWidgetItem(str(start)))
            self.table.setItem(i, 1, QTableWidgetItem(str(end)))

        layout.addWidget(self.table)

        self.save_button = QPushButton("Сохранить в CSV")
        self.save_button.clicked.connect(self.save_to_csv)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_to_csv(self):
        import pandas as pd
        rows = []
        for row in range(self.table.rowCount()):
            start = self.table.item(row, 0).text()
            end = self.table.item(row, 1).text()
            rows.append({"start": start, "end": end})
        df = pd.DataFrame(rows)
        df.to_csv("visibility_intervals.csv", index=False)
