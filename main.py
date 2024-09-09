if True:
    from reset_random import reset_random

    reset_random()
import os
import sys

import pandas as pd
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFont, QTextCursor, QTextOption
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QGridLayout,
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QLabel,
    QPlainTextEdit,
    QFrame,
    QTableView,
    QAbstractItemView,
    QComboBox,
    QScrollArea,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

import train
from data_handler import load_data, preprocess_car_hacking, preprocess_unsw_nb15
from utils import Stream, Worker, PandasDfToPyqtTable


class MainGUI(QWidget):
    def __init__(self):
        super(MainGUI, self).__init__()

        self.setWindowTitle("Vehicular Communication Intrusion Detection System")
        self.width_p = QApplication.desktop().availableGeometry().width() // 100
        self.height_p = QApplication.desktop().availableGeometry().height() // 100

        app.setFont(QFont("Roboto Mono"))
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        self.main_layout = QGridLayout()
        self.main_layout.setAlignment(Qt.AlignTop)

        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignTop)
        self.right_layout = QVBoxLayout()
        self.right_layout.setAlignment(Qt.AlignTop)

        self.gb_1 = QGroupBox("Input Data")
        self.gb_1.setFixedHeight(self.height_p * 15)
        self.gb_1.setFixedWidth(self.width_p * 49)
        self.grid_1 = QGridLayout()
        self.grid_1.setContentsMargins(0, 0, 0, 0)
        self.gb_1.setLayout(self.grid_1)

        self.data_combo = QComboBox()
        self.data_combo.addItems(["<- Choose Dataset ->", "UNSW_NB15", "CAR_HACKING"])
        self.data_combo.setEditable(True)
        le = self.data_combo.lineEdit()
        le.setAlignment(Qt.AlignCenter)
        le.setReadOnly(True)
        self.data_combo.setLineEdit(le)
        self.grid_1.addWidget(self.data_combo, 0, 0)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_thread)
        self.grid_1.addWidget(self.load_btn, 0, 1)

        self.pp_btn = QPushButton("PreProcess")
        self.pp_btn.clicked.connect(self.pp_thread)
        self.grid_1.addWidget(self.pp_btn, 0, 2)

        self.train_btn = QPushButton(
            "Graph Convolutional Gated Recurrent Neural Network"
        )
        self.train_btn.clicked.connect(self.train_thread)
        self.grid_1.addWidget(self.train_btn, 1, 0, 1, 2)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)
        self.grid_1.addWidget(self.reset_btn, 1, 2)

        self.gb_2 = QGroupBox("Data Table")
        self.gb_2.setFixedHeight(self.height_p * 45)
        self.gb_2.setFixedWidth(self.width_p * 49)
        self.grid_2_scroll = QScrollArea()
        self.grid_2_scroll.setFrameShape(False)
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.grid_2_widget.hide()
        self.grid_2 = QGridLayout(self.grid_2_widget)
        self.gb_2.setLayout(self.gb_2_v_box)
        self.grid_2.setSpacing(20)
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)

        self.gb_3 = QGroupBox("Progress")
        self.gb_3.setFixedHeight(self.height_p * 37)
        self.gb_3.setFixedWidth(self.width_p * 49)
        self.grid_3 = QGridLayout()
        self.gb_3.setLayout(self.grid_3)

        self.progress_pte = QPlainTextEdit()
        self.progress_pte.setStyleSheet("background-color: transparent;")
        self.progress_pte.setFrameShape(QFrame.NoFrame)
        self.progress_pte.setReadOnly(True)
        self.progress_pte.setWordWrapMode(QTextOption.WordWrap)
        self.grid_3.addWidget(self.progress_pte, 0, 0)

        self.gb_4 = QGroupBox("Visualization")
        self.gb_4.setFixedHeight(self.height_p * 99)
        self.gb_4.setFixedWidth(self.width_p * 50)
        self.grid_4_scroll = QScrollArea()
        self.grid_4_scroll.setFrameShape(QFrame.NoFrame)
        self.gb_4_v_box = QVBoxLayout()
        self.grid_4_widget = QWidget()
        self.grid_4 = QGridLayout(self.grid_4_widget)
        self.grid_4.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.grid_4_scroll.setWidgetResizable(True)
        self.grid_4_scroll.setWidget(self.grid_4_widget)
        self.gb_4_v_box.addWidget(self.grid_4_scroll)
        self.gb_4_v_box.setContentsMargins(0, 0, 0, 0)
        self.gb_4.setLayout(self.gb_4_v_box)

        self.main_layout.addLayout(self.left_layout, 0, 0)
        self.main_layout.addLayout(self.right_layout, 0, 1)

        self.left_layout.addWidget(self.gb_1, Qt.AlignTop)
        self.left_layout.addWidget(self.gb_2, Qt.AlignTop)
        self.left_layout.addWidget(self.gb_3, Qt.AlignTop)
        self.right_layout.addWidget(self.gb_4, Qt.AlignTop)

        self.thread_pool = QThreadPool()

        sys.stdout = Stream(fn=self.update_progress)

        self.original_df = pd.DataFrame()
        self.preprocess_df = pd.DataFrame()
        self.x = []
        self.y = []
        self.thread_pool = QThreadPool()
        self.index = 0

        self.reset()
        self.setLayout(self.main_layout)
        self.showMaximized()

    def update_progress(self, text):
        cursor = self.progress_pte.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.progress_pte.setTextCursor(cursor)
        self.progress_pte.ensureCursorVisible()

    def add_table(self, df):
        table_view = QTableView(self)
        model = PandasDfToPyqtTable(df)
        table_view.setFixedWidth((self.gb_2.width() // 100) * 95)
        table_view.setFixedHeight((self.gb_2.height() // 100) * 95)
        table_view.setModel(model)
        table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_view.verticalHeader().hide()
        table_view.resizeColumnsToContents()
        self.grid_2.addWidget(table_view, self.grid_2.count() + 1, 0, Qt.AlignHCenter)

    def add_plot(self, fig, title):
        canvas = FigureCanvasQTAgg(figure=fig)
        canvas.setFixedSize(640, 480)
        self.grid_4.addWidget(QLabel(title), self.index, 0)
        self.grid_4.addWidget(canvas, self.index + 1, 0)
        self.index += 2

    def load_thread(self):
        if self.data_combo.currentIndex() != 0:
            self.reset()
            worker = Worker(self.load_runner)
            worker.signals.finished.connect(self.load_finisher)
            self.thread_pool.start(worker)
            self.load_btn.setEnabled(False)
            self.data_combo.setEnabled(False)
        else:
            self.show_message_box("Data Error", "Choose Dataset!!!")

    def load_runner(self):
        self.original_df = load_data(self.data_combo.currentText())

    def load_finisher(self):
        self.add_table(self.original_df.head(100))
        self.load_btn.setEnabled(False)
        self.pp_btn.setEnabled(True)

    def pp_thread(self):
        worker = Worker(self.pp_runner)
        worker.signals.finished.connect(self.pp_finisher)
        self.thread_pool.start(worker)
        self.pp_btn.setEnabled(False)

    def pp_runner(self):
        pf = (
            preprocess_unsw_nb15
            if self.data_combo.currentText() == "UNSW_NB15"
            else preprocess_car_hacking
        )
        self.preprocess_df, self.x, self.y = pf(self.original_df.copy(deep=True))

    def pp_finisher(self):
        self.add_table(self.preprocess_df.head(100))
        self.train_btn.setEnabled(True)

    def train_thread(self):
        reset_random()
        self.add_plot(train.ACC_PLOT, "Accuracy Plot")
        self.add_plot(train.LOSS_PLOT, "Loss Plot")
        worker = Worker(self.train_runner)
        worker.signals.finished.connect(self.train_finisher)
        self.thread_pool.start(worker)
        self.train_btn.setEnabled(False)
        self.pp_btn.setEnabled(False)

    def train_runner(self):
        train.train(
            self.preprocess_df.copy(deep=True),
            self.x,
            self.y,
            self.data_combo.currentText(),
        )

    def train_finisher(self):
        for t in train.RESULTS_PLOT:
            for v in train.RESULTS_PLOT[t]:
                self.add_plot(train.RESULTS_PLOT[t][v], v)

    @staticmethod
    def clear_layout(layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                continue
            w = item.widget()
            if w:
                w.deleteLater()

    @staticmethod
    def show_message_box(title, msg):
        msg_box = QMessageBox()
        msg_box.setFont(QFont("Roboto Code", 10, 1))
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.exec_()

    def disable(self):
        self.data_combo.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.pp_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.original_df = pd.DataFrame()
        self.preprocess_df = pd.DataFrame()
        self.x = []
        self.y = []
        self.progress_pte.clear()

    def reset(self):
        self.disable()
        self.clear_layout(self.grid_2)
        self.clear_layout(self.grid_4)


if __name__ == "__main__":
    app = QApplication([sys.argv])
    app.setStyle("Fusion")
    window = MainGUI()
    sys.exit(app.exec_())
