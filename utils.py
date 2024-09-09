import os
import prettytable
import os
import sys
import traceback

import pandas as pd
import prettytable
from PyQt5.QtCore import (
    QAbstractTableModel,
    Qt,
    pyqtSlot,
    QObject,
    pyqtSignal,
    QRunnable,
)
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from performance_evaluator.metrics import evaluate
from performance_evaluator.plots import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


class PandasDfToPyqtTable(QAbstractTableModel):
    def __init__(self, df):
        QAbstractTableModel.__init__(self)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role=None):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._df.columns[col]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return col
        return None


class Stream(QObject):
    fn = pyqtSignal(str)

    def write(self, text):
        self.fn.emit(str(text))

    def flush(self):
        pass


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(e)
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def print_df_to_table(df):
    field_names = list(df.columns)
    p_table = prettytable.PrettyTable(field_names=field_names)
    p_table.add_rows(df.values.tolist())
    print(
        "\n".join(
            [
                "\t\t{0}".format(p_)
                for p_ in p_table.get_string().splitlines(keepends=False)
            ]
        )
    )


class TrainingCallback(Callback):
    def __init__(self, acc_loss_path, plt1, plt2):
        self.acc_loss_path = acc_loss_path
        self.plt1 = plt1
        self.plt2 = plt2
        if os.path.isfile(self.acc_loss_path):
            self.df = pd.read_csv(self.acc_loss_path)
        else:
            self.df = pd.DataFrame(
                [], columns=["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
            )
            self.df.to_csv(self.acc_loss_path, index=False)
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs=None):
        self.df.loc[len(self.df.index)] = [
            int(epoch + 1),
            round(logs["accuracy"], 4),
            round(logs["val_accuracy"], 4),
            round(logs["loss"], 4),
            round(logs["val_loss"], 4),
        ]
        self.df.to_csv(self.acc_loss_path, index=False)
        print(
            "[EPOCH :: {0}] -> Acc :: {1} | Val_Acc :: {2} | Loss :: {3} | Val_Loss :: {4}".format(
                epoch + 1, *[format(v, ".4f") for v in self.df.values[-1][1:]]
            )
        )
        plot_acc_loss(
            self.df, self.plt1, self.plt2, os.path.dirname(self.acc_loss_path)
        )


def plot_line(plt_, y1, y2, epochs, for_, save_path):
    ax = plt_.gca()
    ax.clear()
    ax.plot(range(epochs), y1, label="Training", color="dodgerblue")
    ax.plot(range(epochs), y2, label="Validation", color="orange")
    ax.set_title("Training and Validation {0}".format(for_))
    ax.set_xlabel("Epochs")
    ax.set_ylabel(for_)
    ax.set_xlim([0, epochs])
    ax.legend()
    plt_.tight_layout()
    plt_.savefig(save_path)


def plot_acc_loss(df, plt1, plt2, save_dir):
    epochs = len(df)
    acc = df["accuracy"].values
    val_acc = df["val_accuracy"].values
    loss = df["loss"].values
    val_loss = df["val_loss"].values
    plot_line(
        plt1, acc, val_acc, epochs, "Accuracy", os.path.join(save_dir, "accuracy.png")
    )
    plot_line(plt2, loss, val_loss, epochs, "Loss", os.path.join(save_dir, "loss.png"))


def plot(y, pred, prob, plts, results_dir, classes):
    for_ = os.path.basename(results_dir)
    print("[INFO] Evaluating {0} Data".format(for_))
    os.makedirs(results_dir, exist_ok=True)

    m = evaluate(y, pred, prob, classes)
    df = m.overall_metrics
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print_df_to_table(df)

    fig = plts[for_]["CONF_MAT"]
    ax = fig.gca()
    ax.clear()
    confusion_matrix(
        y, pred, classes, ax=ax, xticklabels_rotation=0, yticklabels_rotation=0
    )
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "conf_mat.png"))

    fig = plts[for_]["PR_CURVE"]
    ax = fig.gca()
    ax.clear()
    precision_recall_curve(y, prob, classes, ax=ax, legend_ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pr_curve.png"))

    fig = plts[for_]["ROC_CURVE"]
    ax = fig.gca()
    ax.clear()
    roc_curve(y, prob, classes, ax=ax, legend_ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "roc_curve.png"))
