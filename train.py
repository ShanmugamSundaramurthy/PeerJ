if True:
    from reset_random import reset_random

    reset_random()
import os
import shutil

from matplotlib import pyplot as plt
from data_handler import UNSW_NB15_CLASSES, CAR_HACKING_CLASSES
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from spektral.layers import GCNConv
from utils import TrainingCallback, plot
from models import buildGCGRN

plt.rcParams["font.family"] = "Roboto Mono"

ACC_PLOT = plt.figure(num=1)
LOSS_PLOT = plt.figure(num=2)
RESULTS_PLOT = {
    "Train": {
        "CONF_MAT": plt.figure(num=3),
        "PR_CURVE": plt.figure(num=4),
        "ROC_CURVE": plt.figure(num=5),
    },
    "Test": {
        "CONF_MAT": plt.figure(num=6),
        "PR_CURVE": plt.figure(num=7),
        "ROC_CURVE": plt.figure(num=8),
    },
}


def train(df, x, y, name):
    reset_random()
    CLASSES = ["Normal", "Attack"] if name == "UNSW_NB15" else CAR_HACKING_CLASSES

    similarity_matrix = cosine_similarity(df.values[:, :-1].T)
    adjacency_matrix = (similarity_matrix > 0.8).astype(int)
    adjacency_matrix = GCNConv.preprocess(adjacency_matrix)

    print("[INFO] Splitting Train|Test Data")
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, shuffle=True, random_state=1
    )
    y_cat = to_categorical(y, len(CLASSES))
    train_y_cat = to_categorical(train_y, len(CLASSES))
    test_y_cat = to_categorical(test_y, len(CLASSES))
    if name == "UNSW_NB15":
        x = np.expand_dims(x, axis=2)
        train_x = np.expand_dims(train_x, axis=2)
        test_x = np.expand_dims(test_x, axis=2)
    print("[INFO] X Shape :: {0}".format(x.shape))
    print("[INFO] Y Shape :: {0}".format(y_cat.shape))
    print("[INFO] Train X Shape :: {0}".format(train_x.shape))
    print("[INFO] Train Y Shape :: {0}".format(train_y_cat.shape))
    print("[INFO] Test X Shape :: {0}".format(test_x.shape))
    print("[INFO] Test Y Shape :: {0}".format(test_y_cat.shape))

    model_dir = os.path.join("models", name)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    acc_loss_csv_path = os.path.join(model_dir, "acc_loss.csv")
    model_path = os.path.join(model_dir, "model.h5")
    training_cb = TrainingCallback(acc_loss_csv_path, ACC_PLOT, LOSS_PLOT)
    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=False,
    )

    model = buildGCGRN(x.shape[1:], adjacency_matrix, len(CLASSES))

    initial_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
        print("[INFO] Loading Pre-Trained Model :: {0}".format(model_path))
        model.load_weights(model_path)
        initial_epoch = len(pd.read_csv(acc_loss_csv_path))

    print("[INFO] Fitting Data")
    model.fit(
        [x, adjacency_matrix],
        y_cat,
        validation_data=([x, adjacency_matrix], y_cat),
        epochs=25,
        batch_size=512 if name == "UNSW_NB15" else 1024,
        initial_epoch=initial_epoch,
        callbacks=[training_cb, checkpoint],
        verbose=0,
    )

    model.load_weights(model_path)

    train_prob = model.predict(train_x)
    train_pred = np.argmax(train_prob, axis=1).ravel().astype(int)
    plot(
        train_y.ravel().astype(int),
        train_pred,
        train_prob,
        RESULTS_PLOT,
        "results/{0}/Train".format(name),
        CLASSES,
    )

    test_prob = model.predict(test_x)
    test_pred = np.argmax(test_prob, axis=1).ravel().astype(int)
    plot(
        test_y.ravel().astype(int),
        test_pred,
        test_prob,
        RESULTS_PLOT,
        "results/{0}/Test".format(name),
        CLASSES,
    )


if __name__ == "__main__":
    from data_handler import load_data, preprocess_car_hacking, preprocess_unsw_nb15

    # train(*preprocess_unsw_nb15(load_data("UNSW_NB15")), "UNSW_NB15")
    train(*preprocess_car_hacking(load_data("CAR_HACKING")), "CAR_HACKING")
