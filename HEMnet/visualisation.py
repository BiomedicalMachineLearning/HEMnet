# Example command:
# python /scratch/imb/Xiao/HEMnet/HEMnet/visualisation.py -b /scratch/imb/Xiao/HE_test/10x/ -t 14_01_2020/training_results -p 14_01_2020/prediction_results -o 14_01_2020 -i 1957_T
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns


def roc_plot(total_actual, total_predicted, prefix=""):
    fpr, tpr, _ = roc_curve(total_actual, total_predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5), dpi=300)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC curve', fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('{}_ROC_curve.pdf'.format(prefix))
    #plt.show()


def plot_confusion_matrix(total_actual, total_predicted, classes=None, prefix=""):
    total_predicted_int = np.rint(total_predicted)
    cm = confusion_matrix(total_actual, total_predicted_int)
    # Normalise Confusion Matrix by dividing each value by the sum of that row
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Make DataFrame from Confusion Matrix and classes
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    # Display Confusion Matrix
    plt.figure(figsize=(5, 5), dpi=300)
    ax = cm_plot = sns.heatmap(cm_df, vmin=0, vmax=1, annot=True, fmt='.2f', cmap='Blues', square=True,
                          annot_kws={"size": 25})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix', fontsize=20)
    # Display axes labels
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig('{}_confusion_matrix.pdf'.format(prefix))
    return cm_plot


def learning_curve(total_train_accuracy, total_val_accuracy, acc, prefix=""):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(total_train_accuracy, ls='-', color='blue', label="train_accuracy: %0.2f" % total_train_accuracy[-1])
    plt.plot(total_val_accuracy, ls='-', color='red', label="val_acc: %0.2f test_acc: %0.2f"
                                                            % (total_val_accuracy[-1], acc))
    plt.title('Learning Curve', fontsize=20)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.savefig('{}_learning_curve.pdf'.format(prefix))
    # plt.show()


def loss_curve(total_val_loss, total_train_loss, prefix=""):
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(total_train_loss, ls='--', color='blue', label="train_loss")
    plt.plot(total_val_loss, ls='--', color='red', label="val_loss")
    plt.title('Loss Curve', fontsize=20)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig('{}_loss_curve.pdf'.format(prefix))
    # plt.show()


def find_max_x_y(filenames):
    x_max = y_max = 0
    for i, file in enumerate(filenames):
        x, y = file.split(".")[0].split("_")[-2:]
        if int(x) > int(x_max):
            x_max = int(x)
        if int(y) > int(y_max):
            y_max = int(y)
    return x_max, y_max


def prediction_heatmap(filenames, total_predicted, prefix=""):
    x_max, y_max = find_max_x_y(filenames)
    zero_data = np.zeros(shape=(x_max + 1, y_max + 1))
    all_neg1_data = zero_data - 1
    df = pd.DataFrame(all_neg1_data, columns=np.arange(y_max + 1), index=np.arange(x_max + 1))
    for i, file in enumerate(filenames):
        x, y = file.split(".")[0].split("_")[-2:]
        df.iloc[int(x), int(y)] = float(total_predicted[i])
    mask = df < 0
    plt.figure(figsize=(15, 15), dpi=300)
    ax = sns.heatmap(df, vmin=0, vmax=1, annot=False, fmt='.2f', cmap='RdYlBu',
                square=True, mask=mask, cbar_kws={"shrink": 0.5})
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # 'RdYlBu'
    plt.title('Predicted probability of all tiles from one WSI', fontsize=20)
    # Display axes labels
    plt.ylabel('Tile position Y', fontsize=20)
    plt.xlabel('Tile position X', fontsize=20)
    plt.tight_layout()
    plt.savefig('{}_prediction_heatmap.pdf'.format(prefix))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type = Path,
                        help = 'Base Directory')
    parser.add_argument('-t', '--train_results', type = Path,
                        help = 'Directory containing training results - relative to base directory')
    parser.add_argument('-p', '--predict_results', type=Path,
                        help='Directory containing predict results - relative to base directory')
    parser.add_argument('-o', '--out_dir', type = Path,
                        help = 'Output Directory - relative to Base Directory')
    parser.add_argument('-i', '--WSI', type=str, default="All",
                        help='Name of Whole Slide Image for visualisation')
    '''
    parser.add_argument('-w', '--saved_model', type=Path,
                        help='Path to saved_model - relative to Output Directory')
    parser.add_argument('-m', '--cnn_base', type=str, default="xception",
                        help='pre-trained convolutional neural network base')
    parser.add_argument('-g', '--num_gpus', type=int, default=2,
                        help='Number of GPUs for training model')
    '''
    parser.add_argument('-v', '--verbosity', action = 'store_true',
                        help = 'Increase output verbosity')

    args = parser.parse_args()

    ####################
    # Paths and Inputs #
    ####################
    # Paths
    BASE_PATH = args.base_dir
    TRAN_RESULTS_PATH = BASE_PATH.joinpath(args.train_results)
    TEST_RESULTS_PATH = BASE_PATH.joinpath(args.predict_results)
    OUTPUT_PATH = BASE_PATH.joinpath(args.out_dir)
    # SAVED_MODEL = OUTPUT_PATH.joinpath(args.saved_model)

    # User selectable parameters
    # CNN_BASE = args.cnn_base
    # NUM_GPUS = args.num_gpus
    WSI = args.WSI
    VERBOSE = args.verbosity

    # Verbose functions
    if VERBOSE:
        verbose_print = lambda *args: print(*args)
        verbose_save_img = lambda img, path, img_type: img.save(path, img_type)
        verbose_save_fig = lambda fig, path, dpi=300: fig.savefig(path, dpi=dpi)
    else:
        verbose_print = lambda *args: None
        verbose_save_img = lambda *args: None
        verbose_save_fig = lambda *args: None

    total_train_accuracy = np.load(TRAN_RESULTS_PATH / "total_train_accuracy.npy")
    total_train_loss = np.load(TRAN_RESULTS_PATH / "total_train_loss.npy")
    total_val_accuracy = np.load(TRAN_RESULTS_PATH / "total_val_accuracy.npy")
    total_val_loss = np.load(TRAN_RESULTS_PATH / "total_val_loss.npy")
    total_actual = np.load(TEST_RESULTS_PATH / "total_actual.npy")
    total_predicted = np.load(TEST_RESULTS_PATH / "total_predicted.npy")
    filenames = np.load(TEST_RESULTS_PATH / "file_names.npy")

    OUTPUT_PATH = OUTPUT_PATH.joinpath('visualisation')
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    classes = ["cancer", "non-cancer"]
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    class_list = label_encoder.classes_

    roc_plot(total_actual, total_predicted, prefix=OUTPUT_PATH.joinpath(WSI))
    plot_confusion_matrix(total_actual, total_predicted, prefix=OUTPUT_PATH.joinpath(WSI))

    acc = accuracy_score(total_actual, np.rint(total_predicted))
    print("Prediction accuracy is %.2f" % acc)

    learning_curve(total_train_accuracy, total_val_accuracy, acc, prefix=OUTPUT_PATH.joinpath(WSI))
    loss_curve(total_val_loss, total_train_loss, prefix=OUTPUT_PATH.joinpath(WSI))

    if WSI != "All":
        prediction_heatmap(filenames, total_predicted, prefix=OUTPUT_PATH.joinpath(WSI))









