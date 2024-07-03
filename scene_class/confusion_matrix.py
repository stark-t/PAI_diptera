from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


# Confusion Matrix Function:
def cm_analysis(
    y_true,
    y_pred,
    labels,
    ymap=None,
    figsize=(7, 7),
    filename=None,
    filename_tex=None,
    plot=None,
    fontsize=12,
):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    # fix confusion matrix for plot
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum) * 100
    # cm_perc = np.divide(cm, (cm_sum * 100.0 + 1e-5))
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/ \n   %d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)

    cm_pd = pd.DataFrame(cm_perc, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_pd,
        annot=annot,
        fmt="",
        ax=ax,
        cmap="Blues",
        square=True,
        cbar=False,
        annot_kws={"size": fontsize},
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize, rotation=360)
    ax.set_xlabel("\nPredicted", fontsize=fontsize)
    ax.set_ylabel("Actual\n", fontsize=fontsize)
    ax.figure.tight_layout()
    # ax.figure.subplots_adjust(bottom=0.2)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    if plot is not None:
        plt.show()
        plt.close()
        # plt.pause(1)  # 3 seconds, I use 1 usually
        # plt.close("all")
    else:
        plt.close()

    if filename_tex is not None :
        with open(filename_tex, 'w') as f:
            for i in range(len(labels)):
                for j in range(len(labels)):
                    f.write("{:6.1f}".format(cm[i, j]))
                    if j != len(labels) - 1:
                        f.write(", ")
                    else:
                        f.write("\n")
            if i != len(labels) - 1:
                f.write(", ")

    return cm_pd
