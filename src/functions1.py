"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Function set 1.
"""

################### Function set 1 ###################
##### When importing packages, avoid circular imports. #####
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from umap import UMAP


def print_loss(history, path, prefix="", saveLoss=False, showInfo=True):
    """
    Output loss information.

    :param history: class LossHistory() object;
    :param path: Directory for saving losses;
    :param prefix: The prefix of the output file;
    :param saveLoss: Whether to save to file, default not;
    :param showInfo: Do you want to output detailed information.
    """
    if showInfo:
        print("training loss:")
        print("  The loss of each eopch:", np.array(history.loss)[:10])
        print("  The reconstruction loss of each eopch:", np.array(history.reconLoss)[:10])
        print("  The KL loss for each eopch:", np.array(history.KLLoss)[:10])
    if saveLoss:
        with open(path + prefix + "loss.txt", 'w') as file:
            for item in history.loss: file.write(f"{item.numpy():.4f} ")
        with open(path + prefix + "recon_loss.txt", 'w') as file:
            for item in history.reconLoss: file.write(f"{item.numpy():.4f} ")
        with open(path + prefix + "kl_loss.txt", 'w') as file:
            for item in history.KLLoss: file.write(f"{item.numpy():.4f} ")

def plot_loss(history, path, prefix="", saveImg=False):
    """
    Draw line charts of all losses.

    :param history: class LossHistory() object;
    :param path: Directory for saving losses;
    :param prefix: The prefix of the output file;
    :param saveImg: Whether to save to file, default not.
    """
    plotLossImg(history.loss, path + prefix +"loss.png", saveImg, title="loss")
    plotLossImg(history.reconLoss, path + prefix + "recon_loss.png", saveImg, title="reconstruction loss")
    plotLossImg(history.KLLoss, path + prefix + "kl_loss.png", saveImg, title="KL loss")
    
def plotLossImg(loss, pathImg, saveImg, title = "epoch loss", xlab = "epoch", ylab = "loss"):
    """
    Draw a line chart of losses.

    :param loss: Loss List;
    :param pathImg: The save path of the file (including the file name);
    :param saveImg: Whether to save to file;
    :param title: image title;
    :param xlab: X-axis text;
    :param ylab: Y-axis text;
    """
    lossImg = plt.figure()
    # Draw a line chart
    plt.plot(loss)
    # Add tags and titles
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    # display graphics
    # plt.show()
    if saveImg:
        lossImg.savefig(pathImg)
        lossImg.savefig(pathImg.rstrip("png") + "pdf")
   


def my_kmeans(z, cluster, seed=42):
    """
    KMeans clustering.

    :param z: Z variable;
    :param cluster: Number of cell cluster;
    :param seed: random seed;
    :return group_pred: predict results;
    :return centers: cluster center.
    """
    kmeans = KMeans(n_clusters=cluster, random_state=seed, n_init='auto')
    kmeans.fit(z)
    group_pred = kmeans.labels_ + 1
    return group_pred, kmeans.cluster_centers_

def get_group_pred(z, cluster, path="", img_type="tSNE", saveImg=False, showInfo=True, plot_center=True):
    """
    Obtain the results of clustering prediction.

    :param z: Z variable;
    :param path: The save path of the file (including file name);
    :param cluster: Number of cell cluster;
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param saveImg: Whether to save to file;
    :param showInfo: Do you want to output detailed information;
    :param plot_center: Do you want to draw cluster centers;
    :return group_pred: predict results;
    :return centers: cluster center;
    :return result_data: Cell coordinates after tSNE/UMAP dimensionality reduction.
    """
    result_data = None
    if img_type == "tSNE":
        tsne = TSNE(n_components=2, learning_rate=200, perplexity=30, random_state=42)
        tsne_data = tsne.fit_transform(z)
        result_data = tsne_data
        group_pred, centers = my_kmeans(tsne_data, cluster)
        # Draw a clustering diagram
        plot_cluster_img(group_pred, tsne_data, centers, path+"tSNE_pred.png", img_type, saveImg, showInfo, plot_center)
    else:   # umap
        umap = UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
        umap_data = umap.fit_transform(z)
        result_data = umap_data
        group_pred, centers = my_kmeans(umap_data, cluster)
        # Draw a clustering diagram
        plot_cluster_img(group_pred, umap_data, centers, path+"UMAP_pred.png", img_type, saveImg, showInfo, plot_center)
    return group_pred, centers, result_data

def plot_cluster_img(group_label, dim_data, centers, path="", img_type="tSNE", saveImg=False, showInfo=True, plot_center=True):
    """
    Draw a clustering diagram with two-dimensional data obtained from tSNE/UMAP.

    :param group_label: The cluster labels of each sample (predicted results);
    :param dim_data: Reduced dimensional data;
    :param centers: cluster center;
    :param path: The save path of the file (including file name);
    :param img_type: The function for dimensionality reduction clustering, default tSNE, can also be UMAP;
    :param saveImg: Whether to save to file;
    :param showInfo: Do you want to output detailed information;
    :param plot_center: Do you want to draw cluster centers.
    """
    img = None
    if showInfo | True:
        img, ax = plt.subplots()
        # Cancel the upper and right border lines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel(img_type + "_1")
        plt.ylabel(img_type + "_2")
        plt.title(img_type)

        cols = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628","#F781BF"]
        # Add legend (at this time, each category needs to be drawn separately)
        cluster = len(np.unique(group_label))
        for c in range(1, cluster+1):
            temp_data = dim_data[group_label==c,:]
            plt.scatter(temp_data[:, 0], temp_data[:, 1], c=cols[c], label="cluster"+str(c), s=10)
        plt.legend(title="Cluster", loc="center right", bbox_to_anchor=(1.2, 0.5), frameon=False)
        if plot_center:
            plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100)
        # plt.show()
    if saveImg:
        img.savefig(path, bbox_inches='tight')
        img.savefig(path.rstrip("png") + "pdf", bbox_inches='tight')

def modify_map(group_pred, group_pred_plus, result_data, cluster):
    """
    Modify the cell coordinates based on the new prediction results.

    :param group_pred: The cluster labels of each sample (predicted results1);
    :param group_pred_plus: The cluster labels of each sample (predicted results2);
    :param result_data: Cell coordinates after tSNE/UMAP dimensionality reduction;
    :param cluster: Number of cell cluster;
    :return new_result_data: Cell coordinates after tSNE/UMAP dimensionality reduction.
    """
    centers = {}
    for i in range(1, cluster+1):
        # cluster center
        center = np.mean(result_data[(group_pred_plus==i) & (group_pred==group_pred_plus)], axis=0)
        centers[i] = [center[0], center[1]]
    
    # Modify the cell coordinates based on the new prediction results
    new_result_data = np.zeros_like(result_data)
    for i in range(len(group_pred)):
        if group_pred[i] != group_pred_plus[i]:
            # if the predicted results are different, modify the coordinates
            new_result_data[i, 0] = result_data[i, 0] - centers[group_pred[i]][0] + centers[group_pred_plus[i]][0]
            new_result_data[i, 1] = result_data[i, 1] - centers[group_pred[i]][1] + centers[group_pred_plus[i]][1]
        else:
            # if the predicted results are the same, keep the original coordinates
            new_result_data[i, 0] = result_data[i, 0]
            new_result_data[i, 1] = result_data[i, 1]
    return new_result_data



def class2class(group_pred, group_true, cluster, showInfo=True):
    """
    Realize the mapping between predicted results.

    :param group_pred: The cluster labels of each sample (predicted results1);
    :param group_true: The cluster labels of each sample (predicted results2);
    :param cluster: Number of cell cluster;
    :param showInfo: Do you want to output detailed information;
    :return new_group_pred: The result of modifying the mapping;
    :return comb: mapped vector.
    """
    # Obtain the highest accuracy for relationships
    _, comb = max_correspond(group_pred, group_true, cluster, showInfo)
    return conv_code(group_pred, comb, cluster), comb

def conv_code(group_label, comb, cluster):
    """
    Map the input group_label.

    :param group_label: The cluster labels of each sample (predicted results);
    :param comb: mapped vector, like [2,3,1];
    :param cluster: Number of cell cluster;
    :return new_group_pred: The result of modifying the mapping.
    """
    new_group_label = np.zeros(len(group_label), dtype=np.int_)
    for i in range(cluster):
        new_group_label[group_label == (i+1)] = comb[i]
    return new_group_label

def max_correspond(group_pred, group_true, cluster, showInfo=True):
    """
    Input two sequences and return the maximum accuracy of the corresponding values.
    Obtain the correspondence with the highest accuracy, such as [1,2,3] ->[1,3,2].

    :param group_pred: The cluster labels of each sample (predicted results1);
    :param group_true: The cluster labels of each sample (predicted results2);
    :param cluster: Number of cell cluster;
    :param showInfo: Do you want to output detailed information;
    :return accs: The result of modifying the mapping;
    :return comb: mapped vector.
    """
    # like: [1,2,3,4]
    order = [i for i in range(1,cluster+1)]
    # like: np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
    combs = np.array(list(itertools.permutations(order)))
    # Store the accuracy of each arrangement, such as 6 out of 3 categories.
    accs = np.zeros(len(combs))
    # Calculate accuracy
    for i in range(len(combs)):
        temp = conv_code(group_pred, combs[i], cluster)
        accs[i] = np.sum(temp == group_true)/len(temp)
    if showInfo:
          print("max correspond:")
          print("  accuracy:", accs)
    return np.max(accs), combs[np.argmax(accs)]

def cluster_correspond(group_true, group_pred, cluster):
    """
    Obtain the corresponding vector matrix for each class (i.e. 1 if it belongs to that class, 0 if it does not belong, matrix size is (cluster, cell)).

    :param group_true: The cluster labels of each sample (predicted results1);
    :param group_pred: The cluster labels of each sample (predicted results2);
    :param cluster: Number of cell cluster;
    """
    # Record the corresponding position of the original group_true
    cell = np.zeros((cluster, len(group_true)), dtype=np.int_)
    # Record the corresponding position of the original group_pred
    cell_pred = np.zeros((cluster, len(group_true)), dtype=np.int_)
    for i in range(len(group_true)):
        cell[group_true[i]-1,i] = 1
        cell_pred[group_pred[i]-1,i] = 1
    return cell, cell_pred



def save_pred(barcode, group_pred, path):
    """
    Save prediction results.

    :param barcode: Barcode List of Cells;
    :param group_pred: The cluster labels of each sample (predicted results);
    :param path: The save path of the file (including file name).
    """
    with open(path, "w") as pred:
        pred.write("barcode\tprediction\n")
        for i in range(len(barcode)):
            temp = str(barcode[i]) + "\t" + str(group_pred[i]) + "\n"
            pred.write(temp)

def save_pred_plus(barcode, group_pred, group_pred_plus, path):
    """
    Save prediction results.

    :param barcode: Barcode List of Cells;
    :param group_pred: The cluster labels of each sample (predicted results1);
    :param group_pred_plus: The cluster labels of each sample (predicted results2);
    :param path: The save path of the file (including file name).
    """
    with open(path, "w") as pred:
        pred.write("barcode\tprediction\tprediction\n")
        for i in range(len(barcode)):
            temp = "{}\t{}\t{}\n".format(barcode[i], group_pred[i], group_pred_plus[i])
            pred.write(temp)

