from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import cv2
import sys
import os

def recon(test_img, e_faces, x_mean, n, output_img, MSE):
    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE).flatten().reshape(W*H,1)
    ma_img = img - x_mean
    weight = np.dot(ma_img.T, e_faces)
    recon = x_mean + np.dot(weight[0, 0:n], e_faces[:, 0:n].T).reshape(W*H,1)
    cv2.imwrite(output_img, recon.reshape(W, H))
    if MSE==True:
        mse = ((cv2.imread(test_img, cv2.IMREAD_GRAYSCALE) - recon.reshape(W, H)) ** 2).mean(axis=None)
        #print("mse: {}".format(mse))

def test_tsne(test_path, e_faces, n):
    test_imgs = np.array([])
    labels = []
    for f in glob.glob(os.path.join(test_path, "*.png")):
        pos = os.path.basename(f).find('_')
        labels.append(os.path.basename(f)[:pos])
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        if test_imgs.size == 0:
            test_imgs = img
        else:
            test_imgs = np.vstack((test_imgs, img))
    test_imgs = np.transpose(test_imgs) #(W*H, Nt)
    test_mean = np.mean(test_imgs, axis = 1).reshape(W*H,1)
    test_ma_img = test_imgs - test_mean
    test_weights = np.dot(test_ma_img.T, e_faces[:, 0:n]) #(Nt, n)
    return test_weights, np.array(labels)

def visualize_scatter(data_2d, label_ids, id_to_label_dict, figsize=(15,15)):
    plt.figure(figsize=figsize)
    plt.grid()
    nb_classes = len(np.unique(label_ids))
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.gist_rainbow(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.title("T-SNE DIM=100", fontsize=20)
    plt.savefig('T-SNE dim100.png')

def train(train_path):
    x = np.array([])
    W, H = 0, 0
    for f in glob.glob(os.path.join(train_path, "*.png")):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if W==0 and H==0:
            W = img.shape[0]
            H = img.shape[1]
        img = img.flatten()
        if x.size == 0:
            x = img
        else:
            x = np.vstack((x, img))
    x = np.transpose(x) #(W*H, N)
    x_mean = np.mean(x, axis = 1).reshape(W*H,1)
    ma_data = x - x_mean
    U, S, V = np.linalg.svd(ma_data, full_matrices=False)
    return U, x_mean, W, H

def data_preprocess(data_path):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    if not os.path.exists(train_path) and not os.path.exists(test_path):
        os.makedirs(train_path)
        os.makedirs(test_path)
        for f in glob.glob(os.path.join(data_path, "*.png")):
            if f.endswith("_8.png") or f.endswith("_9.png") or f.endswith("_10.png"):
                shutil.move(f, test_path)
            else:
                shutil.move(f, train_path)
    return train_path, test_path


data_path, TA_img_path, TA_output_path =  sys.argv[1], sys.argv[2], sys.argv[3]
#dataset partition
train_path, test_path = data_preprocess(data_path)

e_faces, x_mean, W, H = train(train_path)

#plot mean face
cv2.imwrite("mean.png", x_mean.reshape(W, H))
#plot top 5 eigenfaces
for i in range(5):
    norm_image = cv2.normalize(e_faces[:,i].reshape(W*H,1), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image.astype(np.uint8)
    cv2.imwrite("eig{}.png".format(i), norm_image.reshape(W, H))

#recontruct 8_6.png
eig_chooses = [5, 50, 150, e_faces.shape[1]]
for n in eig_chooses:
    recon(os.path.join(train_path, "8_6.png"), e_faces, x_mean, n, "rec{}.png".format(n), True)

#test dataset dim100 t-sne
dim100, labels = test_tsne(test_path, e_faces, 100)
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v:k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])
tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(dim100)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
visualize_scatter(tsne_result_scaled, label_ids, id_to_label_dict)

#for TA testing
recon(TA_img_path, e_faces, x_mean, e_faces.shape[1], TA_output_path, False)
