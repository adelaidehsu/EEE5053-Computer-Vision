from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
import cv2
import sys
import os

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

def preprocess(train_path):
    x = np.array([])
    c = []
    W, H = 0, 0
    for f in glob.glob(os.path.join(train_path, "*.png")):
        pos = os.path.basename(f).find('_')
        c.append(int(os.path.basename(f)[:pos]))
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if W==0 and H==0:
            [W, H] = img.shape
        img = img.flatten()
        if x.size == 0:
            x = img
        else:
            x = np.vstack((x, img))
    x = np.transpose(x) #(W*H, N)
    return x, x.shape[1], c, W, H

def pca(X, W, H):
    x_mean = np.mean(X, axis = 1).reshape(W*H,1)
    ma_data = X - x_mean
    U, S, V = np.linalg.svd(ma_data, full_matrices=False)
    return U, x_mean

def lda(X, C_lst, W, H, dim):
    c = np.unique(np.array(C_lst))
    X = X.T
    [d, N] = X.shape
    glb_mean = np.mean(X, axis = 1).reshape(d,1)
    Sw = np.zeros((d, d), dtype = np.float32)
    Sb = np.zeros((d, d), dtype = np.float32)
    for i in c:
        Xi = X[:, np.where(C_lst==i)[0]]
        C_mean = np.mean(Xi, axis=1).reshape(d,1)
        Sw = Sw + np.dot((Xi-C_mean), (Xi-C_mean).T)
        Sb = Sb + N*np.dot((C_mean-glb_mean), (C_mean-glb_mean).T)
    U, S, V = np.linalg.svd(np.linalg.inv(Sw).dot(Sb), full_matrices=False)
    return U[:, 0:dim]

def test(test_path):
    test_imgs = np.array([])
    W, H = 0, 0
    c = []
    for f in glob.glob(os.path.join(test_path, "*.png")):
        pos = os.path.basename(f).find('_')
        c.append(int(os.path.basename(f)[:pos]))
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if W==0 and H==0:
            W, H = img.shape[0], img.shape[1]
        img = img.flatten()
        if test_imgs.size == 0:
            test_imgs = img
        else:
            test_imgs = np.vstack((test_imgs, img))
    test_imgs = np.transpose(test_imgs) #(W*H, Nt)
    return test_imgs, c

def project(X, Weight, W, H):
    X_mean = np.mean(X, axis = 1).reshape(W*H, 1)
    X_ma = X - X_mean
    y = np.dot(X_ma.T, Weight)
    return y

def visualize_scatter(data_2d, label_ids, id_to_label_dict, figsize=(15,15)):
    plt.figure(figsize=figsize)
    plt.grid()
    nb_classes = len(np.unique(label_ids))
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.gist_rainbow(label_id / float(nb_classes)),
                    linewidth='2',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.title("T-SNE DIM=30", fontsize=20)
    plt.savefig('T-SNE dim30.png')
    plt.close()

data_path, TA_foutput_path =  sys.argv[1], sys.argv[2]
#dataset partition
train_path, test_path = data_preprocess(data_path)
#pca
X, N, C_lst, W, H = preprocess(train_path)
c = np.unique(np.array(C_lst))
e_faces, glb_mean = pca(X, W, H)
weights = project(X, e_faces, W, H)
lda_eig_vec = lda(weights[:, 0:N-len(c)], C_lst, W, H, len(c)-1)
f_faces = np.dot(e_faces[:, 0:N-len(c)], lda_eig_vec)
#plot first five fisher faces
for i in range(5):
    norm_image = cv2.normalize(f_faces[:,i].reshape(W*H,1), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image.astype(np.uint8)
    cv2.imwrite("fis{}.png".format(i), norm_image.reshape(W, H))
#test t-sne 30 dim
test_X, test_labels = test(test_path)
test_c = np.unique(np.array(test_labels))
dim30 = project(test_X, f_faces[:, 0:30], W, H)
label_to_id_dict = {v:i for i, v in enumerate(test_c)}
id_to_label_dict = {v:k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in test_labels])
tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(dim30)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
visualize_scatter(tsne_result_scaled, label_ids, id_to_label_dict)
#For TA testing (First Fisher Face)
first_fish = cv2.normalize(f_faces[:,0].reshape(W*H,1), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
first_fish.astype(np.uint8)
cv2.imwrite(TA_foutput_path, first_fish.reshape(W, H))


#k-NN experiment
neighbors = [1, 3, 5]
reduc_dim = [3, 10, 39]
param = [i for i in range(9)]
Y_train = np.array(C_lst)
Y_test = np.array(test_labels)
#pca
pca_test_acc = []
pca_val_acc = []
for k in neighbors:
    for n in reduc_dim:
        X_train = weights[:, 0:n]
        X_test = project(test_X, e_faces[:, 0:n], W, H)
        knn = KNeighborsClassifier(n_neighbors=k)
        val_scores = cross_val_score(knn, X_train, Y_train, cv=3, scoring='accuracy')
        pca_val_acc.append(val_scores.mean())
        fitted = knn.fit(X_train, Y_train)
        Y_pred = fitted.predict(X_test)
        test_accuracy = metrics.accuracy_score(Y_test, Y_pred)
        pca_test_acc.append(test_accuracy)
plt.plot(param, pca_val_acc)
plt.xlabel('(K, N) Index')
plt.ylabel('Recognition Rate(%)')
plt.savefig('PCA_Validation_Acc.png')
plt.close()
plt.plot(param, pca_test_acc)
#print(pca_val_acc)
#print(pca_test_acc)
plt.xlabel('(K, N) Index')
plt.ylabel('Recognition Rate(%)')
plt.savefig('PCA_Test_Acc.png')
plt.close()

#fisherface
f_val_acc = []
f_test_acc = []
for k in neighbors:
    for n in reduc_dim:
        X_train = project(X, f_faces[:, 0:n], W, H)
        X_test = project(test_X, f_faces[:, 0:n], W, H)
        knn = KNeighborsClassifier(n_neighbors=k)
        val_scores = cross_val_score(knn, X_train, Y_train, cv=3, scoring='accuracy')
        f_val_acc.append(val_scores.mean())
        fitted = knn.fit(X_train, Y_train)
        Y_pred = fitted.predict(X_test)
        test_accuracy = metrics.accuracy_score(Y_test, Y_pred)
        f_test_acc.append(test_accuracy)
plt.plot(param, f_val_acc)
plt.xlabel('(K, N) Index')
plt.ylabel('Recognition Rate(%)')
plt.savefig('f_Validation_Acc.png')
plt.close()
plt.plot(param, f_test_acc)
#print(f_val_acc)
#print(f_test_acc)
plt.xlabel('(K, N) Index')
plt.ylabel('Recognition Rate(%)')
plt.savefig('f_Test_Acc.png')
plt.close()