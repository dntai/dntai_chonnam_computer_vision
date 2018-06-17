import cv2, glob, os, numpy as np, matplotlib.pyplot as plt, tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd, seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from .utils import dump_object, load_object

DEFAULT_CLUSTERS = 4000
MODULE_DIR       = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR         = os.path.join(MODULE_DIR, 'data')
DEFAULT_PREFIX   = ''

class BoWDetection(object):
    def __init__(self, cache_dir = DATA_DIR, prefix = DEFAULT_PREFIX): # for predict, evaluation

        self.feature_detector = cv2.xfeatures2d.SIFT_create()

        self.cache_dir       = cache_dir
        self.prefix          = prefix

        file_kmeans_cluster  = os.path.join(self.cache_dir, self.prefix + 'kmeanscluster.pkl')
        file_scale_transform = os.path.join(self.cache_dir, self.prefix + 'scale_transform.pkl')
        file_classifier      = os.path.join(self.cache_dir, self.prefix + 'classifier.pkl')

        self.kmeans_cluster  = None
        flag = 0

        if os.path.exists(file_kmeans_cluster) == True:
            self.kmeans_cluster = joblib.load(file_kmeans_cluster)
            self.kmeans_cluster.verbose = 0
            self.n_clusters = self.kmeans_cluster.n_clusters
            print('Load kmeans cluster finished!')            
            flag = flag + 1
           
        if os.path.exists(file_scale_transform) == True:
            self.scale_transform = joblib.load(file_scale_transform)
            print('Load scale transform finished!')
            flag = flag + 1

        if os.path.exists(file_classifier) == True:
            self.classifier = joblib.load(file_classifier)
            print('Load classifier finished!')  
            flag = flag + 1
        
        if flag < 3:
            print("Load incomplete!")
        else:
            print("Load completed!")

        pass
    # __init__

    def predict(self, image, labels = None, verbose = 1, save_path_image = None, save_path_hist = None):
        # Extract Feature
        _, descriptors = self.feature_detector.detectAndCompute(image, None)
        # Calculate cluster of codewords 
        image_clusters = self.kmeans_cluster.predict(descriptors)
        # Generate image model = histogram of codewords
        vocab = np.zeros((1, self.n_clusters))
        for cluster in image_clusters:
            vocab[0][cluster] += 1
        norm_vocab = self.scale_transform.transform(vocab)
        # predict label
        predict_label = self.classifier.predict(norm_vocab)

        if verbose>=1:
            # View image
            plt.imshow(image[:,:,0], cmap = 'gray')
            plt.title('View Image')
            plt.axis('off')
            if save_path_image is not None:
                save_figure(plt.gcf(), save_path_image)
            plt.show()
            # Local patches
            print('Image local patches: ', descriptors.shape)
            print(descriptors)
            # Codewords
            print('%s codewords: '%(len(descriptors)))
            print(image_clusters)
            # Histogram model of codewords
            print('Histogram Model:', vocab.shape, vocab)
            print('Normal Histogram Model:', norm_vocab.shape, norm_vocab)
            # View histogram model
            plt.figure(figsize=(18,9))
            plt.subplot(1,2,1), plotHist(vocab[0], 'Histogram Model of Image ')        
            plt.subplot(1,2,2) , plotHist(norm_vocab[0], 'Normalize histogram Model of Image')        
            plt.show()
            if save_path_hist is not None:
                save_figure(plt.gcf(), save_path_hist)
            plt.close()
            # Classification
            if labels is None:
                print('Predict label: ', predict_label[0])
            else:
                print('Predict label: ', predict_label[0], ' - ', labels[predict_label[0]])
        return predict_label[0]
    # predict

    def evaluate(self, images):
        # Generate image model = histogram of codewords
        vocab = np.zeros((len(images), self.n_clusters))
        for i in range(len(images)):
            _, descriptors = self.feature_detector.detectAndCompute(images[i], None)
            # Calculate cluster of codewords 
            if descriptors is None:
                descriptors = np.zeros([1, 128])
            image_clusters = self.kmeans_cluster.predict(descriptors)
            for cluster in image_clusters:
                vocab[i][cluster] += 1
        norm_vocab = self.scale_transform.transform(vocab)
        # classifier
        predict_labels = self.classifier.predict(norm_vocab)
        return predict_labels

    def show_confusion_matrix(self, y_test, y_pred, labels = None, save_path = None):
        plot_confusion_matrix(y_test, y_pred, labels, title='Average accuracy {acc:0.2f}%\n')
        if save_path is not None:
            save_figure(plt.gcf(), save_path)
        plt.show()
        plt.close()
    # show_confusion_matrix
    
# BoWDetection

def plotHist(histogram, title='Codewords Dictionary', save_path = None):
    x_scalar = np.arange(len(histogram))
    y_scalar = histogram 
    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title(title)

def save_figure(fig, save_path):
    if save_path != None:
        (dir, file) = os.path.split(save_path)
        if os.path.exists(dir) == False:
            os.makedirs(dir)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        # data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        # plt.gcf().savefig(save_path)
        # Save just the portion _inside_ the second axis's boundaries
        # extent = plt.gca().get_tightbbox(plt.gcf().canvas.renderer).transformed(plt.gcf().dpi_scale_trans.inverted())
        fig.savefig(save_path)
# def save_figure

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=True,
                          title='Average accuracy \n',
                          cmap=plt.cm.Blues, verbose = 0, precision = 0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    import itertools
    
    cm  = confusion_matrix(y_test, y_pred)
    accuracy = (np.sum(np.diag(cm)) / np.sum(cm)) * 100.0

    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100.0
        if verbose == 1:
            print("Normalized confusion matrix")
    else:
        if verbose == 1:
            print('Confusion matrix, without normalization')
    
    if verbose == 1:
        print(cm)

    plt.figure(figsize=(18, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title.format_map({'acc':accuracy}), fontsize=25)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '{:.'+ '%d'%(precision) +'f} %' if normalize else '{:d}'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=16)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)