import cv2, glob, os, numpy as np, matplotlib.pyplot as plt, tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd, seaborn as sn
from sklearn.metrics import confusion_matrix

from .utils import dump_object, load_object

DEFAULT_CLUSTERS = 4000
MODULE_DIR       = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR         = os.path.join(MODULE_DIR, 'data')


class BoWDetection(object):
    def __init__(self, x_train, y_train, load_cached = True, cache_dir = DATA_DIR, prefix = ''): # for train
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        self.n_clusters = 0
        self.kmeans_cluster = None
        self.x_org_train = x_train
        self.y_org_train = y_train
        
        self.x_mask        = []
        self.x_keypoints   = []
        self.x_descriptors = []

        self.load_cached = True
        self.cache_dir   = cache_dir
        self.prefix      = prefix

        if os.path.exists(self.cache_dir) == False:
            os.makedirs(self.cache_dir)
        pass
    # __init__

    def build_descriptor(self):
        import copyreg
        def _pickle_keypoints(point): # prevent error for serialization cv2.KeyPoint (registry pickle to specific class)
            return cv2.KeyPoint, (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

        file_keypoints    = os.path.join(self.cache_dir, self.prefix + 'keypoints.pkl')
        file_descriptors  = os.path.join(self.cache_dir, self.prefix + 'descriptors.pkl')
        file_mask         = os.path.join(self.cache_dir, self.prefix + 'masks.pkl')
        
        if self.load_cached == True and os.path.exists(file_keypoints) == True and os.path.exists(file_descriptors) == True and os.path.exists(file_mask) == True:
            self.x_keypoints    = load_object(file_keypoints)
            self.x_descriptors  = load_object(file_descriptors)
            self.x_mask         = load_object(file_mask)
        else:
            self.x_keypoints   = []
            self.x_descriptors = []
            self.x_mask = np.zeros(shape = (len(self.x_org_train)), dtype = np.bool)
            for i in tqdm.tqdm(range(len(self.x_org_train))):
                [keypoints, descriptors] = self.feature_detector.detectAndCompute(self.x_org_train[i], None)
                if keypoints is not None and descriptors is not None:
                    self.x_keypoints.append(keypoints)
                    self.x_descriptors.append(descriptors)
                    self.x_mask[i] = True
            print('Dump keypoints: '   , dump_object(self.x_keypoints   , file_keypoints))
            print('Dump descriptors: ' , dump_object(self.x_descriptors , file_descriptors))
            print('Dump masks: '       , dump_object(self.x_mask        , file_mask))
        
        self.x_train        = self.x_org_train[self.x_mask]
        self.y_train        = self.y_org_train[self.x_mask]
        print('Total images: %d'%(len(self.x_train)))
    # build_descriptor

    def view_descriptor(self, n_len = 8, labels = None):
        idx = np.random.randint(0, len(self.x_train), n_len);
        plt.figure(figsize=(18,18));
        for i in range(len(idx)):
            image = self.x_train[idx[i]].copy();
            image = cv2.drawKeypoints(self.x_train[idx[i]], self.x_keypoints[idx[i]], image)
            plt.subplot(1, n_len, i + 1);
            if labels is None:
                plt.title('%s'%(self.y_train[idx[i]])); plt.axis('off');
            else:
                plt.title('%s'%(labels[self.y_train[idx[i]]])); plt.axis('off');
            self.test = image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap = 'gray');
        plt.show()
        pass

    def explain_keypoint(self, image_idx, key_idx):
        if self.x_train is not None:
            image = self.x_train[image_idx].copy();
            image = cv2.drawKeypoints(self.x_train[image_idx], [self.x_keypoints[key_idx][key_idx]], image)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap = 'gray')
            plt.show()
        print('\n+ Keypoint:')
        kp = self.x_keypoints[image_idx][key_idx]
        dsc = self.x_descriptors[image_idx][key_idx]
        print('angle:', kp.angle)
        print('class_id:', kp.class_id)
        print('octave (image scale where feature is strongest):', kp.octave)
        print('pt(x,y):', kp.pt)
        print('response:', kp.response)
        print('size:', kp.size)
        print('\n+ Descriptor:')
        plt.imshow(dsc.reshape(16,8), interpolation='none');
        plt.show()

    def build_local_patches(self):
        file_local_patches  = os.path.join(self.cache_dir, self.prefix + 'localpatches.pkl')
        self.local_patches  = None
        if self.load_cached == True and os.path.exists(file_local_patches) == True:
            self.local_patches = load_object(file_local_patches)
        else:
            # local patches from SIFT Features
            # restructures list into vstack array of shape: M samples x N features for sklearn
            total_patches = 0
            for descriptor in self.x_descriptors:
                total_patches = total_patches + len(descriptor) 
            shape = [total_patches] + list(self.x_descriptors[0].shape[1:])
            
            idx = 0
            self.local_patches = np.zeros(shape, dtype = self.x_descriptors[0].dtype)
            for descriptor in tqdm.tqdm(self.x_descriptors):
                self.local_patches[idx: idx + len(descriptor)] = descriptor[...]
                idx = idx + len(descriptor)
            dump_object(self.local_patches, file_local_patches)
        print('Total local patches in dictionary: %s' % (len(self.local_patches)))
    # build_local_patches

    def build_codewords(self, n_clusters = DEFAULT_CLUSTERS, verbose = 1):
        from sklearn.externals import joblib
        file_kmeans_cluster = os.path.join(self.cache_dir, self.prefix + 'kmeanscluster.pkl')
        file_kmeans_ret = os.path.join(self.cache_dir, self.prefix + 'kmeansret.pkl')
        if self.load_cached == True and os.path.exists(file_kmeans_cluster) == True and os.path.exists(file_kmeans_ret) == True:
            self.kmeans_cluster = joblib.load(file_kmeans_cluster)
            print('Load kmeans cluster finished!')            
            self.n_clusters = n_clusters
            self.kmeans_ret = joblib.load(file_kmeans_ret)
            print('Load kmeans ret finished!')
        else:
            # Run kmeans to generate codewords dictionary
            # Each cluster denotes a particular visual word 
            self.n_clusters = n_clusters
            self.kmeans_cluster = MiniBatchKMeans(n_clusters = self.n_clusters, batch_size = self.n_clusters, init_size = 3 * self.n_clusters, verbose = verbose)
            self.kmeans_ret = self.kmeans_cluster.fit_predict(self.local_patches);
            joblib.dump(self.kmeans_cluster, file_kmeans_cluster)
            print('Dump kmeans cluster finished!')
            joblib.dump(self.kmeans_ret, file_kmeans_ret)
            print('Dump kmeans ret finished!')
    # build_codewords

    def build_histogram_train(self):
        from sklearn.externals import joblib
        file_mega_histogram = os.path.join(self.cache_dir, self.prefix + 'mega_histogram.pkl')
        file_norm_mega_histogram = os.path.join(self.cache_dir, self.prefix + 'norm_mega_histogram.pkl')
        file_scale_transform = os.path.join(self.cache_dir, self.prefix + 'scale_transform.pkl')
        
        if self.load_cached == True and os.path.exists(file_mega_histogram) == True and os.path.exists(file_norm_mega_histogram) == True and os.path.exists(file_scale_transform) == True:
            self.mega_histogram = joblib.load(file_mega_histogram)
            print('Load mega histogram finished!')            
            self.norm_mega_histogram = joblib.load(file_norm_mega_histogram)
            print('Load norm mega histogram finished!')            
            self.scale_transform = joblib.load(file_scale_transform)
            print('Load scale transform finished!')            
        else:
            # vocabulary comprises of a set of histograms of encompassing all descriptions for all images
            self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(len(self.x_train))]);
            old_count = 0;
            for i in range(len(self.x_train)):
                l = len(self.x_descriptors[i])
                for j in range(l):
                    idx = self.kmeans_ret[old_count+j]
                    self.mega_histogram[i][idx] += 1
                old_count += l
            # Normalize Codewords Dictionary for Learning
            self.scale_transform = StandardScaler().fit(self.mega_histogram)
            self.norm_mega_histogram = self.scale_transform.transform(self.mega_histogram)

            joblib.dump(self.mega_histogram, file_mega_histogram)
            print('Dump mega histogram finished!')
            joblib.dump(self.norm_mega_histogram, file_norm_mega_histogram)
            print('Dump norm mega histogram finished!')
            joblib.dump(self.scale_transform, file_scale_transform)
            print('Dump scale transform finished!')
    # build_histogram_train
    
    def view_image(self, imageidx, labels = None, save_path = None):
        if labels is not None:
            plt.imshow(self.x_train[imageidx,:,:,0], cmap='gray'),  plt.axis('off'), plt.title("Image %s (%s)"%(imageidx, labels[self.y_train[imageidx]]))
        else:
            plt.imshow(self.x_train[imageidx,:,:,0], cmap='gray'),  plt.axis('off'), plt.title("Image %s (%d)"%(imageidx, self.y_train[imageidx]))
        plt.show()
        plt.close()

    def view_hist(self, imageidx, labels = None, save_path = None):
        print('Histogram of Image %s based on codewords: ' %(imageidx), self.mega_histogram[imageidx])
        print('Normalize histogram of Image %s based on codewords: ' %(imageidx), self.norm_mega_histogram[imageidx])
        plt.figure(figsize=(18,9))
        plt.subplot(1,2,1), plotHist(self.mega_histogram[imageidx], 'Histogram of Image %s'%(imageidx))        
        plt.subplot(1,2,2), plotHist(self.norm_mega_histogram[imageidx], 'Normalize histogram of Image %s'%(imageidx))        
        save_figure(plt.gcf(), save_path= save_path)
        plt.show()
        plt.close()
    # view_hist

    def view_hist_codewords(self, save_path = None):
        hist = np.sum(self.mega_histogram, axis=0)
        nhist = np.sum(self.norm_mega_histogram, axis=0)
        print('Histogram of Codewords: ', hist)
        print('Normalize histogram of Codewords: ', nhist)
        plt.figure(figsize=(18,9))
        plt.subplot(1,2,1), plotHist(hist, 'Histogram of Codewords')        
        plt.subplot(1,2,2), plotHist(nhist, 'Normalize histogram of Codewords')        
        save_figure(plt.gcf(), save_path= save_path)
        plt.show()
        plt.close()
    

    def train(self, verbose = False):
        from sklearn.externals import joblib
        file_classifier = os.path.join(self.cache_dir, self.prefix + 'classifier.pkl')
        
        if self.load_cached == True and os.path.exists(file_classifier) == True:
            self.classifier = joblib.load(file_classifier)
            print('Load classifier finished!')  
        else:
            # Training SVM
            self.classifier  = SVC(verbose = verbose)
            print("Training SVM")
            print('Total training images: %s'%(len(self.norm_mega_histogram)))
            print("Train labels:", self.y_train)

            self.classifier.fit(self.norm_mega_histogram, self.y_train)
            print("Training completed");
            joblib.dump(self.classifier, file_classifier)
            print('Dump classifier finished!')
        # if

    # train

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

    def show_confusion_matrix(self, y_test, y_pred, save_path = None):
        plot_confusion_matrix(y_test, predict_labels, ds.labels, title='Average accuracy {acc:0.2f}%\n')
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