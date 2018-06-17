import os, numpy as np, matplotlib.pyplot as plt, cv2
from  keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

mlcnn_dir,_  = os.path.split(os.path.realpath(__file__))
model_dir    = os.path.join(mlcnn_dir, "model")
data_dir     = os.path.join(mlcnn_dir, "data")

fer13_private_dir    = os.path.join(mlcnn_dir, "data", "fer2013", "PrivateTest")
fer13_public_dir     = os.path.join(mlcnn_dir, "data", "fer2013", "PublicTest")

fer13_test_data_x    = os.path.join(model_dir, "fer2013_test_data_x.npy")
fer13_test_data_y_one_hot = os.path.join(model_dir, "fer2013_test_data_y_one_hot.npy")

class MLCNNEmotionDetection(object):

    model_path = os.path.join(model_dir, "fer2013.hdf5")
    labels     = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    mean       = 129.08181762695312
    std        = 65.34889221191406

    detector = None

    def __init__(self, model_path, labels, mean, std):
        self.labels= labels
        self.model = load_model(model_path)        
        print(model_path)
        
        self.num_labels = len(self.labels)
        self.mean = mean # featurewise_center on private test
        self.std  = std   # featurewise_std_normalization on private test
        self.batch_size = 128
        
        pass        

    def preprocessing_input(self, images): # RGB_UBYTE (batch, height, width, channel)
        if type(images) is np.ndarray and len(images.shape)==4 and images.shape[1]==48 and images.shape[2]==48 and images.shape[3]==1:
            result = images
        elif type(images) is np.ndarray and len(images.shape)==3 and images.shape[1]==48 and images.shape[2]==48:
            result = images.reshape((48, 48, 1))
        else:
            result = np.zeros((len(images), 48, 48, 1))
            for idx, image in enumerate(images):
                image = cv2.resize(image, (48, 48))
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                result[idx] = np.reshape(image, (48, 48, 1))             
        return (result - self.mean) / self.std

    def get_test_fer2013(self, to_one_host = False):
        test_data_x = np.load(fer13_test_data_x)
        test_data_y_one_hot = np.load(fer13_test_data_y_one_hot)
        if to_one_host == False:
            test_data_y_one_hot = np.argmax(test_data_y_one_hot, axis = 1)
        return (test_data_x, test_data_y_one_hot)

    def detect(self, x_test, verbose = 0):
        x_new_test     = self.preprocessing_input(x_test)
        y_pred         = self.model.predict(x_new_test, verbose = verbose)
        return np.argmax(y_pred, axis = 1)          
    
    def show_confusion_matrix(self, y_pred, y_test, normalize=True, title='Average accuracy: {precision: 4.3f}\n',
                         verbose = 0, save_path = None):
        fig, (cm, precision) = plot_confusion_matrix(y_pred, y_test, self.labels, normalize, title, verbose, save_path)
        return fig, (cm, precision)

    def show_images(self, images, labels = None, labels_pred = None, title = 'Images',  rows = 4, cols = 8, save_path = None, verbose = 1, random = True):
        img_show = images
        lbl_show = labels
        lbl_pred = labels_pred
        if random == True:
            idx = np.random.randint(low=0, high=len(images), size=(rows * cols))
            img_show = images[idx]
            if lbl_show is not None:
                lbl_show = labels[idx]
            if lbl_pred is not None:
                lbl_pred = labels_pred[idx]
        fig = plot_images(img_show, labels = lbl_show, labels_pred = lbl_pred, 
                          decode_labels = self.labels, title = title, 
                          rows = rows, cols = cols, save_path = save_path, verbose = verbose)
    # show_images

    # Singleton Pattern
    @staticmethod
    def getDetector(model_path = None, labels = None, mean = None, std = None):   
        if MLCNNEmotionDetection.detector == None:
            if model_path is None:
                model_path = MLCNNEmotionDetection.model_path
            if labels is None:
                labels = MLCNNEmotionDetection.labels
            if mean is None:
                mean = MLCNNEmotionDetection.mean
            if std is None:
                std = MLCNNEmotionDetection.std
            MLCNNEmotionDetection.detector = MLCNNEmotionDetection(model_path, labels, mean, std)
        return MLCNNEmotionDetection.detector
    # getDetector
# MLCNNEmotionDetection    
        

def plot_confusion_matrix(y_pred, y_test, labels, normalize=True, title='Average accuracy: {precision: 4.3f}\n',verbose = 0, save_path = None):
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import seaborn as sn

    # calculate confusion matrix from pred, test
    cm = confusion_matrix(y_pred, y_test)
    # precision
    precision = np.sum(np.diag(cm)) / np.sum(cm)
    # normalize confusion matrix
    if normalize == True:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100.0
    if verbose >= 2:
        print('Average accuracy: {precision: 4.3f}\n' % (precision))
        print(cm)

    df_cm = pd.DataFrame(cm, index = [labels[i] for i in range(len(labels))], columns = [labels[i] for i in range(len(labels))])
    plt.figure(figsize = (8,8))

    sn.set(font_scale=1.2)#for label size
    sn.heatmap(df_cm, annot=True,fmt='.1f')

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=18)
    plt.yticks(tick_marks, labels, fontsize=18)

    plt.title(title.format_map({'precision': precision}), fontsize=25)
    
    if verbose >= 1:
        plt.show()
    
    fig = plt.gcf()
    
    if save_path is not None:
        # draw the figure first...
        fig.canvas.draw()
    
        # Now we can save it to a numpy array.
        # fig_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.savefig(save_path)

    plt.close()
    
    return fig, (cm, precision)
# plot_confusion_matrix

def plot_images(images, labels = None, labels_pred = None, decode_labels = None, title = '', 
                rows = 4, cols = 8, save_path = None, is_path = False, data_dir = "", verbose = 1):
    # idx = sorted(range(len(label_batch)), key=lambda k: decode_output(label_batch[k]))
    plt.figure(figsize=(16, 8))
    if is_path == True:
        images = [np.array(Image.open(os.path.join(data_dir, images[i]))) 
                  for i in range(len(images))]
    for row in range(rows):
        for col in range(cols):
            plt.subplot(rows, cols,row * cols + col + 1)
            plt.axis('off')
            title_image = ''
            if labels is not None:
                if decode_labels is not None:
                    title_image = title_image + ' %s '%(decode_labels[int(labels[row * cols + col])])
                else:
                    title_image = title_image + ' %s '%(labels[row * cols + col])
            # if

            if labels_pred is not None:
                plt.subplots_adjust(hspace = 0.5)
                if decode_labels is not None:
                    title_image = title_image + '\n %s(pre) '%(decode_labels[int(labels_pred[row * cols + col])])
                else:
                    title_image = title_image + '\n %s(pre) '%(labels_pred[row * cols + col])
            # if
            if title_image != '':
                plt.title(title_image)

            if len(images[row * cols + col].shape) == 2 or images[row * cols + col].shape[2] == 1:
                plt.imshow(images[row * cols + col][:,:,0], cmap = 'gray')
            else:
                plt.imshow(images[row * cols + col])
    plt.suptitle(title)

    fig = plt.gcf()

    if save_path is not None:
        # draw the figure first...
        fig.canvas.draw()
    
        # Now we can save it to a numpy array.
        # fig_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.savefig(save_path)
    # save_path

    if verbose == 1:
        plt.show()

    plt.close()

    return fig
# plot_images