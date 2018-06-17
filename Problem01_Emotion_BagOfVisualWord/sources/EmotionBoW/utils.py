import matplotlib.pyplot as plt, cv2, numpy as np, os
import pickle

#################################################################################
# UTILS  FUNCTIONS
#################################################################################
def dump_object(object, path):
    f = open(path, 'wb')
    pickle.dump(object, f)
    f.close()
    return True
# dump_object

def load_object(path):
    f = open(path, 'rb')
    object = pickle.load(f)
    f.close()
    return object
# load_object

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

def plot_images(images, labels, decode_labels = None, title = '', 
                rows = 8, cols = 4, save_path = None, is_path = False, data_dir = ""):
    # idx = sorted(range(len(label_batch)), key=lambda k: decode_output(label_batch[k]))
    plt.figure(figsize=(16, 8))
    if is_path == True:
        images = [np.array(Image.open(os.path.join(data_dir, images[i]))) 
                  for i in range(len(images))]
    for row in range(rows):
        for col in range(cols):
            plt.subplot(rows, cols,row * cols + col + 1)
            plt.axis('off')
            if decode_labels is not None:
                plt.title('%s'%(decode_labels[int(labels[row * cols + col])]))
            else:
                plt.title('%s'%(labels[row * cols + col]))
            if len(images[row * cols + col].shape) == 2 or images[row * cols + col].shape[2] == 1:
                plt.imshow(images[row * cols + col][:,:,0], cmap = 'gray')
            else:
                plt.imshow(images[row * cols + col])
    plt.suptitle(title)
    save_figure(plt.gcf(), save_path)
    plt.show()
    plt.close()
    
def plotHist(labels, title='Codewords Dictionary'):
    dictionary = {}
    labels.sort()
    for labeli in labels:
        if dictionary.get(labeli)==None:
            dictionary[labeli] = 1
        else:
            dictionary[labeli] += 1
    names  = list(dictionary.keys())
    names.sort()
    values = [dictionary[key] for key in names]
    ranges = np.arange(len(names))
    plt.title(title)
    plt.bar(ranges, values, tick_label =names) 

def plotHistory(epoch, acc, val_acc, xlabel = 'epoch', ylabel = 'accuracy', title='Model Accuracy', is_show = True):
    # Be sure to only pick integer tick locations.
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # summarize history for accuracy
    plt.plot(epoch, acc)
    plt.plot(epoch, val_acc)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['train','test'], loc='upper left')
# plotHistory



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
    plt.title(title, fontsize=25)
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
# plot_confusion_matrix

#################################################################################
# OTHERS  FUNCTIONS
#################################################################################