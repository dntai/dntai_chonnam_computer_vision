import os, tarfile, io, tqdm, glob, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import pickle

from .utils import plot_images, plotHist, save_figure

DATA_DEFAULT_ROOT_DIR               = os.path.abspath('../data')
DATA_DEFAULT_ZIP_FILE               = os.path.join(DATA_DEFAULT_ROOT_DIR, "fer2013.tar.gz")
DATA_DEFAULT_ZIP_EXTRACT_DIR        = os.path.join(DATA_DEFAULT_ROOT_DIR, "dataset")

DEFAULT_VERBOSE                     = 1
DATASET_DEFAULT_FILE                 = os.path.join(DATA_DEFAULT_ZIP_EXTRACT_DIR, 'fer2013','fer2013.csv')
DATASET_DEFAULT_DIR                  = os.path.join(DATA_DEFAULT_ZIP_EXTRACT_DIR, 'fer2013')
DATASET_DEFAULT_DELETE_OLD_DIR       = False

DATASET_DEFAULT_DB_FILE              = os.path.join(DATA_DEFAULT_ROOT_DIR, 'fer2013.h5')

DATASET_DEFAULT_LABELS               = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class Fer2013Dataset(object):
    def __init__(self, data = None, **kwargs):            
        if data is None:
            self.data = Fer2013Dataset.setup(default_args = kwargs)
        else:
            self.data = data

        self.train_files, self.train_label_names, self.train_images, self.train_label_keys, self.train_onehots, self.train_default_labels = self.data[0]
        self.valid_files, self.valid_label_names, self.valid_images, self.valid_label_keys, self.valid_onehots, self.valid_default_labels = self.data[1]
        self.test_files , self.test_label_names , self.test_images , self.test_label_keys , self.test_onehots , self.test_default_labels  = self.data[2]
        self.labels  = self.train_default_labels
        pass
    # __init__

    @staticmethod
    def setup(default_args = {}, **kwargs):
        args = {'dataset_zip_file': DATA_DEFAULT_ZIP_FILE, 
                'dataset_zip_extract_dir': DATA_DEFAULT_ZIP_EXTRACT_DIR,
                'dataset_file': DATASET_DEFAULT_FILE,
                'dataset_dir': DATASET_DEFAULT_DIR,
                'dataset_db_file': DATASET_DEFAULT_DB_FILE,
                'verbose': DEFAULT_VERBOSE}        
        args.update(default_args)
        args.update(kwargs)
        data = setup_fer2013(  dataset_zip_file = args['dataset_zip_file'], dataset_zip_extract_dir = args['dataset_zip_extract_dir'],
                        dataset_file = args['dataset_file'], dataset_dir  = args['dataset_dir'],
                        dataset_db_file = args['dataset_db_file'], verbose = args['verbose'])
        return data
        pass
    # setup
    

    def view_images(self, x, y, title = "Images", save_path = None):
        idx = np.random.randint(low=0, high=len(x), size=(32))
        plot_images(x[idx], y[idx], title=title, decode_labels = self.labels, data_dir = '', rows = 4, cols = 8, is_path = False, save_path = save_path)
        pass
    # view_images

    def view_summary(self, save_path = None):
        print("Number of images in the training dataset:\t{:>6}".format(len(self.train_images)))
        print("Number of images in the public dataset:\t\t{:>6}".format(len(self.valid_images)))
        print("Number of images in the private dataset:\t{:>6}".format(len(self.test_images)))
        print("Image information: %d x %d x %d"%(self.train_images[0].shape[0], self.train_images[0].shape[1], self.train_images[0].shape[2]))


        plt.figure(figsize=(18,9))
        plt.subplot(1,3,1), plotHist(self.train_label_names,'Histogram of Training Dataset')
        plt.subplot(1,3,2), plotHist(self.valid_label_names,'Histogram of Validating Dataset')
        plt.subplot(1,3,3), plotHist(self.test_label_names,'Histogram of Testing Dataset')
        if save_path is not None:
            save_figure(plt.gcf(), save_path)
        plt.show()
        plt.close()
# Fer2013Dataset

def setup_fer2013(dataset_zip_file = DATA_DEFAULT_ZIP_FILE, dataset_zip_extract_dir = DATA_DEFAULT_ZIP_EXTRACT_DIR,
                  dataset_file     = DATASET_DEFAULT_FILE,
                  dataset_dir      = DATASET_DEFAULT_DIR, 
                  dataset_db_file  = DATASET_DEFAULT_DB_FILE, verbose = DEFAULT_VERBOSE):
    flag = -1
    if os.path.exists(dataset_db_file) == True:
        flag = 4
    elif os.path.exists(os.path.join(dataset_dir, "Training"))   == True and os.path.exists(os.path.join(dataset_dir, "PublicTest")) == True and os.path.exists(os.path.join(dataset_dir, "PrivateTest"))== True:
        flag = 3
    elif os.path.exists(dataset_file) == True:
        flag = 2
    elif os.path.exists(dataset_zip_file) == True:
        flag = 1
    else:
        flag = 0

    if flag<=0:
        downloaddir = os.path.split(dataset_zip_file)[0]
        dataset_zip_file = os.path.join(downloaddir,'fer2013.tar.gz')
        download_fer2013(path = downloaddir, is_quiet = (verbose==0), is_force_input = False, is_force_download = False)
    if flag<=1:
        extract_fer2013_zip(input_path = dataset_zip_file, output_path = dataset_zip_extract_dir, verbose = verbose)
    if flag<=2:
        extract_fer2013_dataset(input_path = dataset_file, output_path = dataset_dir, is_forced = False, verbose = verbose)
    if flag<=3:
        files, label_names, images, label_keys, onehots, default_labels = load_fer2013_dataset(dataset_dir = dataset_dir, subdir = "Training", default_labels = None)
        save_fer2013_dataset(files, label_names, images, label_keys, onehots, default_labels, table='train', db_file = dataset_db_file)
        trains = (files, label_names, images, label_keys, onehots, default_labels)

        files, label_names, images, label_keys, onehots, default_labels = load_fer2013_dataset(dataset_dir = dataset_dir, subdir = "PublicTest", default_labels = default_labels)
        save_fer2013_dataset(files, label_names, images, label_keys, onehots, default_labels, table='valid', db_file = dataset_db_file)
        valids = (files, label_names, images, label_keys, onehots, default_labels)

        files, label_names, images, label_keys, onehots, default_labels = load_fer2013_dataset(dataset_dir = dataset_dir, subdir = "PrivateTest", default_labels = default_labels)
        save_fer2013_dataset(files, label_names, images, label_keys, onehots, default_labels, table='test', db_file = dataset_db_file)
        tests = (files, label_names, images, label_keys, onehots, default_labels)
    
    if flag==4:
        files, label_names, images, label_keys, onehots, default_labels = load_fer2013_fromfile(table = "train", db_file = dataset_db_file)
        trains = (files, label_names, images, label_keys, onehots, default_labels)
        files, label_names, images, label_keys, onehots, default_labels = load_fer2013_fromfile(table = "valid", db_file = dataset_db_file)
        valids = (files, label_names, images, label_keys, onehots, default_labels)
        files, label_names, images, label_keys, onehots, default_labels = load_fer2013_fromfile(table = "test", db_file = dataset_db_file)
        tests = (files, label_names, images, label_keys, onehots, default_labels)
    
    return (trains, valids, tests)
# setup_fer2013

def load_fer2013_fromfile(table = "train", db_file = DATASET_DEFAULT_DB_FILE):
    import h5py
    
    files           = None
    label_names     = None
    images          = None
    label_keys      = None
    onehots         = None
    default_labels  = None
    # Save data
    with h5py.File(db_file, 'r') as f:
        if table in f.keys():
            grp = f[table]
            if 'files' in grp.keys():
                files = grp['files'].value.astype('U')
            if 'label_names' in grp.keys():
                label_names = grp['label_names'].value.astype('U')
            if 'images' in grp.keys():
                images = grp['images'].value
            if 'label_keys' in grp.keys():
                label_keys = grp['label_keys'].value
            if 'onehots' in grp.keys():
                onehots = grp['onehots'].value
            if 'default_labels' in grp.keys():
                default_labels = grp['default_labels'].value.astype('U')
    # with
    return files, label_names, images, label_keys, onehots, default_labels
# load_fer2013_fromfile

def save_fer2013_dataset(files, label_names, images, label_keys, onehots, default_labels, table = "train", db_file = DATASET_DEFAULT_DB_FILE):
    import h5py
    # Save data
    with h5py.File(db_file, 'a') as f:
        if f.get(table) is not None:
            f.pop(table)
        grp = f.create_group(table)
        grp.create_dataset(name='files', shape=(len(files),),dtype = h5py.special_dtype(vlen=str), data=files.astype('S'))
        grp.create_dataset(name='label_names', shape=(len(label_names),),dtype = h5py.special_dtype(vlen=str), data=label_names.astype('S'))
        grp.create_dataset(name='images', shape=images.shape, dtype = images.dtype, data=images)
        grp.create_dataset(name='label_keys', shape=label_keys.shape, dtype=label_keys.dtype, data=label_keys)
        grp.create_dataset(name='onehots', shape=onehots.shape, dtype=onehots.dtype, data=onehots)
        grp.create_dataset(name='default_labels', shape=(len(default_labels),),dtype = h5py.special_dtype(vlen=str), data=default_labels.astype('S'))
    # with
# save_fer2013_dataset

def load_fer2013_dataset(dataset_dir = DATASET_DEFAULT_DIR, subdir = "Training", default_labels = None):
    files  = []
    label_names = []
    images = []
    label_keys = []
    onehots    = []
    onehots_dict = {}
    # Load Dataset
    if os.path.exists(os.path.join(dataset_dir, subdir)):
        paths = glob.glob(os.path.join(dataset_dir, subdir, '*','*.png'))
        paths.sort()
        for i in tqdm.tqdm(range(len(paths))):
            path = paths[i]
            file = os.path.relpath(paths[i], dataset_dir)
            label_name = os.path.basename(os.path.split(path)[0])
                
            if onehots_dict.get(label_name) is None:
                onehots_dict[label_name] = 1
            else:
                onehots_dict[label_name] = onehots_dict[label_name] + 1
                
            image = np.array(Image.open(os.path.join(dataset_dir, path)))
            if len(image.shape) == 2:
                image = image.reshape(image.shape[0], image.shape[1], 1)

            files.append(file)
            label_names.append(label_name)
            images.append(image)
        # for

        files        = np.array(files)
        label_names  = np.array(label_names)
        images       = np.array(images)
            
        if default_labels is None:
            default_labels = list(onehots_dict.keys())
            default_labels.sort()                           
            default_labels = np.array(default_labels)
        # if
        default_labels = default_labels.astype('U')
        

        for i in range(len(label_names)):
            label_name = label_names[i]
            onehot = np.zeros(len(default_labels), dtype=np.float32)
            label_key = np.argmax(default_labels == label_name)                
            onehot[label_key] = 1.0
                
            onehots.append(onehot)
            label_keys.append(label_key)
        # for

        label_keys     = np.array(label_keys)
        onehots        = np.array(onehots)
        default_labels = np.array(default_labels)
                
    return files, label_names, images, label_keys, onehots, default_labels
# load_fer2013_dataset

def extract_fer2013_dataset(input_path = DATASET_DEFAULT_FILE, output_path = DATASET_DEFAULT_DIR, is_forced = False, verbose = False):
    from IPython import display 
    import os, sys, cv2, numpy as np, shutil, tqdm
    
    if (is_forced == False) and (os.path.exists(os.path.join(output_path, 'Training')) == True
                                 and os.path.exists(os.path.join(output_path, 'PrivateTest')) == True
                                 and os.path.exists(os.path.join(output_path, 'PublicTest')) == True):
        if verbose >=1:
            print("+ Dataset Directory:")
            print("\n".join(os.listdir(output_path)))
        return

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    data = np.genfromtxt(os.path.join(input_path),delimiter=',', dtype=np.object) #, encoding='bytes');
  
    # Read data (col 0: labels, 1: image buffer, 2: usage)
    labels       = data[1:,0].astype(np.int32)
    image_buffer = data[1:,1]
    images       = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
    usage        = data[1:,2]
    dataset      = list(zip(labels, images, usage))
    
    # display.display('Process: 0 / %d'%(len(dataset)), display_id = 'process')
    for i in tqdm.tqdm(range(len(dataset))):
        d = dataset[i]
        usage_path = os.path.join(output_path, d[-1].decode("utf-8"))
        label_path = os.path.join(usage_path, label_names[d[0]])
        img = d[1].reshape((48,48))
        img_name = 'img_%05d.png' % i
        img_path = os.path.join(label_path, img_name)
        if not os.path.exists(usage_path):
            os.makedirs(usage_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        cv2.imwrite(img_path, img)
        # if verbose >=1:
            # print('Write %s' % (img_path))
            # display.display('Process %d / %d: %s'%(i, len(dataset), img_path), display_id = 'process')
    # for
    if verbose >=1:
        print("+ Dataset Directory:")
        print("\n".join(os.listdir(output_path)))
# extract_fer2013_dataset

def extract_fer2013_zip(input_path = DATA_DEFAULT_ZIP_FILE, output_path = DATA_DEFAULT_ZIP_EXTRACT_DIR, verbose = DEFAULT_VERBOSE):
    if os.path.isfile(os.path.join(output_path,'fer2013','fer2013.csv')) == False and os.path.isfile(input_path) == True:
        tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
        tar = tarfile.open(fileobj=ProgressFileObject(input_path))
        tar.extractall(path = output_path)
        tar.close()
    # if    
    if verbose >=1:
        print("+", output_path + '/fer2013')
        content = os.listdir(output_path + '/fer2013')
        print("\n".join(content))
    # if
# extract_fer2013_zip

def download_fer2013(path = DATA_DEFAULT_ROOT_DIR, is_quiet = True, is_force_input = False, is_force_download = False):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from os.path import expanduser    
    import getpass, os
    
    kaggle_dir = os.path.join(expanduser("~"),'.kaggle')
    if os.path.exists(kaggle_dir) == False:
        os.makedirs(kaggle_dir)
    with open(os.path.join(kaggle_dir,'kaggle.json'), 'wt') as f:
        f.writelines('{"username":"","key":""}')
    
    if hasattr(download_fer2013, 'username') == False or is_force_input == True:
        download_fer2013.username  = getpass.getpass('Input the username:')
    username = download_fer2013.username
    
    if hasattr(download_fer2013, 'key') == False or is_force_input == True:
        download_fer2013.key = getpass.getpass('Input the key:')
    key = download_fer2013.key
    
    if hasattr(download_fer2013, 'kaggle') == False:
        download_fer2013.kaggle = KaggleApi()
    kaggle = download_fer2013.kaggle
          
    kaggle.set_config_value("username", username, quiet = True)
    kaggle.set_config_value("key", key, quiet = True)
    kaggle.authenticate()
    kaggle.competition_download_file("challenges-in-representation-learning-facial-expression-recognition-challenge", "fer2013.tar.gz", 
                                     path = path, force=is_force_download, quiet = is_quiet)
    kaggle.competition_download_file("challenges-in-representation-learning-facial-expression-recognition-challenge", "example_submission.csv", 
                                     path = path, force=is_force_download, quiet = is_quiet)
# kaggle_download_fer2013

def pickle_array_numpy(np_data):
    encoded = []
    for item in np_data:
        serialized = pickle.dumps(item, protocol=0) # protocol 0 is printable ASCII
        encoded.append(serialized.decode("ISO-8859-1"))
    encoded = np.array(encoded)
    return encoded
# pickle_array_numpy

def unpickle_array_numpy(np_data):
    decoded = []
    for item in np_data:
        deserialized = pickle.loads(item.encode("ISO-8859-1"))
        decoded.append(deserialized)
    decoded = np.array(decoded)
    return decoded
# unpickle_array_numpy

#######################################
# EXTRACT TAR FILE
#######################################
# import tarfile, io, os, tqdm

def extract_tarfile(detination, source):
    tarfile.TarFile.fileobject = get_file_progress_file_object_class(on_progress)
    tar = tarfile.open(fileobj=ProgressFileObject(source))
    tar.extractall(path = detination)
    tar.close()
# extract_tarfile

# Extract dataset zip file
def get_file_progress_file_object_class(on_progress):
    class FileProgressFileObject(tarfile.ExFileObject):
        def read(self, size, *args):
          on_progress(self.name, self.position, self.size)
          return tarfile.ExFileObject.read(self, size, *args)
    return FileProgressFileObject
# get_file_progress_file_object_class

class TestFileProgressFileObject(tarfile.ExFileObject):
    def read(self, size, *args):
      on_progress(self.name, self.position, self.size)
      return tarfile.ExFileObject.read(self, size, *args)
# TestFileProgressFileObject

class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        self.pbar = tqdm.tqdm(total=self._total_size)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        self.pbar.update(size)
        # print("Overall process: %d of %d" %(self.tell(), self._total_size))
        return io.FileIO.read(self, size)
    
    def __del__(self):
        self.pbar.close()
# ProgressFileObject

def on_progress(filename, position, total_size):
    print("%s: %d of %s" %(filename, position, total_size))
# on_progress

#######################################