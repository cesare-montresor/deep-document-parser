import glob, math
import numpy as np
import utils
from sklearn.utils import shuffle
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip

csvname = 'labels.csv'

trainfile = 'train.npz'
validfile = 'valid.npz'
testfile = 'test.npz'

datasources_path = './datasources/'
recordings_path = './recordings/'
datasets_path = './datasets/'
video_output_path = './videos/'
default_batch_size = 32


def loadDatasetGenerators(name, batch_size=default_batch_size):  # return generator
    basepath = datasets_path + name + '/'
    train_dataset = loadDataset(basepath + trainfile)
    valid_dataset = loadDataset(basepath + validfile)

    train_size = len(train_dataset['path'])
    valid_size = len(valid_dataset['path'])
    # print(train_dataset.keys())

    sample = utils.loadImage(train_dataset['path'][0])
    sample_shape = sample.shape
    sample_type = type(sample[0][0][0])

    info = {
        'n_train':train_size,
        'n_train_batch': math.ceil(train_size/batch_size),
        'n_valid':valid_size,
        'n_valid_batch': math.ceil(valid_size/batch_size),
        'input_shape': sample_shape,
        'data_type': sample_type
    }

    return datasetGenerator(train_dataset, batch_size), datasetGenerator(valid_dataset, batch_size), info

def datasetGenerator(dataset, batch_size=default_batch_size, augment=False):
    n_dataset = len(dataset)
    paths = dataset['path']
    labels = dataset['y']
    #print(paths)
    #print(labels)
    while 1:
        paths, labels = shuffle(paths, labels)
        for offset in range(0, n_dataset, batch_size):
            batch_paths = paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            X = []
            y = []
            for path,label in zip(batch_paths, batch_labels):
                img = utils.loadImage(path)
                if augment:
                    img = augmentImage(img)
                label_flat = np.array(label).flatten() #make it flat
                #print('label: ',label_flat)

                X.append(img)
                y.append(label_flat)

            X = np.array(X)
            y = np.array(y)
            yield shuffle(X,y)



def datasourceToDataset(name=None, valid_split=0.2, test_split=0.2, reindex_only=False, force=False):
    if name is None: name = 'dataset_'+utils.standardDatetime()
    basepath = datasets_path + name + '/'
    basepath_img = basepath + 'img/'

    if os.path.exists(basepath+trainfile) and not force and not reindex_only:
        return basepath


    vehicles_search_path = datasources_path + 'vehicles/*/*.png'
    nonvehicles_search_path = datasources_path + 'non-vehicles/*/*.png'
    print(vehicles_search_path,nonvehicles_search_path)

    vehicles_paths = glob.glob(vehicles_search_path)
    nonvehicles_paths = glob.glob(nonvehicles_search_path)
    print('vehicles_paths', len(vehicles_paths))
    print('nonvehicles_paths', len(nonvehicles_paths))

    paths = []
    labels = []
    for img_path in vehicles_paths:
        parts = img_path.split('/')
        filename = "_".join(parts[-2:])
        fullpath = basepath_img + filename
        if not reindex_only:
            utils.copy(img_path, fullpath)
        paths.append(fullpath)
        labels.append(1)

    for img_path in nonvehicles_paths:
        parts = img_path.split('/')
        filename = "_".join(parts[-2:])
        fullpath = basepath_img + filename
        if not reindex_only:
            utils.copy(img_path, fullpath)
        paths.append(fullpath)
        labels.append(0)

    train_paths, test_paths,  train_labels, test_labels  = train_test_split(paths,       labels,       test_size=test_split)
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=valid_split)

    train_dataset = {'path': train_paths, 'y': train_labels}
    valid_dataset = {'path': valid_paths, 'y': valid_labels}
    test_dataset  = {'path': test_paths,  'y': test_labels}

    saveDataset(basepath + trainfile, train_dataset)
    saveDataset(basepath + validfile, valid_dataset)
    saveDataset(basepath + testfile,  test_dataset)

    return basepath


def saveDataset(path, data):
    np.savez(path, **data)


def loadDataset(path):
    with np.load(path) as npzfile:
        dataset = {}
        for key in npzfile.keys():
            dataset[key] = npzfile[key].tolist()
    return dataset

def loadDatasets(names):
    dataset = None
    for name in names:
        dataset_part = loadDataset(name)
        if dataset is None:
            dataset = dataset_part
        else:
            for key in dataset_part:
                dataset[key].extends(dataset_part[key])
    return dataset

# video parsing and extraction

def processVideo(self, path, function, live=False, debug=False):
    if not live:
        strdate = '_' + utils.standardDatetime()
        output_video = video_output_path + utils.filenameAppend(path, strdate)
        video = VideoFileClip(path)
        video_clip = video.fl_image(function)
    else:
        vidcap = cv2.VideoCapture(path)
        while True:
            success, image = vidcap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            final_image = self.pipeline(image, debug=debug)
            utils.showImage(final_image)


# augmenting



def augmentImage(img):
    action_list = [randomBrightness, randomRotation ]#, randomPrespective]
    action_num = np.random.randint(0, len(action_list))
    action = action_list[action_num]
    aug_img = action(img)
    return aug_img

def randomBrightness(img, limit=0.4):
    img_new = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_new = np.array(img_new, dtype = np.float64)
    img_new[:,:,2] = img_new[:,:,2] * (np.random.uniform(low=limit, high=2-limit))
    img_new[:,:,2][img_new[:,:,2]>255] = 255 #cap values
    img_new = np.array(img_new, dtype = np.uint8)
    img_new = cv2.cvtColor(img_new,cv2.COLOR_HSV2RGB)
    return img_new



def randomRotation(img, max_rotation = 30):
    height, width, depth = img.shape
    angle = int( max_rotation*np.random.uniform(-1,1) )
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img_new = cv2.warpAffine(img, M, (width, height))
    return img_new

def randomShadows(img, max_shadows = 3, min_aplha=0.1, max_aplha=0.8, min_size=0.2, max_size=0.8 ):
    img_new = img.copy()
    height, width, depth = img_new.shape
    # print(width,height)
    shadow_num = int(max_shadows * np.random.uniform())+1
    for i in range(shadow_num):
        x = int(width * np.random.uniform())
        y = int(height * np.random.uniform())
        w2 = int( (width * np.random.uniform(min_size,max_size))/2 )
        h2 = int( (height * np.random.uniform(min_size,max_size))/2 )
        top, bottom = y - h2, y + h2
        left, right = x - w2, x + w2
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        img_new[top:bottom, left:right, :] = img_new[top:bottom, left:right, :] * np.random.uniform(min_aplha,max_aplha)
    return img_new


def randomNothing(img):
    return img
