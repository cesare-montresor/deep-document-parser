import glob, os, cv2
import numpy as np
import utils
import dataset as ds
import model as md

original_image = 'originale_100.png'
train_file = 'train'
vaid_file = 'valid'
test_file = 'test.npz'
dataset_name = 'default'
dataset_path = './datasets/'
#dataset_img_path = dataset_path+'img/'

points = [[0.058, 0.025], [0.963, 0.0259], [0.058, 0.973], [0.963, 0.973]]

max_zoom = 0.4
interval=0.4

markersize = 1
colors = [
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,0,255]
]

n_train = 5000
n_valid = int(n_train*0.3)
n_test = n_valid

status = 'train'  # generate, train, evaluate
model_path = None  # None = last generated

def main(status):

    if status == 'generate':
        generateDatasets(dataset_name, n_train, n_valid, n_test)
        main('train')
    elif status == 'train':
        model_name = md.train(dataset_name,batch_size=512,debug=False)
        main('evaluate')
        pass
    elif status == 'evaluate':
        model = md.loadModel(model_path)
        dataset = ds.loadDataset(dataset_path + dataset_name +'/'+ test_file)
        for path,y in zip(dataset['path'],dataset['y']):
            img = utils.loadImage(path)
            p = model.predict(img[None, :, :, :], batch_size=1)
            print(p,y)
            p=p[0]
            y_pred = [[p[0],p[1]],[p[2],p[3]],[p[4],p[5]],[p[6],p[7]]]
            img_dbg = drawPoints(img,y_pred,colors)
            utils.showImage(img_dbg)
    pass


def generateDatasets(dataset_name, n_train, n_valid, n_test):
    img = utils.loadImage(dataset_path + original_image)
    basepath = dataset_path + dataset_name + '/'
    basepath_img = basepath + 'img/'
    #training set
    print("Generating: training dataset")
    img_list, points_list = augmentImage(img, n_train, max_zoom=max_zoom, interval=interval)

    img_paths = []
    for i in range(len(img_list)):
        image = img_list[i]
        path = basepath_img + 'train_{0:04d}.png'.format(i)
        cv2.imwrite(path, image)
        img_paths.append(path)
    dataset = {
        'path': img_paths,
        'y': points_list,
    }
    ds.saveDataset(basepath + train_file, dataset)

    # validation set
    print("Generating: validation dataset")
    img_list, points_list = augmentImage(img, n_valid, max_zoom=max_zoom, interval=interval)
    img_paths = []
    for i in range(len(img_list)):
        image = img_list[i]
        path = basepath_img + 'valid_{0:04d}.png'.format(i)
        cv2.imwrite(path, image)
        img_paths.append(path)
    dataset = {
        'path': img_paths,
        'y': points_list,
    }
    ds.saveDataset(basepath + vaid_file, dataset)

    # test set
    print("Generating: test dataset")
    img_list, points_list = augmentImage(img, n_test, max_zoom=max_zoom, interval=interval)
    img_paths = []
    for i in range(len(img_list)):
        image = img_list[i]
        path = basepath_img + 'test_{0:04d}.png'.format(i)
        cv2.imwrite(path, image)
        img_paths.append(path)
    dataset = {
        'path': img_paths,
        'y': points_list,
    }
    ds.saveDataset(basepath + test_file, dataset)


    pass


def augmentImage(img, num, max_zoom=0.2, interval=0.1):
    img_list = []
    point_list = []
    for i in range(num):
        img_shift, points_shift = randomPrespective(img,points,max_zoom=max_zoom,interval=interval)

        img_list.append(img_shift)
        point_list.append(points_shift)
    return img_list, point_list

def randomPrespective(img,points,max_zoom=0.2,interval=0.1):
    shift_dirs = [
        [1, 1],
        [-1, 1],
        [1, -1],
        [-1, -1]
    ]

    w, h = img.shape[1], img.shape[0]
    points_shift = []
    for (px, py), (sx, sy) in zip(points, shift_dirs):
        zoom=np.random.uniform(0.01, max_zoom)
        seedx, seedy = np.random.uniform(zoom, zoom+interval, (2))
        px += (sx / 2 * seedx)
        py += (sy / 2 * seedy)
        points_shift.append([px, py])

    points_abs = []
    for px, py in points:
        points_abs.append((int(w * px), int(h * py)))

    points_abs_shift = []
    for px, py in points_shift:
        points_abs_shift.append((int(w * px), int(h * py)))

    transMat = cv2.getPerspectiveTransform(np.array(points_abs, np.float32), np.array(points_abs_shift, np.float32))

    shift = cv2.warpPerspective(img, transMat, (w, h) )
    return shift, points_shift

def drawPoints(img,points,colors):
    img = img.copy()
    points_abs = []
    w, h = img.shape[1], img.shape[0]
    for px, py in points:
        points_abs.append((int(w * px), int(h * py)))

    for (px, py), color in zip(points, colors):
        center = (int(px * w), int(py * h))
        tl = center[0] - 1, center[1] - 1
        br = center[0] + 1, center[1] + 1
        cv2.rectangle(img, tl, br, color, -1)
    return img



main(status)