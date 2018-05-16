import h5py, os
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy.io import loadmat
from random import shuffle
import cv2

def loadFromH5(filename, keys):
    result = []
    h5file = h5py.File(filename, 'r')
    for key in keys:
        result.append(h5file[key].value)
    h5file.close()
    return result

def buileToH5(filename, datas, overWrite = True):
    if overWrite and os.path.exists(filename):
        os.system('del ' + filename)
    h5file = h5py.File(filename, 'w')
    for key, data in datas.items():
        h5file.create_dataset(key, data=data)
    h5file.close()

def imgshow_h5(filename):
    data, fingers = loadFromH5(filename, ['whole_hand', 'every_finger'])
    print(data.shape)
    print(fingers.shape)
    img = data[0,:,:,0]
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    for i in range(6):
        plt.imshow(fingers[0,:,:,i], cmap=plt.cm.gray)
        plt.show()

def imgshow_mat(filename):
    data = loadmat(filename)
    img = data['segmap']
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    
def find_max(img):
    length = img.shape[1]
    index = np.argmax(img)
    return index // length, index % length

def existNan(data):
    return np.any(np.isnan(data))

def norm(A):
    max_val = np.max(A)
    min_val = np.min(A)
    return (A - min_val) / (max_val - min_val)

def build_dataset(input_folder, output_folder, data_name, *, maxExamples = 2000, img_size = (48, 48), kernal_size = (7, 7)):
    data_path = os.path.join(input_folder, data_name)
    output_path = os.path.join(output_folder, data_name)
    if not os.path.exists(output_path):
        print('Creat folder ' + data_name)
        os.mkdir(output_path)

    l = os.listdir(data_path)
    file_num = len(l)

    for i in range(file_num // maxExamples + int(file_num % maxExamples != 0)):
        filename = os.path.join(output_path, data_name + str(i) + '.h5')
        if os.path.exists(filename):
            print('dataset ' + filename + ' already exist')
            continue
        print('building data set ' + filename)
        sub_l = l[i * maxExamples : min(file_num, (i + 1) * maxExamples)]
        m = len(sub_l)
        cot = 0

        whole_hand = []
        every_finger = []

        for j in range(m):
            print('Processing\t%d\t/\t%d' % (j + 1, m), end='\r')
            data = loadmat(os.path.join(data_path, sub_l[j]))
            segmap = data['segmap']
            depth = data['depth']
            bbox = data['bbox']
            hmap = data['hmap']

            if existNan(segmap) or existNan(bbox) or existNan(hmap):
                print('No. %d fail because of nan' %(j + 1))
                continue

            bb_index = [int(bbox[0,1]) - 1, int(bbox[0,1] + bbox[0,3]) - 1, int(bbox[0,0]) - 1, int(bbox[0,0] + bbox[0,2]) - 1]
            _depth = depth * (segmap / 255)
            sub_depth = _depth[bb_index[0] : bb_index[1], bb_index[2] : bb_index[3]]
            try:
                sub_depth = cv2.resize(sub_depth, img_size)
            except:
                print('No. %d fail because of resize' %(j + 1), sub_depth.shape)
                continue
            sub_depth = norm(sub_depth)
            sub_depth = sub_depth[:,:,np.newaxis]

            hmap_list = []
            something_wrong = False
            for k in range(6):
                img = hmap[:,:,k]
                img = img[bb_index[0] : bb_index[1], bb_index[2] : bb_index[3]]
                try:
                    img = cv2.resize(img, img_size)
                except:
                    something_wrong = True
                    break
                
                tem = np.zeros(img_size)
                tem[np.unravel_index(np.argmax(img), img.shape)] = 1
                tem = norm(cv2.GaussianBlur(tem, kernal_size, 0)) #kernal is chageable
                hmap_list.append(tem[:,:,np.newaxis])

            if something_wrong:
                print('No. %d fail because of resize' % (j + 1), img.shape)
                continue
            sub_hmap = np.concatenate(hmap_list, axis=2)

            if existNan(sub_depth) or existNan(sub_hmap):
                print('No. %d fail because of nan after resize' %(j + 1))
                continue

            whole_hand.append(sub_depth[np.newaxis])
            every_finger.append(sub_hmap[np.newaxis])

            cot += 1

        whole_hand = np.concatenate(whole_hand, axis=0)
        every_finger = np.concatenate(every_finger, axis=0)

        dicts = {'num' : cot, 'whole_hand' : whole_hand, 'every_finger' : every_finger}
        buileToH5(filename, dicts)
        print('finish ' + filename + ' with %d in %d' % (cot, m))  

def batch_data(path, batch_size = 32):
    l = os.listdir(path)
    shuffle(l)
    for name in l:
        print('in file ' + name)
        num, img, label = loadFromH5(os.path.join(path, name), ['num', 'whole_hand', 'every_finger'])
        
        index = list(range(num))
        shuffle(index)

        img = img[index]
        label = label[index]

        for i in range(num // batch_size + int(num % batch_size != 0)):
            head = i * batch_size
            rear = min(num, (i + 1) * batch_size)
            yield img[head : rear], label[head : rear]

def smoothFilter(data, n = 30):
    result = []
    sum = 0
    for i in range(data.shape[0]):
        if (i < n):
            sum += data[i]
            result.append(sum / (i + 1))
        else:
            sum += data[i]
            sum -= data[i-n]
            result.append(sum / n)
    return np.array(result)