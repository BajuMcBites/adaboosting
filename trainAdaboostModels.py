import sys
import pickle
import numpy as np
import adaboost as ada


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def combineBatches(batches):
    labels = []
    data = []

    for batch in batches:
        data.append(batch[b'data'])
        labels.append(batch[b'labels'])

    data = np.concatenate(data, 0)
    labels = np.concatenate(labels, 0)

    return (data, labels)

def convertDataToImages(data, im_size):
    images = []

    for index in range(np.size(data, 0)):
        temp_image = np.reshape(data[index, :], [im_size[0], im_size[1], 3])
        gray_image = temp_image[:, :, 0] * 0.299 + temp_image[:, :, 1] * 0.587 + temp_image[:, :, 2] * 0.114
        images.append(adaboost.Image(gray_image))
        if ((index + 1) % 5000 == 0):
            print("Converted " + str(index + 1) + " rows to images")

    return images

def convertLabels(labels, match):
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == match:
            new_labels.append(1)
        else:
            new_labels.append(-1)
    return new_labels



if __name__ == "__main__":

    if (len(sys.argv) != 3):
        print("incorrect amount of arguments")
        exit(1)
    
    class_to_train = int(sys.argv[1])
    file_to_save = sys.argv[2]

    batch1 = unpickle("./CIFAR-10/cifar-10-batches-py/data_batch_1.pkl")
    batch2 = unpickle("./CIFAR-10/cifar-10-batches-py/data_batch_2.pkl")
    batch3 = unpickle("./CIFAR-10/cifar-10-batches-py/data_batch_3.pkl")
    batch4 = unpickle("./CIFAR-10/cifar-10-batches-py/data_batch_4.pkl")
    batch5 = unpickle("./CIFAR-10/cifar-10-batches-py/data_batch_5.pkl")

    [data,labels] = combineBatches([batch1, batch2, batch3, batch4, batch5])

    labels = convertLabels(labels, class_to_train)
    images = convertDataToImages(data, [32, 32])

    model = ada.ImAdaBoost(2000, [32, 32])
    model.train(images, labels, 30)
    model.store_model(file_to_save)

