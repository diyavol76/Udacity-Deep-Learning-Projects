import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import PIL
from PIL import Image
import matplotlib.pyplot as plt

batch_size = 20
val_fraction = 0.2
num_workers = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_data(data_main_dir):
    train_data_path = os.path.join(data_main_dir, "train")
    test_data_path = os.path.join(data_main_dir, "test")



    train_transform = transforms.Compose([transforms.RandomRotation(10),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.RandomRotation(10),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])

    all_train_data = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    return all_train_data,test_dataset

def split_load_data(all_train_data,test_dataset):

    # obtain training indices that will be used for validation
    num_train = len(all_train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(val_fraction * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(all_train_data, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(all_train_data, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              shuffle=True)

    loaders_scratch = {'train': train_loader, 'test': test_loader, 'valid': valid_loader}

    return loaders_scratch


def imshow(img):
    img = img *0.5 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    print(np.amax(img),np.amin(img))


def plot_images(train_loader,classes):

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    images = images.numpy()  # convert images to numpy for display
    max_img=np.amax(images)
    min_img=np.amin(images)
    unnormalizer=np.maximum(max_img,np.abs(min_img))
    images=images/unnormalizer
    #print(images)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(50, 10))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(4, 5, idx + 1, xticks=[], yticks=[])
        #plt.imshow(images[idx])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])



def get_class_names(train_data):

    classes = [classes_name.split(".")[1] for classes_name in train_data.classes]

    return classes


import torch.nn as nn
import torch.optim as optim




def get_optimizer_scratch(model):
    ## TODO: select and return an optimizer
    return optim.SGD(model.parameters(), lr=0.001)

if __name__ == '__main__':

    data_main_dir = r'D:\git-repos\landmark_images'
    print(torch.__version__)


    train,test=get_data(data_main_dir)
    loaders_strach=split_load_data(train,test)

    classes=get_class_names(train)

    plot_images(loaders_strach['train'],classes)
    plt.show()

    use_cuda = torch.cuda.is_available()

    ## TODO: select loss function
    criterion_scratch = nn.CrossEntropyLoss()




