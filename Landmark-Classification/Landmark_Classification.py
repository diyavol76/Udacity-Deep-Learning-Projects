import os
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

batch_size = 20
val_fraction = 0.2
num_workers = 0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
_train_data=datasets.ImageFolder

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

    _train_data = datasets.ImageFolder(root=train_data_path, transform=train_transform)

    test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)

    return _train_data,test_dataset

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


import torch.nn as nn


# define the CNN architecture
class Net(nn.Module):
    ## TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()

        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(3 * 3 * 256, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, len(classes))

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        ## Define forward behavior
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.pool(self.bn5(F.relu(self.conv5(x))))
        x = self.pool(self.bn6(F.relu(self.conv6(x))))

        x = x.view(-1, 3 * 3 * 256)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)

        return x


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        print('deneme')
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            ## TODO: find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss

        ######################
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            ## TODO: update average validation loss
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)

                valid_loss += loss

        train_loss = train_loss / len(loaders['train'])
        valid_loss = valid_loss / len(loaders['valid'])

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        ## TODO: if the validation loss has decreased, save the model at the filepath stored in save_path
        if valid_loss < valid_loss_min:
            print(f'Validation loss reduced {valid_loss_min} -> {valid_loss}. Saving Model...')
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)

    return model


def custom_weight_init(m):
    ## TODO: implement a weight initialization strategy

    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 1.0 / np.sqrt(m.in_features))
        m.bias.data.fill_(1)

if __name__ == '__main__':

    data_main_dir = r'D:\git-repos\landmark_images'
    print(torch.__version__)


    train,test=get_data(data_main_dir)
    loaders_scratch=split_load_data(train,test)

    classes=get_class_names(train)

    #images, labels = iter(loaders_scratch['train']).next()
    #classes = train.classes

    #plot_images(loaders_scratch['train'],classes)
    #plt.show()

    use_cuda = torch.cuda.is_available()

    ## TODO: select loss function
    criterion_scratch = nn.CrossEntropyLoss()

    # -#-# Do NOT modify the code below this line. #-#-#

    # instantiate the CNN
    model_scratch = Net()

    # move tensors to GPU if CUDA is available
    if use_cuda:
        model_scratch.cuda()

    # -#-# Do NOT modify the code below this line. #-#-#

    model_scratch.apply(custom_weight_init)
    model_scratch = train(5, loaders_scratch['train'], model_scratch, get_optimizer_scratch(model_scratch),criterion_scratch, use_cuda, 'ignore.pt')




