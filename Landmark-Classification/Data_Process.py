import torch
import numpy as np
import torchvision.transforms as transforms
import os
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 20
val_fraction = 0.2
num_workers = 0


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
    print(type(all_train_data))
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
                                               num_workers=num_workers
                                               )

    valid_loader = torch.utils.data.DataLoader(all_train_data, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              shuffle=True)

    loaders_scratch = {'train': train_loader, 'test': test_loader, 'valid': valid_loader}

    return loaders_scratch