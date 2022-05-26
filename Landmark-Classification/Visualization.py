import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img * 0.5 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    print('max-min value', np.amax(img), np.amin(img))


def plot_images(train_loader, classes):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    images = images.numpy()  # convert images to numpy for display
    max_img = np.amax(images)
    min_img = np.amin(images)
    unnormalizer = np.maximum(max_img, np.abs(min_img))
    images = images / unnormalizer
    # print(images)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(50, 30))
    # display 20 images
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        # plt.imshow(images[idx])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]], fontsize=35)



