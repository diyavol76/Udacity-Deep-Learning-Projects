import Data_Process
import matplotlib.pyplot as plt
import Visualization
import torch
import Network_Model
import numpy as np



def get_class_names(train_data):

    classes = [classes_name.split(".")[1] for classes_name in train_data]

    return classes

def get_optimizer_scratch(model):

    return optim.SGD(model.parameters(), lr=0.01)


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


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    print("Test started...")
    # set the module to evaluation mode
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

def default_weight_init(m):
    reset_parameters = getattr(m, 'reset_parameters', None)
    if callable(reset_parameters):
        m.reset_parameters()


def Train_the_model(num_epochs=5):
    global model_scratch

    model_scratch.apply(default_weight_init)
    # train the model
    model_scratch = train(num_epochs, loaders_scratch, model_scratch, get_optimizer_scratch(model_scratch),
                          criterion_scratch, use_cuda, 'model_scratch.pt')
    # load the model that got the best validation accuracy
    model_scratch.load_state_dict(torch.load('model_scratch.pt'))


if __name__ == '__main__':
    train_data, test_data = Data_Process.get_data(r'D:\git-repos\landmark_images')

    loaders_scratch = Data_Process.split_load_data(train_data, test_data)

    classes = get_class_names(train_data.classes)
    print("classes : ",len(classes))
    #Network_Model.Net.n_=classes

    #Visualization.plot_images(loaders_scratch['train'], classes)
    #plt.show()

    use_cuda = torch.cuda.is_available()

    print("use_cuda",use_cuda)
    print(torch.__version__)
    import torch.nn as nn
    import torch.optim as optim


    criterion_scratch = nn.CrossEntropyLoss()

    model_scratch = Network_Model.Net()

    # move tensors to GPU if CUDA is available
    if use_cuda:
        model_scratch.cuda()
        #print("Cuda OK")
    #TODO
    print(torch.version.cuda)
    Train_the_model(5)

    test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)


