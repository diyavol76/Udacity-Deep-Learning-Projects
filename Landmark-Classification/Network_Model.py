import torch.nn.functional as F
import torch.nn as nn

#n_classes =len(classes)

#self.n_classes =None
# define the CNN architecture
class Net(nn.Module):
    ## TODO: choose an architecture, and complete the class

    def __init__(self):
        super(Net, self).__init__()

        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(14 * 14 * 32, 256)
        self.fc2 = nn.Linear(256, 50) # n_classes = 50

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)

        x = x.view(-1, 32 * 14 * 14)
        # add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        x = self.fc2(x)

        return x