import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt


def train_net_and_eval(net_class, optimizer_class, input_size, num_classes, num_epochs, batch_size, learning_rate):
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # net

    net = net_class(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(net.parameters(), lr=learning_rate)

    # Train the Model
    losses = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            model_output = net(images)
            loss = criterion(model_output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'The loss for epoch {epoch} is: {loss}')
        losses.append(float(loss))


    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28))
        model_output = net(images)
        label_predictions = torch.argmax(model_output, dim=1)
        correct += sum(label_predictions == labels)
        total += labels.size(0)

    acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)

    return losses, acc


# q1

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 1e-3


# Neural Network Model
class Net1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


losses, acc = train_net_and_eval(Net1, torch.optim.SGD, input_size, num_classes, num_epochs, batch_size, learning_rate)

plt.plot(losses, label='q1')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.text(5, 1.7, f'acc for q1: {round(float(acc), 2)}%', size='x-small')


# q2

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 500
learning_rate = 1e-3

# Neural Network Model
class Net2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out

losses, acc = train_net_and_eval(Net2, torch.optim.Adam, input_size, num_classes, num_epochs, batch_size, learning_rate)

plt.plot(losses, label='q2')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.text(5, 1.65, f'acc for q2: {round(float(acc), 2)}%', size='x-small')


# q3

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 500
learning_rate = 1e-3

# Neural Network Model
class Net3(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.relu1(out1)
        out3 = self.fc2(out2)
        return out3

losses, acc = train_net_and_eval(Net3, torch.optim.Adam, input_size, num_classes, num_epochs, batch_size, learning_rate)

plt.plot(losses, label='q3')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.text(5, 1.6, f'acc for q3: {round(float(acc), 2)}%', size='x-small')


plt.legend()
plt.show()



