import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 500
learning_rate = 1e-3

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


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
losses = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        model_output = net(images)
        loss = criterion(model_output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'The loss for epoch {epoch} is: {loss}')
    losses.append(float(loss))

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    model_output = net(images)
    label_predictions = torch.argmax(model_output, dim=1)
    correct += sum(label_predictions == labels)
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
