import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters


num_epochs = 5
batch_size = 1
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
# 데이터 타입 변환, 정규화
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                              download=True, transform=transform)
train_dataset = torchvision.datasets.ImageFolder(root = "../dataset/gender/train/", transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                             download=True, transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root = "../dataset/gender/test/", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,

                                          shuffle=False)


# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('male', 'female')



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))


# CNN 네트워크 구조 짜는 CLASS
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)  # KERNEL SIZE 2, STRIDE 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)  # COLOR CHANNEL(INPUT) 3, OUTPUTCHANNEL 6, KERNEL SIZE 5
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)  # COLOR CHANNEL(INPUT) 3, OUTPUTCHANNEL 6, KERNEL SIZE 5
        self.fc1 = nn.Linear(512 * 3 * 2 * 3 * 2, 512)  # INPUT 16 * 5 * 5, OUTPUT 120
        self.fc2 = nn.Linear(512, 256)  # INPUT 120, OUTPUT 84
        self.fc3 = nn.Linear(256, 64) # INPUT 84, OUTPUT 10
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = self.pool(F.relu(self.conv3(x)))  # -> n, 16, 5, 5
        x = self.pool(F.relu(self.conv4(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 512 * 3 * 2 * 3 * 2)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = F.relu(self.fc3(x))  # -> n, 84
        x = self.fc4(x)  # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)
writer = SummaryWriter()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss, i)

        if (i + 1) % 10000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn_gender' + str(num_epochs) + '.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    writer.add_scalar('Acc/test', acc)
    print(f'Accuracy of the network: {acc} %')

    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        writer.add_scalar('Acc/class', acc)
writer.close()
