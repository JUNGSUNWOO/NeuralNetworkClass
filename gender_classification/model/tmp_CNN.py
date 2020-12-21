import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

print("data loading...")
train_dataset = torchvision.datasets.ImageFolder(root = "../dataset/gender/tmp/", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root = "../dataset/gender/tmp_test/", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('woman', 'man')

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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 384, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(384, 128, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(384)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 14 * 14, 128 * 14)
        self.fc2 = nn.Linear(128 * 14, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        y = self.bn3(F.leaky_relu(self.conv3_1(x)))
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        x = self.pool(F.leaky_relu(x))
        x = self.pool2(F.leaky_relu(self.conv5(x)))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("newtork init")
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.75)

n_total_steps = len(train_loader)

train_acc = []
train_loss = []
train_total = 0
train_correct = 0

print('training')
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()#기울기 초기화
        loss.backward()#역전파 알고리즘 계산
        optimizer.step()#가중치 수정
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        acc = 100.0 * train_correct / train_total
        if (i + 1) % 200 == 0:
            train_acc.append(acc)
            train_loss.append(loss.item())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Acc: {acc:.4f}, Loss: {loss.item():.4f}')


print('Finished Training')
PATH = './imdb_cnn.pth'
# model.load_state_dict(torch.load(PATH))
torch.save(model.state_dict(), PATH)



model.eval()
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

        for i in range(2):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

plt.subplot(211)
plt.plot(train_loss, '-', color="r",
         label="train loss")
plt.ylim(-0.1, 1.1)
plt.grid()
plt.legend()

plt.subplot(212)
plt.plot(train_acc, '-', color="b",
         label="val acc")
plt.ylim(-0.1, 101.1)
plt.grid()
plt.legend()
plt.show()