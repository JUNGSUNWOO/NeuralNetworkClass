import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os
from pathlib import Path
from gender_classification.metrics import TimeMeter, MeanValue, Accuracy
from gender_classification.model.CNN_gender_2 import ConvNet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 20
batch_size = 8
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
train_dataset = torchvision.datasets.ImageFolder(root = "D:/dataset/dataset/gender/tmp", transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root = "D:/dataset/dataset/gender/tmp_test", transform=transform)
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


print("newtork init")
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

#make log directory
path_logdir = "./log_" + str(num_epochs)
logdir = Path(path_logdir)
if not logdir.exists():
    os.makedirs(str(logdir))

global_step = 0
start_epoch = 0
#

#
n_total_steps = len(train_loader)


print('training')
for epoch in range(num_epochs):
    print('-------epoch: {:d}-------'.format(epoch))
    model.train()
    mean_loss, acc = MeanValue(), Accuracy()
    tm = TimeMeter()
    tm.start()
    train_log = {}
    for i, (images, labels) in enumerate(train_loader):
        tm.add_counter()
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        optimizer.zero_grad()#기울기 초기화
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        loss.backward()#역전파 알고리즘 계산
        optimizer.step()#가중치 수정
        global_step += 1
        _, predicted = torch.max(outputs, 1)

        mean_loss.add(loss.detach().cpu().numpy())
        acc.add(predicted.detach().cpu().numpy(), labels.detach().cpu().numpy())
        if i % 200 == 0:
            torch.cuda.synchronize()
            tm.stop()
            print('step: {:d}/{:d}, loss: {:.4f}, acc: {:.2%}'
                  .format(i, n_total_steps, mean_loss.get(), acc.get()))
            train_log[global_step] = {
                'loss': mean_loss.get(), 'acc': acc.get()}
            tm.reset()
            tm.start()
            mean_loss.reset()
            acc.reset()

    with torch.no_grad():
        model.eval()
        mean_loss, acc = MeanValue(), Accuracy()
        n_class_correct = [0 for i in range(2)]
        n_class_samples = [0 for i in range(2)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            mean_loss.add(loss.detach().cpu().numpy())
            acc.add(predicted.detach().cpu().numpy(), labels.detach().cpu().numpy())

            val_log = {global_step: {'loss': mean_loss.get(), 'acc': acc.get()}}
            for i in range(2):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        print('val_loss: {:.4f}, val_acc: {:.2%}'.format(
            mean_loss.get(), acc.get()))
        for i in range(2):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
        # save checkpoint
        vars_to_saver = {
            'net': model.state_dict(), 'optim': optimizer.state_dict(),
            'epoch': epoch, 'global_step': global_step}
        cpt_file = logdir / 'checkpoint_{:d}.pk'.format(epoch)
        torch.save(vars_to_saver, str(cpt_file))

        log_file = logdir / 'log_{:d}.pk'.format(epoch)
        torch.save({'train': train_log, 'val': val_log}, str(log_file))

print('Finished Training')
PATH = 'C:/Users/정선우/PycharmProjects/trained_model/GENDER/imdb_gender_' + str(num_epochs) + '.pth'

torch.save(model.state_dict(), PATH)



# parser.add_argument('--restore', default='', help='snapshot path')
#
# if args.restore:
#     state = torch.load(args.restore)
#     start_epoch = state['epoch'] + 1
#     global_step = state['global_step']
#     model.load_state_dict(state['model'])
#     optimizer.load_state_dict(state['optimizer'])