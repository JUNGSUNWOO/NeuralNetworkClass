import dlib
import cv2
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# from model.tmp_CNN import ConvNet

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

model_path = "./model/CNN_gender_5.pth"
img_path = "./2020-11-27.jpg"

device = torch.device('cpu')
model = ConvNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



face_detector = dlib.get_frontal_face_detector()
img = cv2.imread(img_path)
img = torch.tensor(img)
# img.xtype(torch.FloatTensor)
print(model(img))
faces = face_detector(img)
outputs = []
print("{} faces are detected.".format(len(faces)))
face_crop = []
for f in faces:
    # print(f.left(), f.right(), f.top(), f.bottom())
    crop = (img[f.top():f.bottom(),f.left():f.right()])
    cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
    cv2.imshow("img", crop)
    cv2.waitKey()
    face_crop.append(crop)
    #print(model(img_face))
for face in face_crop:
    print(model(face))
# cv2.imshow("img", img)
cv2.imwrite("output.jpg", img)
