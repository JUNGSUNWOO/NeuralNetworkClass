import dlib
import cv2
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# from model.tmp_CNN import ConvNet
from gender_classification.model.CNN_gender_2 import ConvNet
from gender_classification.model.CNN_age import ConvNet as Net2
gender_model_path = 'C:/Users/정선우/PycharmProjects/trained_model/GENDER/imdb_cnn20.pth'
age_model_path = 'C:/Users/정선우/PycharmProjects/trained_model/AGE/imdb_age_100.pth'
img_path = "D:/dataset/FOR_DEMO/KakaoTalk_20201229_180803051.jpg"
img_name = img_path.split('/')[3]
img_Folder = img_path.split('/')[:3]
img_Folder_path = "/".join(img_Folder) + "/result/"

device = torch.device('cpu')
gender_model = ConvNet()
age_model = Net2()
gender_model.load_state_dict(torch.load(gender_model_path, map_location=device))
age_model.load_state_dict(torch.load(age_model_path, map_location=device))

gender_classes = ('man', 'woman')
age_classes = ('0','10', '20','30','40','50','60','70','80','90')

def img_processing(img):
    resize_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)
    new_img = np.asarray(gray_img)
    dim_1 = np.expand_dims(new_img, axis=0)
    dim_2 = np.expand_dims(dim_1, axis=0)
    final_img = torch.from_numpy(dim_2)
    return final_img

def gender_classify(img):
    img = img.float()
    outputs = gender_model(img)

    result = torch.argmax(outputs)
    print("answer is ", gender_classes[result])
    return result
def age_classify(img):
    img = img.float()
    outputs = age_model(img)

    result = torch.argmax(outputs)
    print("answer is ", age_classes[result])
    return result

gender_model.eval()
age_model.eval()
#face detection
face_detector = dlib.get_frontal_face_detector()
dst = cv2.imread(img_path)
img = dst

faces = face_detector(img)
print("{} faces are detected.".format(len(faces)))
font = cv2.FONT_HERSHEY_DUPLEX  # 텍스트의 폰트를 지정.

for f in faces:
    # print(f.left(), f.right(), f.top(), f.bottom())
    print("cropping")
    crop = (img[f.top():f.bottom(),f.left():f.right()])
    cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
    final_img = img_processing(crop)
    print("put img in model")
    gender_result = gender_classify(final_img)
    age_result = age_classify(final_img)
    cv2.putText(img, "gender is " + gender_classes[gender_result], (f.left(), f.top()-50), font, 0.7, (0, 0, 255),cv2.LINE_4)
    cv2.putText(img, "age is " + age_classes[age_result], (f.left(), f.top() - 30), font, 0.7, (0, 0, 255),
                cv2.LINE_4)


    #print(model(img_face))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imwrite(img_Folder_path + img_name.split('.jpg')[0] + "_gender&age classify" + ".jpg", img)

