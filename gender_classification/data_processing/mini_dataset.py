import os
import shutil
from pathlib import Path

train_path = 'D:/dataset/dataset/age/test/'
target_path = 'D:/dataset/mini_dataset/age/test/'
file = os.listdir(train_path)
for f in file:
    file_list = os.listdir(train_path+f)
    print(file_list)
    tmp_path = Path(target_path + f)
    if not tmp_path.exists():
        os.makedirs(tmp_path)
    i = 0
    for fl in file_list:
        print(fl)
        i += 1
        shutil.copy(train_path + f +'/'+ fl, target_path + f +'/'+ fl)
        if i >= 50: break