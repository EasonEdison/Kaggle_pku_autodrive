
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pylab as plt
from keras.preprocessing.image import load_img
import os
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# 将那些mask位置都变成纯黑了
# 可以用这些处理完的图片来做检测
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        True

path = 'D:\\Dataset\\pku-autonomous-driving'
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(os.path.join(path,'train.csv'))

train_imagesfolder = os.listdir(os.path.join(path,"train_images"))  # dir is your directory path
trainimagesfilecount = len(train_imagesfolder)

train_masksfolder = os.listdir(os.path.join(path,"train_masks"))  # dir is your directory path
trainmasksfilecount = len(train_imagesfolder)

#traindata = pd.read_csv(stringpath + r"train.csv")

stringpath = 'D:\\Dataset\\pku-autonomous-driving'
def CreateMaskImages(imageName):
    trainimage = cv2.imread(stringpath + "\\train_images\\" + imageName + '.jpg')
    imagemask = cv2.imread(stringpath + "\\train_masks\\" + imageName + ".jpg", 0)
    try:
        imagemaskinv = cv2.bitwise_not(imagemask)
        res = cv2.bitwise_and(trainimage, trainimage, mask=imagemaskinv)
        #plt.imshow(imagemask)
        res = cv2.resize(res,(640,480))

        cv2.imwrite("D:\\Dataset\\pku-autonomous-driving\\MaskTrain\\" + imageName + ".jpg", res)

    except:
        print("exception for image" + imageName)
        cv2.imwrite("D:\\Dataset\\pku-autonomous-driving\\MaskTrain\\" + imageName + ".jpg", trainimage)
        print('b')


for i in range(len(train_data)):
    ImageName = train_data.loc[i, "ImageId"]
    print('{} / {}'.format(i,len(train_data)))
    #print(ImageName)
    CreateMaskImages(ImageName)