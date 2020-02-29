import csv
import pandas as pd

path = 'D:\\jieya\\Pku-cdpn\\train_data.csv'
train = pd.read_csv(path)

# 这里的应该是要原图
def str2coor(mess):
    values = mess.split(', ')
    #values = mess.split(', ')
    values[0] = values[0][1:]
    x = (float(values[0]) + float(values[2])) / 2
    y = (float(values[1]) + float(values[3])) / 2
    X = float(values[4])
    Y = float(values[5])
    Z = float(values[6])
    return x,y,X,Y,Z

header = ['ID_image','x','y','X','Y','Z']
rows = []
for i in range(len(train)):
    img_name = train['ID_image'][i]
    x,y,X,Y,Z = str2coor(train['else'][i])
    rows.append((img_name,x,y,X,Y,Z))
    print('{} / {}'.format(i,len(train)))

with open('fit_2_3.csv','w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

