import numpy as np
import os
import pandas as pd
# 引入基本的回归器
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import  mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from numpy.linalg import svd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import random
# 读取数据集
dataset  = pd.read_csv('/home/ql-b423/Desktop/TXH/RB/fit_2_3.csv')

X = np.asarray(dataset.get('X'),dtype='float32').reshape(-1,1)
Y = np.asarray(dataset.get('Y'),dtype='float32').reshape(-1,1)
Z = np.asarray(dataset.get('Z'),dtype='float32').reshape(-1,1)

X_avr = X.mean()
Y_avr = Y.mean()
Z_avr = Z.mean()

X_sub_avr = X - X_avr
Y_sub_avr = Y - Y_avr
Z_sub_avr = Z - Z_avr

A = np.concatenate((X_sub_avr,Y_sub_avr,Z_sub_avr),axis=1)

u,s,v_T = svd(A)

a = v_T[0,2]
b = v_T[1,2]
c = v_T[2,2]
d = -(a * X_avr + b * Y_avr +  c * Z_avr)
print(a,b,c,d)