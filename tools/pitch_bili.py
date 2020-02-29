import csv
import pandas as pd
import numpy as np

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []

    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])


    return coords

train = pd.read_csv('D:\\Dataset\\pku-autonomous-driving\\train.csv')


num_npi = 0
num_ppi = 0
num_zero = 0
total_npi = 0.0
total_ppi = 0.0

# 2.6 0.1468
# +3： 3.113093639479098
# +3 bili： 0.14682795266081636
# -3： -3.0918971010525573
# -3 bili： 0.8528701392802512

for i in range(len(train['ImageId'])):
    str_mess = str2coords(train['PredictionString'][i])
    for mess in str_mess:
        pitch = mess['roll']
        if pitch > 2.6:
            num_ppi += 1
            total_ppi += pitch
        elif pitch < -2.6:
            num_npi += 1
            total_npi += pitch
        else:
            num_zero += 1
# 2.5
# +3： 3.0434538712340116
# -3： -3.053348604421911

# 2.6   0.142   0.140
# +3： 3.0561653633413353
# -3： -3.0623535235091732
#
#
# 2.7   0.139   0.139
# +3： 3.0661015890688295
# -3： -3.070516786079834
#
# 2.8 0.135 0.134
# +3： 3.0740335695226824
# -3： -3.0781245254490996
#
# 2.9   0.13    0.13
# +3： 3.0830451288779193
# -3： -3.0860652666666666

num_all = num_zero + num_npi + num_ppi
print('+3：',total_ppi / num_ppi)
print('+3 bili：',num_ppi / num_all)
print('-3：',total_npi / num_npi )
print('-3 bili：',num_npi / num_all )
print('0：',num_zero )
