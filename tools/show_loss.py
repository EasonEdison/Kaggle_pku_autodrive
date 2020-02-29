import pandas as pd
import matplotlib.pyplot as plt
history = pd.read_csv('/OP_test_branch_pitch_s32/history_3.csv')
history['train_loss'].iloc[100:].plot()
#series = history.dropna()['val_loss']
#plt.scatter(series.index, series,edgecolors='red')

# fig, axes = plt.subplots(1, 2, figsize=(15, 15))
#
# axes[0] = history['train_loss'].iloc[100:].plot()
#
# series = history.dropna()['dev_loss']
#
# axes[1].scatter(series.index, series,edgecolors='red')

plt.show()