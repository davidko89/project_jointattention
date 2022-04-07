# train:test = 8:2
#%%
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

PROJECT_PATH= Path(__file__).parents[1]
print(PROJECT_PATH)

#%%
dataset = pd.read_csv(Path(PROJECT_PATH, 'data/ija_video_file_with_label.csv')).dropna()
print(dataset)

#%%
dataset.columns = ['filename', 'label']

dataset['id'] = dataset['filename'].apply(lambda x:x.split('_')[0])

result = dataset[['id', 'label']].drop_duplicates().reset_index(drop = True)
print(result)

#%%
id = result['id']
label = result['label']

print(id)
print(label)

#%%
x_train, x_valid, y_train, y_valid = train_test_split(id, label, test_size=0.2, shuffle=True, stratify=label, random_state=34)

print(x_train)
print(x_valid)

#%%
x_train_list = x_train.to_list()
dataset['train'] = False
dataset['train'] = dataset['id'].apply(lambda x:x in x_train_list)

dataset.to_csv(Path(PROJECT_PATH, 'data', "ija_label_train.csv"), index = False)
# X = result.loc[result.label == 1.0, 'id'].to_numpy()
# print(np.arange(len(X)))
# print(np.random.shuffle(np.arange(len(ex))))
# int(10 * 0.8)
# arr[:8]
# arr[8:]
# %%
