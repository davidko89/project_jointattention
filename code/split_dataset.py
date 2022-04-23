#%%
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_PATH = Path(__file__).parents[1]

dataset = pd.read_csv(Path(PROJECT_PATH, "data/ija_video_file_with_dx.csv")).dropna()
print(dataset)


dataset.columns = ["file_name", "label"]

dataset["id"] = dataset["file_name"].apply(lambda x: x.split("_")[0])

result = dataset[["id", "label"]].drop_duplicates().reset_index(drop=True)
print(result)


id = result["id"]
label = result["label"]

# split1
split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
for train_index, test_valid_index in split.split(id, label):
    train_set = result.iloc[train_index]
    test_valid_set = result.iloc[test_valid_index]

split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for test_index, valid_index in split2.split(test_valid_set.id, test_valid_set.label):
    test_set = test_valid_set.iloc[test_index]
    valid_set = test_valid_set.iloc[valid_index]

# print(train_set.shape, test_set.shape, valid_set.shape)


train_list = train_set["id"].to_list()
test_list = test_set["id"].to_list()
valid_list = valid_set["id"].to_list()


#%%
dataset["train"] = True
dataset["train"] = dataset["id"].apply(lambda x: x in train_list)

dataset["test"] = True
dataset["test"] = dataset["id"].apply(lambda x: x in test_list)

dataset["valid"] = True
dataset["valid"] = dataset["id"].apply(lambda x: x in valid_list)

print(dataset)


#%%
dataset.to_csv(Path(PROJECT_PATH, "data", "ija_diagnosis_sets.csv"), index=False)
