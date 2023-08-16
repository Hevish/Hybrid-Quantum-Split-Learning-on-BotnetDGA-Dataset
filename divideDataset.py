import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from math import ceil

SEED = 61096
num_folds = 5
num_sample = 1000
# Load the dataset
dataset_path = "data_exported_1.8million.csv"
data = pd.read_csv(dataset_path, header=0, encoding='utf-8')
data_shuffled = data.sample(frac=1, random_state=42)
data_subset = data_shuffled.head(num_sample)
data_filtered = data_subset[[ 'CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', 'Entropy', 'InformationRadius', 'LabelBinary' ]].to_numpy()
kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)


for fold, (train_index, test_index) in enumerate(kf.split(data_filtered)):
    print(data_subset)
    data_train, data_test = data_filtered[train_index], data_filtered[test_index]
    print(data_train)
    print("Data Test",data_test)
    folder_name = f"folder{fold+1}"
    os.makedirs(folder_name, exist_ok=True)

    train_path = os.path.join(folder_name, "train.csv")
    np.savetxt(train_path, data_train, delimiter=",")

    test_path = os.path.join(folder_name, "test.csv")
    np.savetxt(test_path, data_test, delimiter=",")

