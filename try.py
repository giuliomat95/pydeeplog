import json
import numpy as np


def get_dataset():
    filepath = 'data/dataset.json'
    with open(filepath, 'r') as fp:
        dataset = json.load(fp)['data']
    dataset = np.array(dataset)
    y_true = dataset[0:5]
    increments = [0.1, -0.1, 0.05, -0.05, 0.2]
    y_pred = [y_true[i]+increments[i] for i in range(len(increments))]
    return y_true, y_pred


y_true, y_pred = get_dataset()
print(y_true)
print(y_pred)
