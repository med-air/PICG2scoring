import json
import pandas as pd
from pathlib import PurePath
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
from collections import Counter


with open('prostate_public_val.json', 'r') as f:
    js = json.load(f)

with open('outputs.txt', 'r') as f:
    pred = f.read()

pred = pred.split('\n')[:-1] # remove the last blank
# map_dict = {
#     'very low': 1,
#     'low': 2,
#     'intermediate': 3,
#     'high': 4,
#     'very high': 5
# }

def text2label(text):
    if 'very low' in text or '1' in text:
        return 1
    elif 'low' in text or '2' in text:
        return 2
    elif 'intermediate' in text or '3' in text:
        return 3
    elif 'high' in text or '4' in text:
        return 4
    elif 'very high' in text or '5' in text:
        return 5
    else:
        return 0

# pred = list(map(lambda x: int(x), pred))
pred = list(map(text2label, pred))
label = []
df = pd.read_csv('prostate/stl_record.csv')
for j in js:
    lesion_ID = PurePath(j['img']).parts[-1].split('.')[0]
    patient_ID = lesion_ID.split('Target')[0][:-5]
    target = int(lesion_ID[-1])
    d = df[df['patient_ID']==patient_ID]
    d = d[d['Target']==target]
    label.append(d['PI-RADS'].values[0])

print(f'acc:{accuracy_score(label, pred)}, avg_acc: {balanced_accuracy_score(label, pred)}')
print(f'mse: {np.mean(np.abs(np.array(label) - np.array(pred)))}')

print(f'Label counter: {Counter(label)}')
print(f'Prediction counter: {Counter(pred)}')
