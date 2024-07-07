import json
import numpy as np
from pathlib import PurePath

with open('prostate_public_train.json', 'r') as f:
    js = json.load(f)

ids = []
for j in js:
    lesion_ID = PurePath(j['DWI']).parts[-1].split('.')[0]
    patient_ID = lesion_ID.split('Target')[0][:-5]
    ids.append(patient_ID)

np.save('train_patient.npy', ids)

with open('prostate_public_val.json', 'r') as f:
    js = json.load(f)

ids = []
for j in js:
    lesion_ID = PurePath(j['DWI']).parts[-1].split('.')[0]
    patient_ID = lesion_ID.split('Target')[0][:-5]
    ids.append(patient_ID)

np.save('val_patient.npy', ids)