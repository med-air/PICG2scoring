# convert prostate_public image to the json format
# Example:
import json
from pathlib import PurePath
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from copy import deepcopy

#####
INSTRUCT = "Prostate MRI images have two modalities, T2W, and non-T2W. On non-T2W, lesion tissue appears brighter or darker than surrounding area. The boundaries of tissues are often not well discerned. While on T2W, lesion is typically hypointense which appears darker than surrounding area. Moreover, we can clearly see the lesion and all surrounding tissues, as well as the clear boundaries of the tissues."
INPUT = "What is the modality of this image? The answer must be from {T2W, non-T2W}."
# OUTPUTS = {
#     1: "very low",
#     2: "low",
#     3: "intermediate",
#     4: "high",
#     5: "very high"
# }
ROOT = join("prostate", "case_input")
#####


# read the label csv
df = pd.read_csv('prostate/stl_record.csv')
# remove 0 PI-RADS
df = df.loc[df['PI-RADS'].apply(lambda x: 1 <= x <= 5)]
patients = list(set(df['patient_ID'].values))
print(f'There are {len(df)} lesions.')
print(f"There are {len(patients)} patients.")


# train / val split
# 10% as validation 
# np.random.shuffle(patients)
# train_num = int(len(patients)*0.9)
# train_patients = patients[:train_num]
# val_patients = patients[train_num:]
train_patients = np.load('train_patient.npy')
val_patients = np.load('val_patient.npy')

train_df = df.loc[df['patient_ID'].apply(lambda x: x in train_patients)]
val_df = df.loc[df['patient_ID'].apply(lambda x: x in val_patients)]
print(f'There are {len(train_df)} training samples and {len(val_df)} validation samples.')

# prostate_public_train.json
train_json = []
for t in tqdm(range(len(train_df))):
    train_lesion_json = {"instruction":INSTRUCT, "input":INPUT}
    patient_ID = train_df['patient_ID'].values[t]
    label = train_df['PI-RADS'].values[t]
    for ch in ['ADC', 'DWI', 'T2W']:
        train_lesion_json['img'] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{train_df['Target'].values[t]}.npy"
        if ch == 'T2W':
            ch = 'T2W.'
        else:
            ch = 'non-T2W.'
        train_lesion_json['output'] = ch
        train_json.append(deepcopy(train_lesion_json))
    # train_lesion_json["output"] = OUTPUTS[label]
    # for ch in ['ADC', 'DWI', 'T2W']:
    #     train_lesion_json[ch] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{train_df['Target'].values[t]}.npy"
    # train_lesion_json['Label'] = str(label)
    # train_lesion_json['patient_ID'] = patient_ID
    # train_lesion_json['target'] = str(train_df['Target'].values[t])
    # train_json.append(train_lesion_json)

with open('prostate_public_train_image.json', 'w') as f:
    json.dump(train_json, f)


# prostate_public_val.json
val_json = []
for t in tqdm(range(len(val_df))):
    val_lesion_json = {"instruction":INSTRUCT, "input":INPUT}
    patient_ID = val_df['patient_ID'].values[t]
    label = val_df['PI-RADS'].values[t]
    # val_lesion_json["output"] = OUTPUTS[label]
    for ch in ['ADC', 'DWI', 'T2W']:
        val_lesion_json['img'] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{val_df['Target'].values[t]}.npy"
        if ch == 'T2W':
            ch = 'T2W.'
        else:
            ch = 'non-T2W.'
        val_lesion_json['output'] = ch
        val_json.append(deepcopy(val_lesion_json))
    # for ch in ['ADC', 'DWI', 'T2W']:
    #     val_lesion_json[ch] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{val_df['Target'].values[t]}.npy"
    # val_lesion_json['Label'] = str(label)
    # val_lesion_json['patient_ID'] = patient_ID
    # val_lesion_json['target'] = str(val_df['Target'].values[t])
    # val_json.append(val_lesion_json)

with open('prostate_public_val_image.json', 'w') as f:
    json.dump(val_json, f)
