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
INSTRUCT = {
    "T2W":"Prostate cancer lesion can be rated with a score indicating the likelihood of clinically significant cancer.\n1: normal appearing transition zone (rare) or a round, completely encapsulated nodule (typical nodule of benign prostatic hyperplasia)\n2: a mostly encapsulated nodule or a homogeneous circumscribed nodule without encapsulation (atypical nodule), or a homogeneous mildly hypointense area between nodules\n3: heterogeneous signal intensity with obscured margins\n4: lenticular or non-circumscribed, homogeneous, moderately hypointense, and less than 1.5 cm in greatest dimension\n5: same as 4, lenticular or non-circumscribed, homogeneous, moderately hypointense, but larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior.",
    "non-T2W":"Signal intensity in the lesion is visually compared to the average signal of normal prostate tissue elsewhere in the same histologic zone. If normal prostate tissue signal is low, then the lesion signal is high. If normal prostate tissue signal is high, then the lesion signal is low. The lesion signal is called an abnormal signal. Prostate cancer lesion can be rated with a score indicating the likelihood of clinically significant cancer.\n1: normal signal\n2: linear/wedge shaped abnormal signal.\n3: focal (discrete and different from background), mild/moderate abnormal signal\n4: focal, marked abnormal signal; less than 1.5 cm in greatest dimension\n5: focal, marked abnormal signal; larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior."
}
INPUT = "Rate the lesion with a score from {1, 2, 3, 4, 5}."
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
    train_lesion_json = {"input":INPUT}
    patient_ID = train_df['patient_ID'].values[t]
    label = train_df['PI-RADS'].values[t]
    train_lesion_json["output"] = str(label)
    for ch in ['ADC', 'DWI', 'T2W']:
        train_lesion_json['img'] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{train_df['Target'].values[t]}.npy"
        if ch == 'T2W':
            o = INSTRUCT["T2W"]
        else:
            o = INSTRUCT["non-T2W"]
        train_lesion_json["instruction"] = o
        train_json.append(deepcopy(train_lesion_json))
    # for ch in ['ADC', 'DWI', 'T2W']:
    #     train_lesion_json[ch] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{train_df['Target'].values[t]}.npy"
    # train_lesion_json['Label'] = str(label)
    # train_lesion_json['patient_ID'] = patient_ID
    # train_lesion_json['target'] = str(train_df['Target'].values[t])
    # train_json.append(train_lesion_json)

with open('prostate_public_train.json', 'w') as f:
    json.dump(train_json, f)


# prostate_public_val.json
val_json = []
for t in tqdm(range(len(val_df))):
    val_lesion_json = {"input":INPUT}
    patient_ID = val_df['patient_ID'].values[t]
    label = val_df['PI-RADS'].values[t]
    val_lesion_json["output"] = str(label)
    for ch in ['ADC', 'DWI', 'T2W']:
        val_lesion_json['img'] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{val_df['Target'].values[t]}.npy"
        if ch == 'T2W':
            o = INSTRUCT["T2W"]
        else:
            o = INSTRUCT["non-T2W"]
        val_lesion_json["instruction"] = o
        val_json.append(deepcopy(val_lesion_json))

with open('prostate_public_val.json', 'w') as f:
    json.dump(val_json, f)
