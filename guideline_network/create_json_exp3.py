# convert prostate_public image to the json format
# Example:
import json
from pathlib import PurePath
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join

#####
INSTRUCT = "Prostate cancer lesion can be rated with a score indicating the likelihood of clinically significant cancer. very low: normal appearing on both left and right image. low: a mostly encapsulated nodule or a homogeneous circumscribed nodule without encapsulation on left image and linear/wedge shaped, hypointensity on right image. intermediate: heterogeneous on left image and focal (discrete and different from background) on right image. High : lenticular or non-circumscribed, homogeneous, moderately hypointense, and less than 1.5 cm in greatest dimensionon left image and focal, marked hypointensity on right image. very high: same as high, but larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior on both images."
INPUT = "Rate the lesion with a score from {very low, low, intermediate, high, very high}."
OUTPUTS = {
    1: "very low",
    2: "low",
    3: "intermediate",
    4: "high",
    5: "very high"
}
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
    train_lesion_json["output"] = OUTPUTS[label]
    for ch in ['ADC', 'DWI', 'T2W']:
        train_lesion_json[ch] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{train_df['Target'].values[t]}.npy"
    train_lesion_json['Label'] = str(label)
    train_lesion_json['patient_ID'] = patient_ID
    train_lesion_json['target'] = str(train_df['Target'].values[t])
    train_json.append(train_lesion_json)

with open('prostate_public_train.json', 'w') as f:
    json.dump(train_json, f)


# prostate_public_val.json
val_json = []
for t in tqdm(range(len(val_df))):
    val_lesion_json = {"instruction":INSTRUCT, "input":INPUT}
    patient_ID = val_df['patient_ID'].values[t]
    label = val_df['PI-RADS'].values[t]
    val_lesion_json["output"] = OUTPUTS[label]
    for ch in ['ADC', 'DWI', 'T2W']:
        val_lesion_json[ch] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{val_df['Target'].values[t]}.npy"
    val_lesion_json['Label'] = str(label)
    val_lesion_json['patient_ID'] = patient_ID
    val_lesion_json['target'] = str(val_df['Target'].values[t])
    val_json.append(val_lesion_json)

with open('prostate_public_val.json', 'w') as f:
    json.dump(val_json, f)
