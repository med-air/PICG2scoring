# convert prostate_public image to the json format
# Example:
'''
[
    {
        "ADC": ,
        "DWI": ,
        "T2W": , 
        "instruction": "Prostate cancer lesion can be rated with a score indicating the likelihood of clinically significant cancer, namely PI-RADS. 
        PI-RADS ranges from 1-5 as follows: 
    PI-RADS 1: very low (clinically significant cancer is highly unlikely to be present)
    PI-RADS 2: low (clinically significant cancer is unlikely to be present)
    PI-RADS 3: intermediate (the presence of clinically significant cancer is equivocal)
    PI-RADS 4: high (clinically significant cancer is likely to be present)
    PI-RADS 5: very high (clinically significant cancer is highly likely to be present)."
  
        "input": "Please rate this lesion in PI-RADS. The answer must be chosen from {very low, low, intermediate, high, very high}.",
        "output": "very high."
    },
]
'''
import json
from pathlib import PurePath
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join

#####
INSTRUCT = "Prostate cancer lesion can be rated with a score indicating the likelihood of clinically significant cancer, namely PI-RADS. PI-RADS can be follows:\nvery low (clinically significant cancer is highly unlikely to be present)\nlow (clinically significant cancer is unlikely to be present)\nmedium (the presence of clinically significant cancer is equivocal)\nhigh (clinically significant cancer is likely to be present)\nvery high (clinically significant cancer is highly likely to be present)"
INPUT = "Please rate this lesion in PI-RADS. The answer must be chosen from {very low, low, medium, high, very high}."
OUTPUTS = {
    1: "very low",
    2: "low",
    3: "medium",
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
