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
    "T2W":"With different likelihood of clinically significant cancer, prostate MRI images can have different appearances.\nWhen the likelihood is very low, there will be normal appearing transition zone (rare) or a round, completely encapsulated nodule (typical nodule of benign prostatic hyperplasia).\nWhen the likelihood is low, there will be a mostly encapsulated nodule or a homogeneous circumscribed nodule without encapsulation (atypical nodule), or a homogeneous mildly hypointense area between nodules in the image.\nWhen the likelihood is intermediate, there is heterogeneous signal intensity with obscured margins.\nWhen the likelihood is high, there will be lenticular or non-circumscribed, homogeneous, moderately hypointense, which is less than 1.5 cm in greatest dimension in the image.\nWhen the likelihood is very high, there will be a lenticular or non-circumscribed, homogeneous, moderately hypointense, which is larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior.",
    "non-T2W":"With different likelihood of clinically significant cancer, prostate MRI images can have different appearances. Signal intensity in the lesion is visually compared to the average signal of normal prostate tissue elsewhere in the same histologic zone. If normal prostate tissue signal is low, then the lesion signal is high. If normal prostate tissue signal is high, then the lesion signal is low. The lesion signal is called an abnormal signal.\nIf the cancer likelyhood is very low, the signal will be normal.\nIf the likelihood is low, there will be linear/wedge shaped abnormal signal.\nIf the likelihood is intermediate, there will be focal (discrete and different from background), mild/moderate abnormal signal\nIf the likelihood is high, there will be focal, marked abnormal signal, which is less than 1.5 cm in greatest dimension\nIf the likelihood is very high, there will be focal, marked abnormal signal, which is larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior."
}
INPUT = "Describe the apperance of this MRI image."
OUTPUTS = {
    "T2W":{
    1: "There is a normal appearing transition zone (rare) or a round, completely encapsulated nodule (typical nodule of benign prostatic hyperplasia) in the image.",
    2: "There is a mostly encapsulated nodule or a homogeneous circumscribed nodule without encapsulation (atypical nodule), or a homogeneous mildly hypointense area between nodules in the image.",
    3: "There is heterogeneous signal intensity with obscured margins in the images.",
    4: "There is a lenticular or non-circumscribed, homogeneous, moderately hypointense, which is less than 1.5 cm in greatest dimension in the image.",
    5: "There is a lenticular or non-circumscribed, homogeneous, moderately hypointense, which is larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior in the image."
    }
    ,
    "non-T2W":{
    1: "The signal is normal.",
    2: "The signal is linear/wedge shaped abnormal.",
    3: "The signal is focal (discrete and different from background) and mild/moderate abnormal.",
    4: "The signal is focal and marked abnormal, which is less than 1.5 cm in greatest dimension.",
    5: "The signal is focal and marked abnormal, which is larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior"
    }
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
    train_lesion_json = {"input":INPUT}
    patient_ID = train_df['patient_ID'].values[t]
    label = train_df['PI-RADS'].values[t]
    for ch in ['ADC', 'DWI', 'T2W']:
        train_lesion_json['img'] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{train_df['Target'].values[t]}.npy"
        if ch == 'T2W':
            o = "T2W"
        else:
            o = "non-T2W"
        train_lesion_json["instruction"] = INSTRUCT[o]
        train_lesion_json["output"] = OUTPUTS[o][label]
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
    for ch in ['ADC', 'DWI', 'T2W']:
        val_lesion_json['img'] = f"{join(ROOT, patient_ID, patient_ID)}_{ch}_Target{val_df['Target'].values[t]}.npy"
        if ch == 'T2W':
            o = "T2W"
        else:
            o = "non-T2W"
        val_lesion_json["instruction"] = INSTRUCT[o]
        val_lesion_json["output"] = OUTPUTS[o][label]
        val_json.append(deepcopy(val_lesion_json))

with open('prostate_public_val.json', 'w') as f:
    json.dump(val_json, f)
