import os 
from llama.llama_adapter_prostate import LLaMA_adapter
import util.misc as misc
import util.extract_adapter_from_checkpoint as extract
from PIL import Image
import cv2
import torch
import llama
import numpy as np
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "."
llama_type = '7B'
llama_ckpt_dir = os.path.join(llama_dir, llama_type)
llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)

misc.load_model(model, '/research/d1/rshr/ttzhang/multi-modal/prostate_finetune_100/checkpoint-60.pth')
model.eval()
model.to(device)
INSTRUCT = {
    "T2W":"With different likelihood of clinically significant cancer, prostate MRI images can have different appearances. Signal intensity in the lesion is visually compared to the average signal of normal prostate tissue elsewhere in the same histologic zone. If normal prostate tissue signal is low, then the lesion signal is high. If normal prostate tissue signal is high, then the lesion signal is low. The lesion signal is called an abnormal signal.\nIf the cancer likelyhood is very low, the signal will be normal.\nIf the likelihood is low, there will be linear/wedge shaped abnormal signal.\nIf the likelihood is intermediate, there will be focal (discrete and different from background), mild/moderate abnormal signal\nIf the likelihood is high, there will be focal, marked abnormal signal, which is less than 1.5 cm in greatest dimension\nIf the likelihood is very high, there will be focal, marked abnormal signal, which is larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior.",
    "non-T2W":"With different likelihood of clinically significant cancer, prostate MRI images can have different appearances.\nWhen the likelihood is very low, there will be normal appearing transition zone (rare) or a round, completely encapsulated nodule (typical nodule of benign prostatic hyperplasia).\nWhen the likelihood is low, there will be a mostly encapsulated nodule or a homogeneous circumscribed nodule without encapsulation (atypical nodule), or a homogeneous mildly hypointense area between nodules in the image.\nWhen the likelihood is intermediate, there is heterogeneous signal intensity with obscured margins.\nWhen the likelihood is high, there will be lenticular or non-circumscribed, homogeneous, moderately hypointense, which is less than 1.5 cm in greatest dimension in the image.\nWhen the likelihood is very high, there will be a lenticular or non-circumscribed, homogeneous, moderately hypointense, which is larger than 1.5 cm in greatest dimension or definite extraprostatic extension/invasive behavior."
}
INPUT = "Describe the apperance of this MRI image."
prompt = llama.format_prompt(INPUT)
# img = Image.fromarray(cv2.imread("your image"))
# img = model.clip_transform(img).unsqueeze(0).to(device)

with open('prostate_public_val.json') as f:
    data = json.load(f)

tf = torch.nn.Upsample(size=(84, 84, 98))
# tf = torch.nn.Upsample(size=(84, 98, 42))
# numpy -> tensor
# resize (resize + center crop)
# set channel_num -> 1
# normalize
avg, std = 0.48, 0.27
transform = lambda x: (tf(torch.from_numpy(x)[None, None, ...]) / 255. - avg) / std

cnter = 0
if not os.path.exists("prostate/feature_input60/"):
    os.mkdir("prostate/feature_input60/")

for data_item in tqdm(data):
    # patient_ID = "Prostate-MRI-US-Biopsy-0012"
    # adc = np.load(f'/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate/case_input/{patient_ID}/{patient_ID}_ADC_Target1.npy')
    # dwi = np.load(f'/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate/case_input/{patient_ID}/{patient_ID}_DWI_Target1.npy')
    # t2w = np.load(f'/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate/case_input/{patient_ID}/{patient_ID}_T2W_Target1.npy')
    # t2w = np.load(data_item['T2W'])
    # dwi = np.load(data_item['DWI'])
    # adc = np.load(data_item['ADC'])

    # t2w = transform(t2w).to(device)
    # dwi = transform(dwi).to(device)
    # adc = transform(adc).to(device)
    # img = torch.cat((t2w, dwi, adc), dim=0).unsqueeze(0).to(device)
    s = 'T2W' if not (cnter+1) % 3 else 'non-T2W'
    instruct = INSTRUCT[s]
    prompt = llama.format_prompt(instruct + INPUT)
    img = np.load(data_item['img'])
    print(data_item["img"])
    img = transform(img).to(device)
    img = torch.cat((img, img, img), dim=1)

    # result = model.generate(img, [prompt])[0]
    result, save_feature = model.generate(img, [prompt])
    head, tail = os.path.split(data_item['img'])
    save_head = head.replace("case_input", "feature_input60")

    if not os.path.exists(save_head):
        os.mkdir(save_head)
    feature_save_path = os.path.join(save_head, "feature_"+tail)

    # for i in range(1, len(save_feature)):
    #     aaa = torch.cat([aaa, save_feature[i]], 1)
    aaa = save_feature[0]
    print(aaa.shape)
    np.save(feature_save_path, aaa.cpu().numpy())
    # print("results1", result)
    # print("results2", result[0])

    # print(len(save_feature))




    # with open('outputs_round2.txt', 'a') as f:
    #     f.write(str(result))
    #     f.write('\n')
    # cnter += 1

    # extract.save(model,'path/to/adapter-7B.pth','BIAS') # Please end it with -llama_type.pth