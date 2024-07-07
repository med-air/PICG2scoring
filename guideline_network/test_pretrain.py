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

misc.load_model(model, '/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate_pretrain/checkpoint-19.pth')
model.eval()
model.to(device)
INSTRUCT = "Prostate MRI images have two modalities, T2W, and non-T2W. On non-T2W, lesion tissue appears brighter or darker than surrounding area. The boundaries of tissues are often not well discerned. While on T2W, lesion is typically hypointense which appears darker than surrounding area. Moreover, we can clearly see the lesion and all surrounding tissues, as well as the clear boundaries of the tissues."
INPUT = "What is the modality of this image? The answer must be from {T2W, non-T2W}."
prompt = llama.format_prompt(INSTRUCT + INPUT)
# img = Image.fromarray(cv2.imread("your image"))
# img = model.clip_transform(img).unsqueeze(0).to(device)

with open('prostate_public_val_image.json') as f:
    data = json.load(f)

tf = torch.nn.Upsample(size=(84, 84, 98))
# tf = torch.nn.Upsample(size=(84, 98, 42))
# numpy -> tensor
# resize (resize + center crop)
# set channel_num -> 1
# normalize
avg, std = 0.48, 0.27
transform = lambda x: (tf(torch.from_numpy(x)[None, None, ...]) / 255. - avg) / std

for data_item in tqdm(data):
    # patient_ID = "Prostate-MRI-US-Biopsy-0012"
    # adc = np.load(f'/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate/case_input/{patient_ID}/{patient_ID}_ADC_Target1.npy')
    # dwi = np.load(f'/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate/case_input/{patient_ID}/{patient_ID}_DWI_Target1.npy')
    # t2w = np.load(f'/zhome/28/e/143966/ssr/LLaMA-Adapter/llama_adapter_v2_multimodal7b/prostate/case_input/{patient_ID}/{patient_ID}_T2W_Target1.npy')
    # t2w = np.load(data_item['T2W'])
    # dwi = np.load(data_item['DWI'])
    # adc = np.load(data_item['ADC'])

    img = np.load(data_item['img'])
    img = transform(img).repeat(1,3,1,1,1).to(device)

    # t2w = transform(t2w).to(device)
    # dwi = transform(dwi).to(device)
    # adc = transform(adc).to(device)
    # img = torch.cat((t2w, dwi, adc), dim=0).unsqueeze(0).to(device)

    result = model.generate(img, [prompt])[0]
    # print(result)
    with open('pretrain_outputs.txt', 'a') as f:
        f.write(result)
        f.write('\n')

    # extract.save(model,'path/to/adapter-7B.pth','BIAS') # Please end it with -llama_type.pth