# Incorporating Clinical Guidelines through Adapting Multi-modal Large Language Model for Prostate Cancer PI-RADS Scoring

This is the PyTorch implemention of our [paper](https://arxiv.org/pdf/2405.08786) Incorporating Clinical Guidelines through Adapting Multi-modal Large Language Model for Prostate Cancer PI-RADS Scoring by Tiantian Zhang1, Manxi Lin2, Hongda Guo, Xiaofan Zhang, Ka Fung
Peter Chiu, Aasa Feragen, and Qi Dou

## Abstract
The Prostate Imaging Reporting and Data System (PI-RADS) is pivotal in the diagnosis of clinically significant prostate cancer through MRI imaging. Current deep learning-based PI-RADS scoring methods often lack the incorporation of common PI-RADS clinical guideline (PICG) utilized by radiologists, potentially compromising scoring accuracy. This paper introduces a novel approach that adapts a multi-modal large language model (MLLM) to incorporate PICG into PI-RADS scoring model without additional annotations and network parameters.We present a designed two-stage fine-tuning process aiming at adapting a MLLM originally trained on natural images to the MRI images while effectively integrating the PICG. Specifically, in the first stage, we develop a domain adapter layer tailored for processing 3D MRI inputs and instruct the MLLM to differentiate MRI sequences. In the second stage, we translate PICG for guiding instructions from the model to generate PICG-guided image features. Through such a feature distillation step, we align the scoring network's features with the PICG-guided image features, which enables the model to effectively incorporate the PICG information. We develop our model on a public dataset and evaluate it on an in-house dataset. Experimental results demonstrate that our approach effectively improves the performance of current scoring networks.

## Setting

We follow the enveriment setting as [LLaMA-adapter v2](https://github.com/OpenGVLab/LLaMA-Adapter)


```python
conda create -n picg -y python=3.8
conda activate picg

# install pytorch
conda install pytorch cudatoolkit -c pytorch -y

# install dependency and llama-adapter
pip install -r requirements.txt
pip install -e .
```

## Dataset

We use the public dataset from [here](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/). Please download `Images`, `Target Data` and `STL Files`. `Target Data` is one `XLSX` file which is named as `Target Data_2019-12-05.xlsx`. 

The case we used and the train/val split can be found [here](https://gocuhk-my.sharepoint.com/:f:/g/personal/tiantianzhang_cuhk_edu_hk/EiRr7xgyS4NEmJmfA2wxFgMBNCCus_B3WX6t4YKbpmRVeA?e=p85D80). Please download all the files, including the stl_record.csv. Create one file named `prostate` and put the `stl_record.csv` under path `prostate`. 

Note that some cases with multiple MRI scans are excluded from our analysis because only one set of lesion mask labels is available, making it impossible to match them correctly. Additionally, we excluded cases with a PI-RADS score of 0. Make sure you have downloaded all the files from the website, including the csv files. We use the STL files to find the lesion and cut the lesion and surrounding tissues. 

The `STLs` folder contains the segmentation label for each lesion. Some of these are segmentation results from ultrasound images. Please refer to the `Target Data_2019-12-05.xlsx` to determine which segmentation files correspond to MRI. After obtaining the mask for each lesion, we draw the bounding box of the mask and expand the box to twice its original size (with the mask in the center of the box). We then extract the region within the box on T2W, ADC, and DWI modality images. We will save the cropped images in the `nii.gz` format, with each case stored in a separate folder, such as `Prostate-MRI-US-Biopsy-0001`:

```
Prostate-MRI-US-Biopsy-0001
├── Prostate-MRI-US-Biopsy-0001_ADC_Target1.nii.gz
├── Prostate-MRI-US-Biopsy-0001_DWI_Target1.nii.gz
└── Prostate-MRI-US-Biopsy-0001_T2W_Target1.nii.gz

```
All cases need to be placed under the `case_input` path, or a path of your own choosing; for example:

```
case_input
├── Prostate-MRI-US-Biopsy-0001
│   ├── Prostate-MRI-US-Biopsy-0001_ADC_Target1.nii.gz
│   ├── Prostate-MRI-US-Biopsy-0001_DWI_Target1.nii.gz
│   └── Prostate-MRI-US-Biopsy-0001_T2W_Target1.nii.gz
├── Prostate-MRI-US-Biopsy-0003
│   ├── Prostate-MRI-US-Biopsy-0003_ADC_Target1.nii.gz
│   ├── Prostate-MRI-US-Biopsy-0003_DWI_Target1.nii.gz
│   └── Prostate-MRI-US-Biopsy-0003_T2W_Target1.nii.gz
├── Prostate-MRI-US-Biopsy-0005
│   ├── Prostate-MRI-US-Biopsy-0005_ADC_Target1.nii.gz
│   ├── Prostate-MRI-US-Biopsy-0005_DWI_Target1.nii.gz
│   └── Prostate-MRI-US-Biopsy-0005_T2W_Target1.nii.gz
├── Prostate-MRI-US-Biopsy-0006
│   ├── Prostate-MRI-US-Biopsy-0006_ADC_Target1.nii.gz
│   ├── Prostate-MRI-US-Biopsy-0006_DWI_Target1.nii.gz
│   └── Prostate-MRI-US-Biopsy-0006_T2W_Target1.nii.gz
...

```

## Create instruction
Next, we need to create the instruction. Please check the paths in the `create_json_pretrain.py` and `create_json.py` files. Ensure the path to the `stl_record.csv` file is set to `prostate/stl_record.csv`, and place the lesion images under the `case_input` path, or choose a path of your own.
```
# step one
python create_json_pretrain.py
# step two
python create_json.py

```

### Train step one：
please make sure you have uncommented the first layer conv of clip(line 108-110 in guideline_network/llama/llama_adapter_prostate.py)
```python
CUDA_VISIBLE_DEVICES=0 python finetune_3ch.py --data_config pretrain.yaml --batch_size 3 --epochs 20 --warmup_epochs 2 --blr 10e-4 --weight_decay 0.02 --llama_path . --output_dir prostate_pretrain --pretrained_path ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth

```
Then freeze the domain adapter layer (comment line 108-110 in guideline_network/llama/llama_adapter_prostate.py)
### Train step two：
```python
CUDA_VISIBLE_DEVICES=0 python finetune_3ch.py --data_config finetune.yaml --batch_size 3 --epochs 60 --warmup_epochs 5 --blr 10e-4 --weight_decay 0.02 --llama_path . --output_dir prostate_finetune --pretrained_path prostate_pretrain/checkpoint-19.pth

```
### get the feature for distillation
```python
#change the path in test_round2_getfeature.py
python test_round2_getfeature.py

```

#### Scoring Network (you can choose your own scoring network, we just show one example here.)

```python 
cd scoring_network
python main.py --root_path {yours} --root_test_path {yours} --used_dataset public  --batch_size 16 --n_threads 0  --loss_select multiFocal  --focalweight  2.0 2.0 1.0 1.0 1.0 --focalgamma  2   --pretrain_path /your/pretrain/r3d50_KM_200ep.pth --model_depth 50 --result_path new_feature_public_resnet50_datasample_distill_5e5_kd4 --n_classes 5 --n_epochs  200  --sample_size 256  --sample_duration 20 --train_txt_file ./data/public_train.txt --test_txt_file ./data/public_test.txt --inf_txt_file ./data/public_test.txt  --datasampler True --learning_rate 5e-5 --optimizer adam --loss_weight 0.4

```
## Contact
If you have any questions, please feel free to leave issues here, or contact [tiantianzhang](tiantianzhang@cuhk.edu.hk).

## Citation
```
@article{zhang2024incorporating,
  title={Incorporating Clinical Guidelines through Adapting Multi-modal Large Language Model for Prostate Cancer PI-RADS Scoring},
  author={Zhang, Tiantian and Lin, Manxi and Guo, Hongda and Zhang, Xiaofan and Chiu, Ka Fung Peter and Feragen, Aasa and Dou, Qi},
  journal={arXiv preprint arXiv:2405.08786},
  year={2024}
}

```
