# LLaMA-Adapter-V2 Multi-modal



# pre-train to classify modalities
CUDA_VISIBLE_DEVICES=0 python finetune_3ch.py --data_config pretrain.yaml --batch_size 3 --epochs 20 --warmup_epochs 2 --blr 10e-4 --weight_decay 0.02 --llama_path . --output_dir prostate_pretrain --pretrained_path ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth 

# fine-tuning to PI-RADS
CUDA_VISIBLE_DEVICES=0 python finetune_3ch.py --data_config finetune.yaml --batch_size 3 --epochs 100 --warmup_epochs 5 --blr 10e-4 --weight_decay 0.02 --llama_path . --output_dir prostate_pretrain --pretrained_path prostate_pretrain/checkpoint-19.pth
