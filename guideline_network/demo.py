import cv2
import llama
import torch
from PIL import Image
import torchvision.transforms as tf

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "."

# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
model, preprocess = llama.load("BIAS-7B", llama_dir, '7B', device)
model.eval()

prompt = llama.format_prompt('Please introduce this painting.')
img = Image.fromarray(cv2.imread("../docs/logo_v1.png"))
t = tf.Compose(
    [
    tf.Resize((84, 84)),
    tf.ToTensor(),
    tf.Normalize((0.26, 0.26, 0.26), (0.47, 0.47, 0.47))
    ]
)
img = t(img).unsqueeze(0).to(device)
# img = preprocess(img).unsqueeze(0).to(device) # B, C, H, W
img = img.unsqueeze(-1).repeat(1,1,1,1,98)
# img = img[:,0,...]
# img = img[:,None,...]

result = model.generate(img, [prompt])[0]

print(result)
