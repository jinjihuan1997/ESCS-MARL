import os
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms

# === 文件路径 ===
bpg_path = "2.bpg"           # BPG 压缩图路径
png_ref_path = "2.png"       # 原始 PNG 图路径
bpg_decoded_path = "2_bpg.png"  # 解码后的 PNG 图路径

# === Step 1: 解码 BPG 到 PNG ===
# 使用 bpgdec.exe 进行解码，要求可执行文件在路径中
os.system(f"bpgdec {bpg_path} -o {bpg_decoded_path}")

# === Step 2: 初始化 LPIPS 模型 ===
loss_fn = lpips.LPIPS(net='alex')  # 也可改为 'vgg' 或 'squeeze'

# === Step 3: 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # LPIPS 模型要求尺寸一致
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# 加载图像
img_ref = transform(Image.open(png_ref_path).convert("RGB")).unsqueeze(0)
img_bpg = transform(Image.open(bpg_decoded_path).convert("RGB")).unsqueeze(0)

# === Step 4: 计算 LPIPS ===
with torch.no_grad():
    score = loss_fn(img_ref, img_bpg)

print(f"LPIPS between {png_ref_path} and decoded {bpg_path}: {score.item():.4f}")
