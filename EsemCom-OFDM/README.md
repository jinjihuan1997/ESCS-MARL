
# Channel-aware GAN Inversion for Semantic Communication

This repository contains the implementation of a channel-aware GAN inversion method designed to extract meaningful semantic information from original inputs and map it into a channel-correlated latent space, which can eliminates the necessity for additional channel encoder and decoder.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (>=3.6)
- PyTorch
- NumPy
- Pillow
- imageio
- tqdm
- lpips
- piq
- pytorch-msssim

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/recusant7/GAN_SeCom.git
cd GAN_SeCom
```

2. Download the pre-trained model checkpoint:

```bash
wget https://github.com/seasonSH/SemanticStyleGAN/releases/download/1.0.0/CelebAMask-HQ-512x512.pt -O pretrained/CelebAMask-HQ-512x512.pt
```

## Running the Code

Run the inversion code with the following command:

```bash
python main.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --outdir results/inversion --dataset ./data/examples --size 512 --batch_size 8 --snr_db 15
```
```bash
python main_improved.py --step 100 --snr_db 15 --channel_mode awgn --batch_size 8 --num_images 16
python main_improved.py --step 100 --snr_db 15 --channel_mode awgn_rayleigh --batch_size 8 --num_images 16
python main_improved.py --step 100 --snr_db 15 --channel_mode awgn_shadow --batch_size 8 --num_images 16
python main_improved.py --step 100 --snr_db 15 --channel_mode awgn_rayleigh_shadow --batch_size 8 --num_images 16

```

Make sure to adjust the arguments based on your requirements. You can find a description of the available arguments in the script.

## Results
![Reconstructed Images](results/vis.png)
The results, including reconstructed images and log files, will be saved in the specified output directory (`--outdir`). Check the log files for average PSNR, MS-SSIM, and LPIPS.

## Acknowledgments

- This code is based on [SemanticGAN](https://github.com/nv-tlabs/semanticGAN_code) and [SemanticStyleGAN](https://github.com/seasonSH/SemanticStyleGAN/tree/main). We extend our sincere thanks to the authors of these projects for their valuable works.


python A.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/examples --latent_dir ./latents --snr_db 15 --step 300 --batch_size 8
python B.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --latent_dir ./latents --out_dir ./reconstructions


python A.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --data ./data/examples --outdir ./latents --steps 300 --lr 0.1 --batch 1 --size 512 --snr_db 15
python B.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --latents ./latents --orig ./data/examples --outdir results/recon --snr_db 15 --bs 1 --workers 4 --verbosity 1


gpustat 是一个轻量级的 Python 脚本，依赖 nvidia-smi，可以更紧凑地显示每块卡的使用信息。
运行指令：
    gpustat 
    watch -n 1 gpustat

nvitop

使用 nvidia-smi --query-gpu + 循环刷新
在 PowerShell 中，可以用 -l（loop）参数让 nvidia-smi 每秒自动刷新一次。以下示例把常见的几乎所有指标都列出来：
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,fan.speed,utilization.encoder,utilization.decoder,clocks.gr,clocks.mem --format=csv -l 1
--query-gpu=... 后面逗号分隔的字段是你想查看的指标，可以按需增删。

常用字段说明：
index：GPU 序号
name：GPU 型号
temperature.gpu：GPU 核心温度 (°C)
utilization.gpu：GPU 计算利用率 (%)
utilization.memory：显存利用率 (%)
memory.total：显存总量 (MiB)
memory.used：显存已用 (MiB)
power.draw：功耗 (W)
fan.speed：风扇转速 (%)
utilization.encoder：硬件编码利用率 (%)
utilization.decoder：硬件解码利用率 (%)
clocks.gr：图形 (Graphics) 时钟频率 (MHz)
clocks.mem：显存时钟频率 (MHz)
--format=csv：以 CSV 格式输出，行首带列名，便于阅读。
-l 1：每秒刷新一次。

python run_experiments.py \
  --ckpt pretrained/CelebAMask-HQ-512x512.pt \
  --dataset ./data/BMP \
  --outdir experiments_csv \
  --device cuda \
  --size 512 \
  --batches 8 16 \
  --channels awgn awgn_rayleigh \
  --snrs 10 15 20 \
  --steps 100 200 300 \
  --num_images_list 0 8 16 \
  --repeat 3

1) 仅 AWGN（不走空口，先验证主流程/指标）
python main_SDR.py `
  --ckpt "pretrained\CelebAMask-HQ-512x512.pt" `
  --dataset ".\data\examples" `
  --outdir "results\inversion_sdr" `
  --size 512 `
  --batch_size 1 `
  --num_workers 0 `
  --w_plus `
  --step 300 `
  --lr 0.1 `
  --snr_db 15 `
  --lambda_lpips 1.0 `
  --lambda_l1 0.3 `
  --lambda_ssim 0.0 `
  --lambda_mean 0.0


2) 开启 SDR 空口（USRP 发、Lime 收，920 MHz）
python main_SDR.py `
  --ckpt "pretrained\CelebAMask-HQ-512x512.pt" `
  --dataset ".\data\examples" `
  --outdir "results\inversion_sdr" `
  --size 512 `
  --batch_size 1 `
  --num_workers 0 `
  --w_plus `
  --step 300 `
  --lr 0.1 `
  --snr_db 15 `
  --lambda_lpips 1.0 `
  --lambda_l1 0.3 `
  --lambda_ssim 0.0 `
  --lambda_mean 0.0 `
  --use_sdr `
  --freq 920000000 `
  --samp_rate 1000000 ` 
  --tx_serial "30B584E" `
  --rx_serial "00090726074D281F" ` 
  --tx_gain 30 `
  --rx_gain 40 `
  --target_rms 0.2 `
  --rx_timeout_s 0.2
3)
python test_connection.py `
  --freq 920000000 `
  --samp_rate 500000 `
  --tone 100000 `
  --tx_driver uhd `
  --rx_driver lime `
  --tx_serial 30B584E `
  --rx_serial 00090726074D281F `
  --tx_gain 60 `
  --rx_gain 50 `
  --rx_antenna LNAL `
  --captures 30 `
  --timeout_s 0.2 `
  --show_plot


python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 8 --channels awgn --snrs 10 15 20 --steps 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 --num_images_list 16 --repeat 5
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 --channels awgn --snrs 10 15 20 --steps 40 60  --num_images_list 8 --repeat 2

python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 15 --steps 400 --num_images_list 16 --repeat 5
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 15 16 17 18 19 20 --steps 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 --num_images_list 16 --repeat 5
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --steps 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 --num_images_list 16 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 20 21 22 23 24 25 26 27 28 29 30 --steps 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 --num_images_list 16 --repeat 5
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 1 2 3 4 5  --steps 40 60 80 100  --num_images_list 8 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 1 2 3  --steps 40 60 80 100  --num_images_list 16 --repeat 1

python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 --channels awgn --snrs 4 5 6  --steps 40 60 80  --num_images_list 4 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 --channels awgn --snrs 4 5 6  --steps 40 60 80  --num_images_list 8 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 --channels awgn --snrs 4 5 6  --steps 40 60 80  --num_images_list 16 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 --channels awgn --snrs 4 5 6  --steps 40 60 80  --num_images_list 32 --repeat 1

python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120 --num_images_list 8 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 16 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 32 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 40 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 48 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 56 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 8 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 64 --repeat 1

python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 10 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 10 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 12 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 12 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 14 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 14 --repeat 1
python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 16 --channels awgn --snrs 4   --steps 40 80 120  --num_images_list 16 --repeat 1

result_analyze.py 的作用是生成系统结果分析图

python auto_experiment.py --ckpt pretrained/CelebAMask-HQ-512x512.pt --dataset ./data/BMP --outdir experiments_csv --device cuda --batches 4 --channels awgn --snrs 10 --steps 100  --num_images_list 8 --repeat 0
