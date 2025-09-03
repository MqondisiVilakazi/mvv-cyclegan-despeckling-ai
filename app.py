import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

# --- 1. SETUP AND CONFIGURATION ---

app = Flask(__name__)

MODEL_DIR = 'checkpoints'
# This path is now only for local testing; it won't exist on the server
BASE_DATA_PATH = r"C:\CycleGAN\data" 
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- 2. MODEL DEFINITION ---

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(True)
        ]
        for _ in range(n_residual_blocks): model += [ResidualBlock(256)]
        model += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(64), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

# --- 3. HELPER FUNCTIONS ---

def resize_and_pad(img_pil, desired_size=256):
    old_size = img_pil.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    resized_img = img_pil.resize(new_size, Image.Resampling.LANCZOS)
    padded_img = Image.new("RGB", (desired_size, desired_size))
    paste_position = ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    padded_img.paste(resized_img, paste_position)
    return padded_img

def find_latest_model(model_dir):
    pattern = re.compile(r'G_AB_epoch_(\d+)\.pth')
    highest_epoch = -1
    latest_model_path = None
    if not os.path.isdir(model_dir): return None
    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > highest_epoch:
                highest_epoch = epoch_num
                latest_model_path = os.path.join(model_dir, filename)
    return latest_model_path

def infer_ground_truth_path(noisy_image_path, base_data_path):
    filename = os.path.basename(noisy_image_path)
    clean_filename = filename.replace("noisy_", "")
    for folder in ["trainB", "testB"]:
        gt_path = os.path.join(base_data_path, folder, clean_filename)
        if os.path.exists(gt_path):
            return gt_path
    return None

def calculate_psnr_ssim(img1_path, img2_path, transform):
    img1_pil = Image.open(img1_path).convert('RGB')
    img2_pil = Image.open(img2_path).convert('RGB')
    img1_tensor = transform(img1_pil)
    img2_tensor = transform(img2_pil)
    img1_np = (img1_tensor.numpy().transpose(1, 2, 0) * 0.5) + 0.5
    img2_np = (img2_tensor.numpy().transpose(1, 2, 0) * 0.5) + 0.5
    psnr = peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)
    ssim = structural_similarity(img1_np, img2_np, data_range=1.0, channel_axis=-1, multichannel=True)
    return psnr, ssim

# --- 4. LOAD THE MODEL ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latest_model_path = find_latest_model(MODEL_DIR)
if latest_model_path:
    model = Generator(3, 3).to(device)
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {latest_model_path}")
else:
    model = None

# --- 5. WEB ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        if file and model:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            original_pil = Image.open(upload_path).convert("RGB")
            padded_img = resize_and_pad(original_pil)
            padded_img.save(upload_path)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            input_tensor = transform(padded_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            output_img_np = (output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
            output_img_pil = Image.fromarray(output_img_np.astype(np.uint8))
            result_filename = "cleaned_" + filename
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            output_img_pil.save(result_path)

            metrics = None
            # UPDATED: Only calculate metrics if the local GT folder exists
            if os.path.isdir(BASE_DATA_PATH):
                gt_path = infer_ground_truth_path(upload_path, BASE_DATA_PATH)
                if gt_path:
                    gt_pil = Image.open(gt_path).convert("RGB")
                    padded_gt = resize_and_pad(gt_pil)
                    # Note: We don't overwrite the original GT file, just use the padded version in memory
                    
                    # Temporarily save padded GT for calculation
                    temp_gt_path = os.path.join(app.config['RESULT_FOLDER'], "temp_gt.png")
                    padded_gt.save(temp_gt_path)

                    original_psnr, original_ssim = calculate_psnr_ssim(upload_path, temp_gt_path, transform)
                    cleaned_psnr, cleaned_ssim = calculate_psnr_ssim(result_path, temp_gt_path, transform)
                    metrics = {
                        "original_psnr": f"{original_psnr:.2f} dB", "original_ssim": f"{original_ssim:.4f}",
                        "cleaned_psnr": f"{cleaned_psnr:.2f} dB", "cleaned_ssim": f"{cleaned_ssim:.4f}"
                    }
            
            original_image_path = os.path.join('uploads', filename).replace('\\', '/')
            result_image_path = os.path.join('results', result_filename).replace('\\', '/')

            return render_template('result.html',
                                   original_image=original_image_path,
                                   result_image=result_image_path,
                                   metrics=metrics,
                                   download_filename=result_filename) 

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Removed debug=True for production readiness