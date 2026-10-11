import os
import time
import urllib.request
import logging
from pathlib import Path
from typing import List, Dict, Union, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try importing torch for the lightweight method
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TORCH_AVAILABLE:
    class LightweightSRNet(nn.Module):
        """Simple 4-layer CNN for 2x super-resolution."""
        def __init__(self, scale_factor=2):
            super().__init__()
            self.scale_factor = scale_factor
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv4(x)
            x = self.pixel_shuffle(x)
            return x

class SuperResolver:
    def __init__(self, method: str = 'enhanced_bicubic', scale: int = 2, device: str = 'cpu'):
        """
        method: 'opencv_dnn' | 'enhanced_bicubic' | 'torch_lightweight'
        scale: upscaling factor (2 recommended)
        device: 'cpu' or 'cuda'
        """
        self.method = method
        self.scale = scale
        self.device = device
        
        self.sr_model = None
        self.torch_model = None
        
        if self.method == 'opencv_dnn':
            self._init_opencv_dnn()
        elif self.method == 'torch_lightweight':
            self._init_torch()
            
    def _init_opencv_dnn(self):
        try:
            from cv2 import dnn_superres
            self.sr_model = dnn_superres.DnnSuperResImpl_create()
            
            # Ensure model file exists
            # Save it to the current file's directory
            current_dir = Path(__file__).parent
            model_path = current_dir / f"ESPCN_x{self.scale}.pb"
            
            if not model_path.exists():
                logger.info(f"Model file {model_path} not found. Attempting to download...")
                self.download_model(model_path, self.scale)
                
            if model_path.exists():
                self.sr_model.readModel(str(model_path))
                self.sr_model.setModel("espcn", self.scale)
                if self.device == 'cuda':
                    self.sr_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.sr_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info(f"Successfully loaded ESPCN_x{self.scale} for opencv_dnn.")
            else:
                logger.warning("Failed to download model. Falling back to enhanced_bicubic.")
                self.method = 'enhanced_bicubic'
        except Exception as e:
            logger.warning(f"Failed to initialize OpenCV DNN Superres: {e}. Falling back to enhanced_bicubic.")
            self.method = 'enhanced_bicubic'
            
    def _init_torch(self):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Falling back to enhanced_bicubic.")
            self.method = 'enhanced_bicubic'
            return
            
        try:
            self.torch_device = torch.device(self.device if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
            self.torch_model = LightweightSRNet(scale_factor=self.scale).to(self.torch_device)
            self.torch_model.eval()
            logger.info(f"Successfully initialized LightweightSRNet on {self.torch_device}.")
        except Exception as e:
            logger.warning(f"Failed to initialize PyTorch model: {e}. Falling back to enhanced_bicubic.")
            self.method = 'enhanced_bicubic'

    @staticmethod
    def download_model(save_path: Union[str, Path], scale: int = 2):
        """Downloads ESPCN model from OpenCV's repository."""
        url = f"https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x{scale}.pb"
        try:
            logger.info(f"Downloading from {url} to {save_path}")
            urllib.request.urlretrieve(url, str(save_path))
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")

    @staticmethod
    def enhanced_bicubic_upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
        """Bicubic upscale + unsharp mask + optional CLAHE. Always available."""
        # 1. Bicubic resize
        h, w = image.shape[:2]
        upscaled = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        # 2. Unsharp masking
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 3)
        sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
        
        # 3. CLAHE on L channel in LAB color space
        try:
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            merged = cv2.merge((cl, a_channel, b_channel))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception as e:
            # Fallback if CLAHE fails (e.g. grayscale image)
            return sharpened

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Super-resolve a single image. Returns enhanced image."""
        if self.method == 'enhanced_bicubic':
            return self.enhanced_bicubic_upscale(image, self.scale)
        elif self.method == 'opencv_dnn':
            try:
                return self.sr_model.upsample(image)
            except Exception as e:
                logger.error(f"Error in opencv_dnn upsample: {e}. Falling back to bicubic.")
                return self.enhanced_bicubic_upscale(image, self.scale)
        elif self.method == 'torch_lightweight' and self.torch_model is not None:
            # Convert HWC BGR to NCHW RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.torch_device)
            
            with torch.no_grad():
                out_tensor = self.torch_model(img_tensor)
                
            out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
            # Convert back to BGR
            return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        else:
            return self.enhanced_bicubic_upscale(image, self.scale)

    def enhance_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Super-resolve multiple images."""
        return [self.enhance(img) for img in images]

    def benchmark_latency(self, image: np.ndarray, num_runs: int = 50) -> dict:
        """Measure average inference time. Returns dict with mean_ms, std_ms, fps."""
        # Warmup
        for _ in range(max(1, min(5, num_runs//2))):
            _ = self.enhance(image)
            
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.enhance(image)
            end = time.perf_counter()
            times.append((end - start) * 1000) # ms
            
        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        
        return {
            'method': self.method,
            'scale': self.scale,
            'device': self.device,
            'mean_ms': round(mean_ms, 2),
            'std_ms': round(std_ms, 2),
            'fps': round(fps, 2)
        }

if __name__ == '__main__':
    # Define paths
    base_dir = Path(r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection")
    dataset_dir = base_dir / r"Dataset\DsPCBSD+\Data_YOLO\images\train"
    workspace_dir = base_dir / "ai_workplace"
    src_dir = workspace_dir / "src"
    
    # Create result directories
    figures_dir = workspace_dir / "results" / "figures"
    tables_dir = workspace_dir / "results" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 5 sample images
    image_paths = []
    if dataset_dir.exists():
        image_paths = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png"))
        image_paths = image_paths[:5]
        
    if not image_paths:
        logger.warning(f"No images found in {dataset_dir}. Creating dummy images for testing.")
        test_imgs = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(3)]
    else:
        test_imgs = [cv2.imread(str(p)) for p in image_paths]
        # Downscale for testing SR to speed up demo
        test_imgs = [cv2.resize(img, (256, 256)) for img in test_imgs]

    methods_to_test = ['enhanced_bicubic', 'opencv_dnn', 'torch_lightweight']
    results_latency = []
    
    # Prepare figure for visual comparison (using first image)
    img_to_show = test_imgs[0]
    img_to_show_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, len(methods_to_test) + 1, figsize=(15, 5))
    axes[0].imshow(img_to_show_rgb)
    axes[0].set_title("Original (Bilinear Upscaled for Display)")
    axes[0].axis('off')
    
    # Run and benchmark each method
    for i, method_name in enumerate(methods_to_test, 1):
        logger.info(f"Testing method: {method_name}")
        resolver = SuperResolver(method=method_name, scale=2, device='cpu')
        
        # Benchmark
        latency = resolver.benchmark_latency(img_to_show, num_runs=10)
        results_latency.append(latency)
        logger.info(f"Latency for {method_name}: {latency}")
        
        # Visual output
        enhanced_img = resolver.enhance(img_to_show)
        enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(enhanced_img_rgb)
        axes[i].set_title(f"{method_name}")
        axes[i].axis('off')
        
    # Save figure
    fig_path = figures_dir / "sr_comparison.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison figure to {fig_path}")
    
    # Save latency table
    df_latency = pd.DataFrame(results_latency)
    csv_path = tables_dir / "sr_latency.csv"
    df_latency.to_csv(csv_path, index=False)
    logger.info(f"Saved latency table to {csv_path}")
    print("\nBenchmark Results:")
    print(df_latency.to_string())
