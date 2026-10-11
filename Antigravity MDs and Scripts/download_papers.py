import requests
import urllib.parse
import os
import time

papers = [
    "A Lightweight PCB Defect Detection Algorithm Based on an Improved YOLOv11s",
    "Lightweight PCB Defect Detection Method Based on SCF-YOLO",
    "GESC-YOLO: Improved Lightweight PCB Defect Detection",
    "A Lightweight PCB Defect Detection Method Based on Group Convolution and Adaptive Pruning",
    "MAS-YOLO: Lightweight Detection Based on Improved YOLOv12",
    "YOLO-DefXpert: Advanced Defect Detection Using Improved YOLOv11",
    "A High-Accuracy PCB Defect Detection Algorithm Based on Improved YOLOv12",
    "Small Defect Detection in PCBs Based on Multiscale Edge Strengthening and Improved YOLOv10",
    "PCB Defect Detection Based on Improved YOLOv8n",
    "PCB Board Defect Detection Algorithm Based on Improved YOLOv8",
    "NeoPCB-YOLO: Next-generation Enhanced Optimization YOLO",
    "RF-YOLO: Enhanced YOLOv8 Combining RFCAConv and FDPN",
    "ABF-YOLO: Fine-Grained PCB Defect Detection with Axial Attention",
    "Application Research of CycleGAN and YOLO in PCB Infrared Defect Detection",
    "YOLOv8x-HICAUps: Attention Based Approach for PCB Defect Detection",
    "Defect Detection Network in PCB Circuit Devices Based on GAN Enhanced YOLOv11",
    "Enhanced YOLOv11 Framework for High Precision PCB Defect Detection",
    "PCB-YOLO: Coordinate Attention and Multi-Scale Feature Fusion",
    "YOLO-HDEW: An Efficient PCB Defect Detection Model",
    "Research on Lightweight PCB Defect Detection (YOLOv11-CGL)",
    "VR-YOLO: PCB Defect Detection with Viewpoint Robustness",
    "Vision Transformer and Mamba-Attention Fusion for PCB Defect Detection",
    "A Lightweight Transformer for PCB Defects Detection (LPDETR)",
    "Deep Learning-Enhanced Defects Detection for Printed Circuit Boards",
    "Performance Analysis of YOLO11 Models in PCB Defect Detection",
    "Defect Detection of PCB Based on Improved YOLOv11",
    "PCB Defect Detection Based on Lightweight DL Fusion Model",
    "MSA-DETR: Multi-Scale Alignment Detection Transformer for PCB",
    "ResNet-SE-CBAM Siamese Networks for Few-Shot PCB Defect Classification",
    "SM-FSOD: Meta-Learning for Few-Shot PCB Defect Detection",
    "Deep Learning Algorithms for Defect Detection on Electronic Assemblies: A Systematic Literature Review"
]

output_dir = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\All Papers"

def get_oa_pdf(title):
    query = urllib.parse.quote(title)
    url = f"https://api.openalex.org/works?filter=title.search:{query}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('results'):
            work = data['results'][0]
            if work.get('open_access') and work['open_access'].get('is_oa'):
                best_oa = work['open_access'].get('oa_url')
                if best_oa and best_oa.endswith('.pdf'):
                    return best_oa
                elif work.get('best_oa_location') and work['best_oa_location'].get('pdf_url'):
                    return work['best_oa_location']['pdf_url']
    except Exception as e:
        print(f"Error querying {title}: {e}")
    return None

def download_pdf(url, title):
    try:
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        file_path = os.path.join(output_dir, f"{safe_title}.pdf")
        if os.path.exists(file_path):
            print(f"Already downloaded: {safe_title}")
            return True
            
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, timeout=15, headers=headers)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {safe_title}")
            return True
    except Exception as e:
        pass
    return False

print("Starting to fetch PDFs from OpenAlex...")
for i, title in enumerate(papers):
    pdf_url = get_oa_pdf(title)
    if pdf_url:
        download_pdf(pdf_url, title)
    else:
        print(f"No direct OA PDF found for: {title}")
    time.sleep(0.5)
