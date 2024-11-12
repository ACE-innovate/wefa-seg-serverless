import runpod
import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
import base64
import json
import sys
from io import BytesIO

# Configure global model settings
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Model loading
save_memory = True
disable_verbosity()
if save_memory:
    enable_sliced_attention()

config = OmegaConf.load('./configs/inference.yaml')
current_model_ckpt = config.pretrained_model
model_config = config.config_file

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(current_model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Model loading function (if changing model during runtime)
def load_model(new_model_ckpt):
    global model, ddim_sampler, current_model_ckpt
    if new_model_ckpt != current_model_ckpt:
        model.load_state_dict(load_state_dict(new_model_ckpt, location='cuda'))
        current_model_ckpt = new_model_ckpt

# Helper functions for image processing
def base64_to_cv2_image(base64_str):
    img_str = base64.b64decode(base64_str)
    np_img = np.frombuffer(img_str, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_to_base64(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str

def process_images(data):
    model_name = data.get('model', './step_357500_slim.ckpt')
    model_ckpt_map = {
        'boys': 'boys.ckpt',
        'men': 'men.ckpt',
        'women': 'women.ckpt',
        'girls': 'girls.ckpt'
    }
    current_model_ckpt = './step_357500_slim.ckpt'
    new_model_ckpt = model_ckpt_map.get(model_name, current_model_ckpt)
    load_model(new_model_ckpt)  # Load model if needed

    seed = int(data.get('seed', 1351352))
    steps = int(data.get('steps', 50))
    guidance_scale = float(data.get('guidance_scale', 3.0))

    ref_image = base64_to_cv2_image(data['ref_image'])
    tar_image = base64_to_cv2_image(data['tar_image'])

    ref_mask_img = base64_to_cv2_image(data['ref_mask'])
    ref_mask = cv2.cvtColor(ref_mask_img, cv2.COLOR_RGB2GRAY)
    ref_mask = (ref_mask > 128).astype(np.uint8)

    tar_mask_img = base64_to_cv2_image(data['tar_mask'])
    tar_mask = cv2.cvtColor(tar_mask_img, cv2.COLOR_RGB2GRAY)
    tar_mask = (tar_mask > 128).astype(np.uint8)

    gen_image = inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale, seed, steps)
    gen_image_base64 = image_to_base64(gen_image)
    return gen_image_base64

# Define the handler function for RunPod
def handler(job):
    # Access input data from the job
    job_input = job["input"]
    
    try:
        # Process the images using the provided data
        result_image_base64 = process_images(job_input)
        return {"status": "success", "output": result_image_base64}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Start the serverless handler with RunPod
runpod.serverless.start({"handler": handler})