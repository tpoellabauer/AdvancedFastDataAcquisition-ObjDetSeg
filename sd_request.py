import requests
import base64
import os
import re
import cv2 as cv
import json
from time import time
import random
import concurrent.futures

lora_base_name = "vision-bs4_r16_a16_c16_a16-ycbv-"

objects = ['01coffee', '02cracker', '03sugar', '04tomato', '05mustard', '06tuna', '07chocolate', '08jello_strawberry', '09potted_meat', '10banana', '11can', '12bleach_cleanser', '13bowl', '14mug', '15driller', '16wood_block', '17scissor', '18pen', '19small_clamp', '20big_clamp', '21foam_brick']

NUM_SOURCE_SAMPLES = 100

# This script implements an API call for the Stable Diffusion Web UI https://github.com/AUTOMATIC1111/stable-diffusion-webui.
# It generates Images for the specified Objects in the variable object_ids.
# It uses the masks in object_id/masks_translated_resized as input for ControlNet.
# Prompts are a concatenation from prompts specified in promptsofAllObjects.json and a weather condition prompt.
# For each mask a number of prompts are randomly selected. (in our case number_of_prompts = 5)

def normalize_string(string):
    # Replace spaces with underscores
    normalized_string = string.replace(' ', '_')
    # Remove special characters except underscore
    normalized_string = re.sub(r'[^a-zA-Z0-9_]', '', normalized_string)
    return normalized_string

def api_request(controlnet_image_file, controlnet_mask_file, obj_name, prompt, negative_prompt, output_image_dir, lora_weight=1):
    with open(controlnet_image_file, "rb") as f:
        # Read the image data
        image_data = f.read()

    with open(controlnet_mask_file, "rb") as f:
        # Read the image data
        mask_data = f.read()

    url = "http://yourserver:7860"

    # inspired by https://github.com/Mikubill/sd-webui-controlnet/pull/194#issuecomment-1465309925
    # Encode the image data as base64
    image_base64 = base64.b64encode(image_data)
    mask_base64 = base64.b64encode(mask_data)

    # Convert the base64 bytes to string
    image_string = image_base64.decode("utf-8")
    mask_string = mask_base64.decode("utf-8")

    payload = {
        "prompt": f'<lora:{lora_base_name}{obj_name}:{lora_weight}>, {prompt}',
        "negative_prompt": negative_prompt,
        "steps": 25,
        "cfg_scale": 5,
        'sampler_name': 'DPM++ SDE', # use your favourite sampler such as 'DPM++ SDE', #'DPM++ 2M', 'DDIM'
        "batch_size": 1,
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "enabled": True,
                        "image": image_string,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny [d14c016b]",
                        "weight": 1.0,
                        "resize_mode": "Crop and Resize",
                        "pixel_perfect": True,
                        "processor_res": 512,
                        "threshold_a": 155,
                        "threshold_b": 255,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "controlnet_masks": mask_string,
                        "control_mode": "ControlNet is more important" # select from "ControlNet is more important", "Balanced", "My prompt is more important"
                    }
                    ,
                    {
                        "enabled": True,
                        "image": image_string,
                        "module": "depth_anything",
                        "model": "controlnet_depth_anything [48a4bc3a]",
                        "weight": 1.0,
                        "resize_mode": "Crop and Resize",
                        "pixel_perfect": True,
                        "processor_res": 512,
                        "guidance_start": 0,
                        "guidance_end": 1,
                        "controlnet_masks": mask_string,
                        "control_mode": "ControlNet is more important"
                    }
                ]
            }
        }
    }
    # Send said payload to said URL through the API.
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    # Assuming r['images'] is a list containing base64 encoded images
    for i in range(len(r['images'])-2):
        image_data = base64.b64decode(r['images'][i])
        starting_ts = int(time())
        file_name = f"{controlnet_image_file.split('.')[0]}".split("/")[-1]
        out_file_path = f"some_file_path.jpg"
        with open(f'{out_file_path}', 'wb') as f:
            f.write(image_data)

#for object_id in objects:
def process_object(data, object_id):
    obj_num_str = object_id[:2]
    
    # find all images of this object
    # Filter files that start with the specified prefix and have image extensions
    image_files = [file for file in all_image_paths if file.startswith(obj_num_str) and file.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]
    # Build the list of image paths
    image_paths = [os.path.join(base_path, file) for file in image_files]

    # pick NUM_SOURCE_SAMPLES random images per object
    image_paths = random.choices(image_paths, k=NUM_SOURCE_SAMPLES)
    
    # Specifies how many prompts are used per mask
    number_of_prompts = 1

    prompt = f"a {data["quality"]} photograph of {object_id} placed on top of a {data["surface"]}, {data["state"]}, {data["composition"]}, {data["colors"]}, {data["imageQuality"]}, {data["weather"]}, at a {data["place"]}, in front of a detailed {data["background"]} background, {data["weather"]}, at {data["timeOfDay"]}"

    negative_prompt = 'deformed, (person:1.2), (human:1.2), illustration, unrealistic, surreal, surrealistic, impressionism, rendering, drawing, painting, low quality, low detail, blurry, (multiple objects:1.2), (two objects:1.2), (more than one object), object in the background, duplicates, broken, (multiples:1.2), assembly, low detail background, non photo-realistic, add-on, addon, (extension:1.2)'

    print(f'starting {object_id}:')
    for j, file in enumerate(image_paths):
        print(f'processed {j}/{len(image_paths)} files, last file started: {file}')
        for i in range(number_of_prompts):
            api_request(file, file.replace(".jpg", "_mask.png"), object_id, prompt, negative_prompt, out_path)

base_path = "somepath"
out_path = "new"
split = "train"

if not os.path.exists(f"{out_path}"):
    os.makedirs(f"{out_path}")

# Get all files in the folder
base_path = f"{base_path}/{split}"
all_image_paths = os.listdir(f"{base_path}")
# remove masks
all_image_paths = [string for string in all_image_paths if not string.endswith("mask.png")]

with open("prompts.json", "r") as json_file:
    data = json.load(json_file)

for object_id in objects:
    process_object(data, object_id)