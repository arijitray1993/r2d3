from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import pdb
import json
import os
import tqdm

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))
    
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)

    for apartment_ind, im_file, caption in tqdm.tqdm(caption_data):

        depth_im_path = im_file.replace(".png", "_depth.pt")

        # if os.path.exists(depth_im_path):
        #    continue

        image = Image.open(im_file)
    
        # prepare image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        pdb.set_trace()
        # save the depth as pt
        # torch.save(prediction.detach().cpu(), depth_im_path)

