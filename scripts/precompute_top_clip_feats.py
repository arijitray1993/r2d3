import torch
import os
import json
from PIL import Image
import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import expand2square
import tqdm
import pdb

if __name__=="__main__":

    model_path = "liuhaotian/llava-v1.5-13b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    
    vision_tower.to(device='cuda')
    image_processor = vision_tower.image_processor

    vision_tower.select_feature = 'cls_patch'

    apartment_folder = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"

    for apartment_id in tqdm.tqdm(os.listdir(apartment_folder)):
        top_down_image_path = os.path.join(apartment_folder, apartment_id, "top_down.png")

        if not os.path.exists(top_down_image_path):
            continue

        # pdb.set_trace()
        if os.path.exists(os.path.join(apartment_folder, apartment_id, "top_down_feats.pt")):
            continue

        top_down_image = Image.open(top_down_image_path).convert("RGB")

        top_down_image = expand2square(top_down_image, tuple(int(x*255) for x in image_processor.image_mean))
        # pdb.set_trace()
        top_down_image = image_processor.preprocess(top_down_image, return_tensors='pt')['pixel_values']

        with torch.no_grad():
            top_down_feats = vision_tower(top_down_image.to(device=vision_tower.device, dtype=vision_tower.dtype))

        top_down_feats = top_down_feats.cpu()

        top_down_feats_path = os.path.join(apartment_folder, apartment_id, "top_down_feats.pt")

        # pdb.set_trace()

        torch.save(top_down_feats, top_down_feats_path)
