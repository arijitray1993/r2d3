from datasets import load_dataset
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import os
import torch
import json
import pdb
import tqdm

if __name__=="__main__":

    ai2thor_image_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"
    lsun_dataset = load_dataset("pcuenq/lsun-bedrooms")['train']

    # load vit large to judge image sim
    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k').cuda()

    # precompute ai2thor room image features 
    ai2thor_room_images = os.listdir(ai2thor_image_path)
    ai2thor_im2feature = {}
    for room_image in tqdm.tqdm(ai2thor_room_images):
        room_im = Image.open(os.path.join(ai2thor_image_path, room_image))
        inputs = processor(images=room_im, return_tensors="pt")

        # ship to cuda
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        ai2thor_im2feature[room_image] = last_hidden_state.detach().cpu()   

    # pdb.set_trace()

    closest_lsun_ai2thor = []
    for ind, entry in enumerate(tqdm.tqdm(lsun_dataset)):
        bedroom_im = entry["image"]
    
        inputs = processor(images=bedroom_im, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.detach().cpu()

        # compare to all ai2thor room images
        closest_dist = float("inf")
        closest_room = None
        for ai2thor_room_im, ai2thor_room_feature in ai2thor_im2feature.items():
            # compute similarity
            # sim = torch.cosine_similarity(last_hidden_state, ai2thor_room_feature, dim=-1)
            
            dist = torch.cdist(last_hidden_state/torch.norm(last_hidden_state), ai2thor_room_feature/torch.norm(ai2thor_room_feature))
            #pdb.set_trace()
            # take diagonal of dist matrix
            dist = dist[0].diag()
            dist = dist.mean()

            if dist < closest_dist and dist < 0.1:
                closest_dist = dist
                closest_room = ai2thor_room_im
        
        # pdb.set_trace()
        if closest_room is None:
            continue

        closest_lsun_ai2thor.append((ind, closest_room, float(closest_dist)))
        
        with open("closest_lsun_ai2thor.json", "w") as f:
            json.dump(closest_lsun_ai2thor, f)
