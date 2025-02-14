import base64
import requests
import tqdm
import json
import os
import pdb
from collections import defaultdict
import random
from PIL import Image
import cv2
import re
import tqdm
import torch
import warnings
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import RealSATDynamic

import os


if __name__=="__main__":

    # Load the dataset
    args= {
    } 

    dataset = RealSATDynamic(args, tokenizer=None, image_processor=None)

    ### load the model
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype="float16", device_map="cuda:0")
    processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

    model.eval()


    all_responses = []
    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        image_paths, images, prompt, text_label, correct_answer, answer_choices, datatype = entry
        
        ### get model output
        inputs = processor(text=prompt, images=images, return_tensors='pt').to(torch.float16).to('cuda:0')

        output = model.generate(
            **inputs, 
            max_new_tokens=20,
        )
        
        response_text = processor.batch_decode(output, skip_special_tokens=True)[0]

        # pdb.set_trace()
        response_text = response_text.lower().split("assistant:")[-1].strip()

        # pdb.set_trace()
        print(prompt, response_text)
        # Save the response
        all_responses.append({
          "prompt": prompt,
          "text_label": text_label,
          "image_path": image_paths,
          "response": response_text,
          "answer": correct_answer,
          "answer_choices": answer_choices,
          "dataset": datatype
        })
        # pdb.set_trace()

        # Save the responses
        
        with open("responses/llavaov_realsat_response.json", "w") as f:
            json.dump(all_responses, f)

        if ind>=4000:
            break
      