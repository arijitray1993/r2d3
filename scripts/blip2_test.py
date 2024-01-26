"""
Test BLIP-2 capability to generate scenes. 
"""

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json

from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel


if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )

    model.to(device)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    prompt = "Question: how many cats are there? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    print(inputs.keys())
    # print the shape of each input
    for k, v in inputs.items():
        print(f"{k}: {v.shape}")

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)

    print(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    
    lora_model = get_peft_model(model, lora_config)

    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/ProcTHOR/data/all_room_json_programs_ai2_train.json", "r"))

    image_data_path = "/projectnb/ivc-ml/array/research/robotics/ProcTHOR/vis/ai2thor"