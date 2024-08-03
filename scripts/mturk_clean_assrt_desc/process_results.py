from collections import defaultdict
import csv
import json
import pdb
import re
import requests

# OpenAI API Key
api_key_file = "/projectnb/ivc-ml/array/research/robotics/openai"
with open(api_key_file, "r") as f:
  api_key = f.read().strip()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption(image_path, prompt, api_key):
  # Getting the base64 string
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4o",
    "messages": [
    {
        "role": "user",
        "content": [
        {
            "type": "text",
            "text": prompt
        },
        ]
    }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response

results_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/Batch_5235057_batch_results.csv"

results = []
with open(results_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

with open('assetid_to_info.json', 'r') as f:
    assetid_to_info = json.load(f)


if True:
    for asset in assetid_to_info:
        entries = assetid_to_info[asset]
        new_entries = []
        for im, obj, desc in entries:
            
            # clean desc
            desc = desc.lower().strip()
            
            # remove first word if it is "a"
            if desc.split(" ")[0] == "a":
                desc = " ".join(desc.split(" ")[1:])
            
            # remove first word if it is "an"
            if desc.split(" ")[0] == "an":
                desc = " ".join(desc.split(" ")[1:])
            
            # remove period at the end
            if desc[-1] == ".":
                desc = desc[:-1]
            if desc.split(" ")[-1] == ".":
                desc = " ".join(desc.split(" ")[:-1])   
            
            # remove "in the air" from the phrase
            if "in the air" in desc:
                desc = desc.replace("in the air", "")
            
            if "in air" in desc:
                desc = desc.replace("in air", "")

            if "floating in the air" in desc:
                desc = desc.replace("floating in the air", "")
            
            if "drawrers" in desc:
                desc = desc.replace("drawrers", "drawers")

            if 'floating mid-air' in desc:
                desc = desc.replace('floating mid-air', '')

            if 'none' in desc:
                desc = obj
            
            if '3d' in desc:
                desc = obj
            
            if 'furniture' in desc:
                desc = obj

            if 'silhouette' in desc:
                desc = desc.replace('silhouette of a ', '')

            if 'decagon' in desc or 'parallelogram' in desc or 'octagon' in desc or 'hexagon' in desc or 'pentagon' in desc or 'heptagon' in desc or 'nonagon' in desc:
                desc = desc.split(' and ')[0]

            if 'shape is' in desc:
                desc = desc.split('shape is ')[0]

            if "floats in" in desc:
                desc = desc.split("floats in")[0]
            

            
            #pdb.set_trace()
            if desc=='n':    
                desc = input() or desc

            desc = re.sub(r'[^A-Za-z0-9 ]+', '', desc)


            if len(desc.split(" ")) > 6:
                # ask gpt4 for shorten it
                prompt = f"Here is a text describing an object: {desc}. Can you please read this description and make it into very few words that just describe the object. For instace, I want a description like 'brown wooden chair'. Just a few words like this. Please just output the text nothing else."
                desc = get_caption(im, prompt, api_key)
                desc = desc.json()['choices'][0]['message']['content']

            print(f"{im} - {desc}")

            new_entries.append((im, obj, desc))
        
        assetid_to_info[asset] = new_entries
        




if False:
    assetid_to_info = defaultdict(list)
    for r in results:
        im1 = r['Input.image_url1']
        im2 = r['Input.image_url2']
        im3 = r['Input.image_url3']
        im4 = r['Input.image_url4']

        obj1 = r['Input.object_class1']
        obj2 = r['Input.object_class2']
        obj3 = r['Input.object_class3']
        obj4 = r['Input.object_class4']

        asset1 = r['Input.asset_name1']
        asset2 = r['Input.asset_name2']
        asset3 = r['Input.asset_name3']
        asset4 = r['Input.asset_name4']

        desc1 = r['Answer.caption1'].lower().strip()
        desc2 = r['Answer.caption2'].lower().strip()
        desc3 = r['Answer.caption3'].lower().strip()
        desc4 = r['Answer.caption4'].lower().strip()

        assetid_to_info[asset1].append((im1, obj1, desc1))
        assetid_to_info[asset2].append((im2, obj2, desc2))
        assetid_to_info[asset3].append((im3, obj3, desc3))
        assetid_to_info[asset4].append((im4, obj4, desc4))


pdb.set_trace()
with open('assetid_to_info.json', 'w') as f:
    json.dump(assetid_to_info, f)

    