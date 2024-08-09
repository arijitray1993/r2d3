# R2D3: Imparting Spatial Reasoning by Reconstructing 3D Scenes from 2D Images

[arxiv] coming soon. in submission.

![The task of the model is to precisely describe a 3D scene using the object names, attributes, and 3D locations that can be rendered in a graphics engine to reflect the image. We use this task to evaluate a multimodal language model's ability to understand the 3D nature of the scene holistically.](teaser_fig.png)
The task is to precisely describe a 3D scene using the object names, attributes, and 3D locations that can be rendered in a graphics engine to reflect the image. We use this task to evaluate a multimodal language model's ability to understand the 3D nature of the scene holistically.

## Pre-requisities

Create a conda environment:

```
conda env create -f r2d3_conda.yml
conda activate r2d3_conda
```

Get all the following data and place it in `custom_datasets/procThor/` (create the folder if not existing):
- `final_data_neurips.json`: https://huggingface.co/datasets/array/r2d3_data/blob/main/final_data_neurips.json 
- `GPT4V_room_descriptions_topdown.json`: https://huggingface.co/datasets/array/r2d3_data/blob/main/GPT4V_room_descriptions_topdown.json 
- `GPT4V_room_descriptions.json`:  https://huggingface.co/datasets/array/r2d3_data/blob/main/GPT4V_room_descriptions.json 
- `asset_descriptions`: https://huggingface.co/datasets/array/r2d3_data/blob/main/asset_descriptions_all.json 

Get all the image tar.gz from here (https://huggingface.co/datasets/array/r2d3_data). Then concatenate all the parts with the following command (this will require around 80GB of disk space):
```
cat images.tar.gz.parta* > images.tar.gz
tar -xvf images.tar.gz
```

Make sure the `images` folder is in `custom_datasets/procThor/`. This folder should have folders corresponding to each room and the images from various corners inside the room folder. In the dataloader, we only select the image with most objects visbile for tasking the model to reconstruct it in 3D.

To be able to run the training/evaluations with infered depth, run:

```
cd scripts
python compute_depth.py
```

## Data Format

### Room images and graphics program data
`final_data_neurips.json` contains the data needed to run your MLLM on the benchmark. 

This JSON contains a list of room entries. Each room entry contains the following information:
`[program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name]`

Here is the description for each of the fields:
- **`program_text`**: This is the graphics program for the room. You can also generate a lighter version of the graphics program without children objects if you do `generate_program_from_roomjson(house_json, include_children=False)` by importing the function from `utils.ai2thor_utils`. 

The program text is in the format:

```
polygon: [[1164, 0, 969], [1164, 0, 1747], [1553, 0, 1747], [1553, 0, 969]]
floor_material: 'OrangeCabinet 1'
wall_material: ['BrownMarbleFake 1', 'BrownMarbleFake 1', 'BrownMarbleFake 1', 'BrownMarbleFake 1']
obj_0: ['TV_Stand_222_1', [1424, 20, 1718], [0, 180, 0]] # asset_id, 3D location, 3D rotation in degrees along each axis.
obj_1: ['Sofa_204_1', [1424, 50, 1415], [0, 0, 0]]
window_0: ['Window_Slider_36x36', [662, 150, 0], [[617, 104, 0], [707, 195, 0]], 2] # asset_id, 3D location, window rectangle shape, wall index.
```

- **`house_json`**: This is the full raw JSON needed for the AI2THOR simulator to generate the room. The `program_text` above is a lighter version of it to fit into context length for MLLMs. You can generate an even lighter version (without children objects) if you do:
```
from utils.ai2thor_utils import generate_program_from_roomjson
program_text = generate_program_from_roomjson(house_json, include_children=False)
```

- **`og_house_json`**: This is the raw AI2THOR JSON for the entire apartment from which the room came from. 

- **`cam_ind_to_position`**: The camera positions for each of the images taken for the room.  

- **`all_imgs`**: The paths to the image files for each of the camera positions in `cam_ind_to_position`. 

- **`all_objs`**: The objects present in each of the images. It's a list of list of objects for each image. In the paper, we choose the image with the most number of objects for the tasking the MLLM to reconstruct the 3D. 

- **`all_seg_frames`**: The segmentation map for each of the images in `all_imgs`. 

- **`color_to_objid`**: The mapping of the seg frame color to the object id. This is the object id present in the program_text.   

- **`obj_id_to_name`**: The object class name for the object ids. 

### Asset Descriptions

Instead of the asset ids, we can also predict it in natural language like:
```
obj_0: Black modern TV stand at location [1424, 20, 1718] with rotation [0, 180, 0]
```
https://huggingface.co/datasets/array/r2d3_data/blob/main/asset_descriptions_all.json has these descrptions in the folloing format: a list with each entry as `[image_of_asset, asset_name, object_class, description caption]`.



## Evaluations
To run the various adaptation approaches in the paper, use the following command and use the appropriate exp_name. 

```
python -m accelerate.commands.launch --num_processes=1 main.py exp_name=<exp_name>
```

Here are the `exp_name` choices for the various adaptation methods:

- `llava_incomplete_oneim_campolygon_nomarksbaseline`: This one runs the standard fine-tuning of LLaVA without precise camera orientation during training. This is denoted as `FT` in the tables in the paper. 
- `llavadino_oneim_campolygon_nomarksbaseline`: This one runs the standard fine-tuning of LLaVA with a Dino encoder instead of ViT. Also without precise camera orientation during training. Denoted as `DinoV2 FT` in the tables.
- `llava_incomplete_oneim_campolygon_onlyyuvdepth_nomarks`: This runs the fine-tuning with estimated overlayed depth on the images (also assuming no precise camera orientation during training). Denoted as `depth` in the tables. 
- `llava_incomplete_oneim_campolygonangle_nomarksbaseline`: This runs the fine-tuning of LLaVA with precise camera orientation specidfied using language. Denoted as `Orient Language` in the tables. 
- `llava_incomplete_oneim_campolygon_randompointorient` : This runs the fine-tuning with the precise camera orientation marked as a combination of langauge and a point marked in the image with the 3D position. Denoted as `Visual Point` in the tables. 

Evaluations will require one 48GB NVIDIA card (preferably L40S, L40, RTX8000, or A6000). 


## Training
The command above runs eval on the test set. To start a training, visit the config file for the correspoding `exp_name` in `confs/exp_name/` and change the following:
- `eval_only` to `False`
- `lora_model_path` to `False`
- `num_eval_steps` to a low number like 3 to speed up the training avoid running full eval for every checkpoint.

All paper numbers are reported until trained to convergence which is around 115K iteration steps.

Training will require at least *two* 48GB NVIDIA cards (preferably L40S, L40, or A6000). 


## Dataset Generation 
Look at `scripts/generate_cfg_data_ProcTHOR.py` to see how the dataset was generated. 

Look at `scripts/generate_gpt4_descriptions.py` and `scripts/generate_gpt4_descriptions_topdown.py` to see how we captioned the rooms. 

Finally, look at `scripts/get_gpt4_obj_descriptions.py` to see how we caption each of the assets. Some assets could not be captioned by GPT4 and we manually caption them. 

Also, take a look at `custom_datasets/dataloaders.py` at class `ProcTHOR_image_camposition_marked` to see how we load and process the data for training/evaluating the model.

## Metrics
Take a look at `models/eval_funcs.py` to see how the metrics are computed. 


## BibTeX

If you use this code, please cite:

```
@misc{ray2024r2d3,
      title={R2D3: Imparting Spatial Reasoning by Reconstructing 3D Scenes from 2D Images}, 
      author={Arijit Ray and Dina Bashkirova and Reuben Tan and Kuo-Hao Zeng and Bryan A. Plummer and Ranjay Krishna and Kate Saenko},
      year={2024},
}
```