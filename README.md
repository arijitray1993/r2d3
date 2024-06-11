# R2D3: Imparting Spatial Reasoning by Reconstructing 3D Scenes from 2D Images

[arxiv] coming soon. in submission.

## Pre-requisities

Create a conda environment:

```
conda env create -f r2d3_conda.yml
conda activate r2d3_conda
```

Get all the data and place it in `custom_datasets/procThor/` (create the folder if not existing):
- `final_data_neurips.json`: 
- `GPT4V_room_descriptions_topdown.json`: 
- `GPT4V_room_descriptions.json`:  

Get all the images and make sure they are in `custom_datasets/procThor/images/train/`. This folder should have folders corresponding to each room and the images from various corners inside the room folder. In the dataloader, we only select the image with most objects visbile for tasking the model to reconstruct it in 3D.

To be able to run the training/evaluations with infered depth, run:

```
cd scripts
python compute_depth.py
```

## Evaluations
To run the various adaptation approaches in the paper, use the following command and use the appropriate exp_name. 

```
python -m accelerate.commands.launch --num_processes=1 main.py exp_name=<exp_name>
```

Here are the `exp_name` choices for the various adaptation methods:

- `llava_incomplete_oneim_campolygon_nomarksbaseline`: This one runs the standard fine-tuning of LLaVA without precise camera orientation during training. This is denoted as `FT` in the tables in the paper. 
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



