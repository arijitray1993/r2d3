import json
import os
import pdb  # noqa
import random
from collections import defaultdict
import itertools
from itertools import combinations
import functools
import pickle as pkl
 
import torch
import tqdm  # noqa
from PIL import Image
from torch.utils.data import Dataset
import time
import torchvision

from torch.utils.data import WeightedRandomSampler
from transformers import Blip2Processor, InstructBlipProcessor # , CodeLlamaTokenizer
from transformers import AutoProcessor

from shapely.geometry.polygon import Polygon

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np
import h5py

import ast
import cv2
import wandb

import sys
# sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
# sys.path.append("models/LLaVA_modified/LLaVA")
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.mm_utils import expand2square
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import csv


class ArkitScenes(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        
        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass


class Stanford2D3DS(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode


        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass


class Structured3D(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode


        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass


class Grit20M(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode


        
    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass