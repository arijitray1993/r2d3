import os
import pdb  # noqa
import sys  # noqa
import math
# import time
from collections import defaultdict  # noqa
 
# training related imports
from models import eval_funcs as eval
import hydra
import custom_datasets.dataloaders as datasets
import models.model_interface as models
import torch
import tqdm

from accelerate import Accelerator
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
 
# import transformers
 
# import models.preprocess as preprocess
import wandb  # noqa
from omegaconf import DictConfig, OmegaConf  # noqa
 
# utils
from utils import utils
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel

import time
import numpy as np
import random

# fix all seeds
seed = 20249
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class GenericModule():
    def __init__(
        self,
        model,
        model_input_choice,
        train_dataloader, 
        test_dataloader,
        metrics=None,
        lr=None,
        max_epochs=0,
        weight_decay=0.01,
        lr_scheduler_choice=None,
        load_checkpoint=None,
        eval_every = 100,
        logger = None, # expects a wandb run object
        exp_name = "",
        eval_only = False,
    ):
        super().__init__()
        self.model = model
        self.logger = logger
        self.exp_name = exp_name
        self.eval_only = eval_only

        # initialize eval functions and metrics
        self.metrics = metrics
        for metric_entry in metrics:
            metric_name = metric_entry[0]
            metric_func = metric_entry[1]
            # metric_args = metric_entry[2]
            # print(metric_name)
            metric_args = {}
            metric_args["exp_name"] = self.exp_name
            metric_args["logger"] = self.logger
            setattr(self, metric_name, getattr(eval, metric_func)(metric_args))

        self.metric_to_track = []

        # set up training
        self.accelerator = Accelerator()
        if train_dataloader is not None:
            self.num_epochs = max_epochs
            self.num_training_steps = self.num_epochs * len(train_dataloader)

            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.lr_scheduler = get_scheduler(
                lr_scheduler_choice,
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.num_training_steps
            )

            self.accelerator.register_for_checkpointing(self.lr_scheduler)

        if load_checkpoint:
            self.accelerator.load_state(load_checkpoint)

        self.model_input_choice = model_input_choice
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        if train_dataloader is not None:
            self.train_dataloader, self.test_dataloader, self.model, self.optimizer = self.accelerator.prepare(
                self.train_dataloader, self.test_dataloader, self.model, self.optimizer)
        else:
            self.test_dataloader, self.model = self.accelerator.prepare(
                self.test_dataloader, self.model)
        self.eval_every = eval_every
        

    def train(self,):
        progress_bar = tqdm.tqdm(range(self.num_training_steps))
        self.global_step = 0
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                #outputs = self.model.forward(*[batch[inp] for inp in self.model_input_choice])

                outputs = self.model.forward(**{inp: batch[inp] for inp in self.model_input_choice})
                loss = outputs.loss

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.set_description(f"train_loss: {loss.item()}")
                progress_bar.update(1)
                self.logger.log({"train_loss": loss.item(), "step": self.global_step})

                # validate at certain intervals
                if self.global_step % self.eval_every == 0 and epoch>0:
                    self.validate()
                
                    # compute metrics after running all eval loop
                    for entry in self.metrics:
                        metric_name = entry[0]
                        getattr(self, metric_name).compute()

                self.global_step += 1
 
    def validate(self,):
        progress_bar = tqdm.tqdm(range(len(self.test_dataloader)))
        count = 0
        for batch in self.test_dataloader:
            #self.model_outputs = self.model.forward(
            #    *[batch[inp] for inp in self.model_input_choice]
            #)
            generated_ids = self.model.generate(**{inp: batch[inp] for inp in self.model_input_choice})

            generated_ids[generated_ids==-200] = 1
            generated_text = self.test_dataloader.dataset.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # pdb.set_trace()
            print("Generated text: ")
            print(generated_text)

            print("Ground truth text: ")
            print(batch["text_labels"][0])

            for entry in self.metrics:
                metric_name = entry[0]
                
                getattr(self, metric_name).update(generated_text, batch)
                
                #self.logger.log(
                #    metric_name,
                #    getattr(self, metric_name),
                #)
            progress_bar.set_description(f"")
            progress_bar.update(1)
            count+=1
            if count>100:
                break
        
        # compute metrics after running all eval loop
        for entry in self.metrics:
            metric_name = entry[0]
            getattr(self, metric_name).compute()

        if not self.eval_only:
            self.save_model()

    def save_model(self,):
        # save the huggingface model locally
        self.model.save_pretrained(f"checkpoints/{self.exp_name}/" +f"{self.logger.name}_{self.global_step}")


@hydra.main(version_base=None, config_path="confs/", config_name="exp_config")
def main(cfg: DictConfig):
    cfg = cfg.exp_name
    print(os.getcwd())

    if not os.path.exists("checkpoints/" + cfg.exp_name):
        os.mkdir(
            "checkpoints/"
            + cfg.exp_name
        )

    wandb.login()
    run = wandb.init(project=cfg.exp_name, config=dict(cfg))
    
    # ======================= load dataloaders =====================
    dataloader_train = None
    if not cfg.eval_only:
        train_dataset = getattr(datasets, cfg.train_dataset_choice)(
            cfg.train_dataset_args
        )
        # this chooses the dataset of choice from data/datasets.py
        # according to the variable "train_dataset_choice" in cfg
        # and initializes it with cfg.train_dataset_args
    
        dataloader_train = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=train_dataset.collate_fn
            if cfg.get("train_collate_fn")
            else None,
            shuffle=False if cfg.get("no_shuffle") or cfg.get("use_sampler") else True,
            num_workers = cfg.num_workers,
            sampler=train_dataset.sampler if cfg.get("use_sampler") else None,
        )
    
    if cfg.eval_only:
        # if eval only, load test data, or else
        # load val data while training to choose hyperparameters.
        dataset_val = getattr(datasets, cfg.val_dataset_choice)(cfg.val_dataset_args)
    else:
        dataset_val = getattr(datasets, cfg.valtrain_dataset_choice)(
            cfg.valtrain_dataset_args
        )
    
    if cfg.get("val_batch_size"):
        val_batch_size = cfg.val_batch_size
    else:
        val_batch_size = cfg.batch_size

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        collate_fn=dataset_val.collate_fn if cfg.get("val_collate_fn") else None,
        num_workers = cfg.num_workers,
        shuffle=False
    )
    print("data loaded...")
    
    # =============== define model choice ============================
    model = getattr(models, cfg.model_choice)(cfg.model_init_args)
    if cfg.eval_only:
        model.eval()  # turns batch norm off.

    if cfg.get("tune_lora"):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[], #["embed_tokens", "lm_head"],
        )
        if cfg.get("lora_model_path"):
            lora_model = PeftModel.from_pretrained(model, cfg.get("lora_model_path"))
            lora_model = lora_model.half()
            
            if not cfg.eval_only:
                # go through the dict and set lora layers to requuires grad True
                
                for name, param in lora_model.named_parameters():
                    if ("q_proj" in name or "v_proj" in name) and ('weight' in name) and ('default' in name):
                        # pdb.set_trace()
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        else:
            lora_model = get_peft_model(model, lora_config)
            lora_model = lora_model.half()
        
        print("Lora models loaded...")

        for name, param in lora_model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

        total_params = sum(param.numel() for param in lora_model.parameters())
        print(f"Total number of params: {total_params}")

        total_params = sum(param.numel() for param in lora_model.parameters() if param.requires_grad)
        print(f"    Trainable params: {total_params}")
        
        # === training module ===
        training_module = GenericModule(
            model=lora_model,
            model_input_choice=cfg.model_input_choice,
            metrics=cfg.metrics,
            lr=cfg.lr,
            max_epochs=cfg.num_epochs,
            weight_decay=cfg.weight_decay,
            lr_scheduler_choice=cfg.get("lr_scheduler"),
            load_checkpoint=cfg.load_checkpoint,
            train_dataloader=dataloader_train,
            test_dataloader=dataloader_val,
            eval_every=cfg.eval_every,
            logger = run,
            exp_name = cfg.exp_name,
            eval_only = cfg.eval_only,
        )

    else:
        print("models loaded...")

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

        total_params = sum(param.numel() for param in model.parameters())
        print(f"Total number of params: {total_params}")

        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"    Trainable params: {total_params}")

        # === training module ===
        training_module = GenericModule(
            model=model,
            model_input_choice=cfg.model_input_choice,
            metrics=cfg.metrics,
            lr=cfg.lr,
            max_epochs=cfg.num_epochs,
            weight_decay=cfg.weight_decay,
            lr_scheduler_choice=cfg.get("lr_scheduler"),
            load_checkpoint=cfg.load_checkpoint,
            train_dataloader=dataloader_train,
            test_dataloader=dataloader_val,
            eval_every=cfg.eval_every,
            logger = run,
            exp_name = cfg.exp_name,
            eval_only = cfg.eval_only,
        )

    if cfg.eval_only:
        training_module.validate()
    else:
        training_module.train()

    wandb.finish()


if __name__ == "__main__":

   main()
