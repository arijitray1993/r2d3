from transformers import Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration
import torch
import torch.nn as nn
import sys
import pdb 

'''
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/MiniGPT-4")
from minigpt4.common.utils import now
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
'''

from transformers import StoppingCriteria, StoppingCriteriaList
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

#from transformers import LlamaForCausalLM, CodeLlamaTokenizer


class Blip2ModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.language_model.lm_head.weight.requires_grad = True

    def generate(self, input_ids, attention_mask, pixel_values, labels=None, min_new_tokens=1, max_new_tokens=100):
        
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

        return generated_ids

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs
    

class InstructBlipModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", max_position_embeddings=2048)


    def generate(self, qformer_input_ids, input_ids, attention_mask, pixel_values, labels=None, min_new_tokens=1, max_new_tokens=100):
        
        generated_ids = self.model.generate(qformer_input_ids=qformer_input_ids,input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

        return generated_ids

    def forward(self, qformer_input_ids, input_ids, attention_mask, pixel_values, labels=None):
        
        outputs = self.model(qformer_input_ids=qformer_input_ids, input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs

class LlavaModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_path = "liuhaotian/llava-v1.5-7b"

        _, self.model, _, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):
        # pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):
        
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        
        return outputs


class LlavaModelAggImgInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_path = "liuhaotian/llava-v1.5-7b"

        _, self.model, _, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):

        # pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):

        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)

        return outputs



class CodeLLamaInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
        self.max_new_tokens = args['max_new_tokens']

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate(self, input_ids):
        generated_ids = self.model.generate(input_ids=input_ids, max_new_tokens=self.max_new_tokens)
        return generated_ids


""" MiniGPT4 interface, in progress
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def get_multimodal_prompt(model, img, prompt):
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).input_ids.cuda()
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]

    seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [seg_embs[0].repeat(len(img), 1, 1), img, seg_embs[1].repeat(len(img), 1, 1)]
    #mixed_embs = [emb for pair in zip(seg_embs[:-1], [img]) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

class MiniGPT4Interface(nn.Module):
    def __init__(self, args):
        super().__init__()
        cfg = Config(args)
        cfg.pretty_print()
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        model = model.cuda().eval()

        stop_words_ids = [torch.tensor([835]).cuda(),
                        torch.tensor([2277, 29937]).cuda()]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    
    def forward(self, pixel_values, prompts, labels=None):
        curr_img_feature, _ = model.encode_img(curr_processed)
        curr_multimodal_embd = get_multimodal_prompt(model, curr_img_feature, prompt)
        outputs = self.model(input_embeds=input_embeds, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs

    def generate(self, input_embeds, 
                 max_new_tokens=100, 
                 stopping_criteria=None, 
                 num_beams=None, do_sample=False,
                 min_length=1, 
                 top_p=1, 
                 repetition_penalty=None, 
                 length_penalty=None, 
                 temperature=0.2):
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        return outputs
"""