# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model

import os
from typing import List, Tuple

import torch
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.prompt import PromptModel,PromptArguments,get_prompt_model,PromptLearningConfig
from deep_training.nlp.models.transformer import TransformerForCausalLM
from transformers import PreTrainedModel

#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False

class Generate:
    @classmethod
    @torch.no_grad()
    def generate(cls,model, tokenizer, query: str, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        # prompt = "Human：" + query + "\nAssistant："
        #自行加模板
        prompt = query
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        return response

    @classmethod
    @torch.no_grad()
    def chat(cls,model, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []

        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        history = history + [(query, response)]
        return response, history


class MyTransformerLM(TransformerForCausalLM):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(MyTransformerLM, self).__init__(*args, **kwargs)
        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()