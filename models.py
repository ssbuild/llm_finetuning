# @Time    : 2023/4/2 22:39
# @Author  : tk
# @FileName: models
from typing import List, Tuple

import torch
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForCausalLM
from transformers import PreTrainedModel
from data_utils import postprocess

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
        return postprocess(response)

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
        if load_in_8bit:
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        super(MyTransformerLM, self).__init__(*args, **kwargs)





class MyTransformer(MyTransformerLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model