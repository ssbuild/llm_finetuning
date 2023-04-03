# @Time    : 2023/4/2 22:39
# @Author  : tk
# @FileName: models
from typing import List, Tuple

import torch
from deep_training.nlp.models.lora import LoraModel, LoraArguments
from deep_training.nlp.models.transformer import TransformerForCausalLM
from transformers import PreTrainedModel
from data_utils import postprocess


class Generate:
    @classmethod
    @torch.no_grad()
    def generate(cls,model, tokenizer, query: str, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        prompt = "Human：" + query + "\nAssistant："
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        return postprocess(response)


class MyTransformerLM(TransformerForCausalLM):
    def __init__(self, *args, **kwargs):
        super(MyTransformerLM, self).__init__(*args, **kwargs)





class MyTransformer(MyTransformerLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraArguments = kwargs.pop('lora_args', None)
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