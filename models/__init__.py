# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: __init__.py
import re
from collections import OrderedDict

from transformers import PretrainedConfig
from torch import nn
from models.llm_model import *


class SftWeightMinMax:

    def save_pretrained_merge_lora(self,sft_weight_path: str):
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model : LoraModel = self.backbone
        model: nn.Module = lora_model.merge_and_unload()
        #保存hf权重，可用infer.py推理
        # torch.save(model.model.state_dict(),weight_path_file)
        model.model.save_pretrained(sft_weight_path)
        return model

    def save_pretrained_merge_lora_and_restore(self, sft_weight_path: str):
        assert os.path.exists(os.path.dirname(sft_weight_path))
        assert self.lora_args is not None and self.lora_args.with_lora
        lora_model: LoraModel = self.backbone
        lora_model.merge_adapter()
        # 保存hf权重，可用infer.py推理
        #torch.save(lora_model.model.model.state_dict(), weight_path_file)
        lora_model.model.model.save_pretrained(sft_weight_path)
        lora_model.unmerge_adapter()

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        assert os.path.exists(sft_weight_path)
        if self.lora_args is not None and self.lora_args.with_lora:
            # 恢复权重
            self.backbone.load_weight(pretrained_model_name_or_path=sft_weight_path,is_trainable=is_trainable)

        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            # 恢复权重
            self.backbone.load_weight(pretrained_model_name_or_path=sft_weight_path, is_trainable=is_trainable)
        else:
            weight_dict = torch.load(sft_weight_path)
            weights_dict_new = OrderedDict()
            valid_keys = ['module','state_dict']
            for k in valid_keys:
                if k in weight_dict:
                    weight_dict = weight_dict[k]
                    break
            for k, v in weight_dict.items():
                rm_key = '_TransformerLightningModule__backbone'
                if k.startswith(rm_key):
                    base_model_prefix = self.backbone.base_model_prefix
                    k = re.sub(r'{}.{}.'.format(rm_key,base_model_prefix), '', k)
                weights_dict_new[re.sub(r'_forward_module\.', '', k)] = v
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(weights_dict_new, strict=strict)

    def save_sft_weight(self,sft_weight_path, merge_lora_weight=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            if merge_lora_weight:
                # lora 合并权重 转换 hf权重
                self.save_pretrained_merge_lora(sft_weight_path)
            else:
                #只保存 lora 权重
                self.backbone.save_pretrained(sft_weight_path)
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            self.backbone.save_pretrained(sft_weight_path)
        else:
            #保存hf权重
            config = self.get_llm_model().config
            config.save_pretrained(sft_weight_path)
            self.get_llm_model().save_pretrained(sft_weight_path)

class MyTransformer(MyTransformerLM,SftWeightMinMax, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args
        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: LoraModel = LoraModel(self.backbone, lora_args)
            print('==' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)
        elif prompt_args is not None and prompt_args.with_prompt:
            self.backbone.enable_input_require_grads()
            model: PromptModel = get_prompt_model(self.backbone, prompt_args)
            print('==' * 30, 'prompt info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

    def get_model_lr(self, model=None, lr=None):
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.prompt_args and self.prompt_args.with_prompt:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model


