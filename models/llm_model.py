# coding=utf8
# @Time    : 2023/5/12 20:41
# @Author  : tk
# @FileName: llm_model
from deep_training.nlp.models.transformer import TransformerForCausalLM
from deep_training.trainer.pl.modelweighter import *
import logging
logger = logging.getLogger(__name__)
class TransformerForLM(TransformerForCausalLM):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        if not load_in_4bit:
            quantization_config = kwargs.get("quantization_config",None)
            if quantization_config:
                load_in_4bit = quantization_config.load_in_4bit

        if not load_in_8bit and not load_in_4bit:
            kwargs.pop("device_map", None)
            kwargs.pop("quantization_config", None)

        super(TransformerForLM, self).__init__(*args, **kwargs)

        # for param in self.model.parameters():
        #     param.requires_grad = False  # freeze the model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)

        # class CastOutputToFloat(nn.Sequential):
        #     def forward(self, x):
        #         return super().forward(x).to(torch.float32)
        #
        # self.model.lm_head = CastOutputToFloat(self.model.lm_head)



    def enable_input_require_grads(self):
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()



class MyTransformer(TransformerForLM, ModelWeightMinMax, with_pl=True):
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        self.prompt_args = prompt_args

        # print('before',self.config)
        #可能扩充词表
        self.resize_token_embs(new_num_tokens)
        # print('after',self.config)

        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: LoraModel = LoraModel(self.backbone, lora_args,auto_prepare_kbit_training=False)
            print('==' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)
            # for name, module in model.named_modules():
            #     if isinstance(module, LoraLayer):
            #         module = module.to(torch.bfloat16)
            #     if 'norm' in name:
            #         module = module.to(torch.float32)
            #     if 'lm_head' in name or 'embed_tokens' in name:
            #         if hasattr(module, 'weight'):
            #             if module.weight.dtype == torch.float32:
            #                 module = module.to(torch.bfloat16)

        elif prompt_args is not None and prompt_args.with_prompt:
            self.backbone.enable_input_require_grads()
            model: PromptModel = get_prompt_model(self.backbone, prompt_args)
            print('==' * 30, 'prompt info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

    def resize_token_embs(self,new_num_tokens):
        if new_num_tokens is not None:
            logger.info(f"new_num_tokens:{new_num_tokens}")
            model: PreTrainedModel = self.backbone.model
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if new_num_tokens != embedding_size:
                logger.info("resize the embedding size by the size of the tokenizer")
                model.resize_token_embeddings(new_num_tokens)

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.with_lora:
            return [(self.backbone, lr)]
        elif self.prompt_args and self.prompt_args.with_prompt:
            return [(self.backbone, lr)]
        return super(MyTransformer, self).get_model_lr(model, lr)


    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            #PromptModel 方法覆盖原来方法
            return self.backbone
        return self.backbone.model





