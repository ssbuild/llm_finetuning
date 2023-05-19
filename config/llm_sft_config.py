# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 10:13

import json
import os

#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
global_args = {
    "load_in_8bit": False, # lora 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
}


# 默认禁用lora 相关模块 , lora 和 adalora 只能同时启用一个
lora_info_args = {
    'with_lora': False,  # 是否启用lora模块
    'lora_type': 'lora',
    'r': 8,
    'target_modules': ['query_key_value'],  # bloom,gpt_neox
    # 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
    # 'target_modules': ['c_attn'], #gpt2
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'fan_in_fan_out': False,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
}

adalora_info_args = {
    'with_lora': False,  # 是否启用adalora模块
    'lora_type': 'adalora',
    'r': 8,
    'target_modules': ['query_key_value'],  # bloom,gpt_neox
    # 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
    # 'target_modules': ['c_attn'], #gpt2
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'fan_in_fan_out': False,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "

    'target_r':8, # Target Lora matrix dimension.
    'init_r': 12, #Intial Lora matrix dimension.
    'tinit': 0, #The steps of initial warmup.
    'tfinal': 0, #The steps of final warmup.
    'deltaT': 1, #Step interval of rank allocation.
    'beta1': 0.85, #Hyperparameter of EMA.
    'beta2': 0.85, #Hyperparameter of EMA.
    'orth_reg_weight': 0.5, #The orthogonal regularization coefficient.
    'total_step': None, #The total training steps.
    'rank_pattern': None, #The saved rank pattern.
}


prompt_info_args = {
    "with_prompt": False,
    "prompt_type": "prefix_tuning", # one of prompt_tuning,p_tuning,prefix_tuning,adaption_prompt
    "task_type": "causal_lm", #  one of seq_cls,seq_2_seq_lm,causal_lm,token_cls
    "prefix_projection": False, # Whether to project the prefix tokens"
    "num_virtual_tokens": 16, # Number of virtual tokens
    # "token_dim": 2048, # The hidden embedding dimension of the base transformer model.
    # "num_transformer_submodules": 1, # The number of transformer submodules in the base transformer model.
    # "num_attention_heads" : 24, # The number of attention heads in the base transformer model.
    # "num_layers": 1, # The number of layers in the base transformer model.
    # "encoder_hidden_size": 2048, # The hidden size of the encoder
}

train_info_args = {
    'devices': 1,
    'data_backend': 'record',  #one of record lmdb, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大
    'model_type': 'bloom',
    # 预训练模型路径 , 从0训练，则置空
    'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
    'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    # 'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
    # 'config_name': '/data/nlp/pre_models/torch/opt/opt-350m/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    # 'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',



    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/finetune_train_examples.json'],
    'max_epochs': 20,
    'max_steps': -1,
    'optimizer': 'lion', # one of adamw,adam,lamb,lion

    'scheduler_type': 'CAWR',
    'scheduler':{'T_mult': 1,
             'rewarm_epoch_num': 0.5,  # 如果 max_epochs is not None !
             # 'T_0': 50000,    # 如果 max_epochs is None , 设定步数
             'verbose': False},

    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': True},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},


    'optimizer_betas': (0.9, 0.999),
    'train_batch_size': 2,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 2e-5,  #
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length':  512, #
    'max_target_length': 100,  # 预测最大长度, 保留字段
    'use_fast_tokenizer': False,
    'do_lower_case': False,

    ##############  lora模块
    'lora': {**lora_info_args},
    'adalora': {**adalora_info_args},
    'prompt': {**prompt_info_args},

}



enable_deepspeed = False

def get_deepspeed_config():
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None
    with open(os.path.join(os.path.dirname(__file__),'deepspeed.json'), mode='r', encoding='utf-8') as f:
        deepspeed_config = json.loads(f.read())
    return deepspeed_config