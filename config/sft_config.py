# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 10:13

import json
import os
import torch
from transformers import BitsAndBytesConfig

# **************切换 配置文件 修改 config.__init__.py

# Quantization parameters are controlled from the BitsandbytesConfig (see HF documenation) as follows:
#
# Loading in 4 bits is activated through load_in_4bit
# The datatype used for the linear layer computations with bnb_4bit_compute_dtype
# Nested quantization is activated through bnb_4bit_use_double_quant
# The datatype used for qunatization is specified with bnb_4bit_quant_type. Note that there are two supported quantization datatypes fp4 (four bit float) and nf4 (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using nf4.

#如果显卡支持int8 可以开启
global_args = {
    "load_in_8bit": False, # lora 如果显卡支持int8 可以开启
    "load_in_4bit": False,

    #load_in_4bit 量化配置
    "quantization_config": None,
    "config_merge": {
    }
}

if global_args['load_in_4bit'] != True:
    global_args['quantization_config'] = None




train_info_args = {
    'devices': 1,
    'data_backend': 'record',  #one of record lmdb, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大
    'model_type': 'rwkv',
    # 预训练模型路径 , 从0训练，则置空
    'model_name_or_path': '/data/nlp/pre_models/torch/rwkv/rwkv-4-169m-pile',
    'config_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-169m-pile/config.json',
    'tokenizer_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-169m-pile',

    # 'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    # 'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
    # 'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',

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

    # *** optimizer
    # lamb,adamw_hf,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,
    # adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion,lion_8bit,lion_32bit,
    # paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,
    # lamb_fused_dp adagrad_cpu_dp adam_cpu_dp adam_fused_dp

    # *** scheduler
    # linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,
    # constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau

    'optimizer': 'lion',
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
    #'do_lower_case': False,


}


#配置检查


if global_args['load_in_8bit'] == global_args['load_in_4bit'] and global_args['load_in_8bit'] == True:
    raise Exception('load_in_8bit and load_in_4bit only set one at same time!')




