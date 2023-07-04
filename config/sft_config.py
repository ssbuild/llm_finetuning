# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 10:13

import json
import os
import torch
from transformers import BitsAndBytesConfig
from config.constant_map import train_info_models

train_model_config = train_info_models['bloom-560m']


global_args = {
    "load_in_8bit": False, 
    "load_in_4bit": False,
    #load_in_4bit 量化配置
    "quantization_config": None,
    "config_merge": {
    }
}


train_info_args = {
    'devices': 1,
    'data_backend': 'parquet',  #one of record lmdb arrow_stream arrow_file,parquet, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大

    # 预训练模型配置
    **train_model_config,


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






