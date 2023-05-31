# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/5/31 14:43

import json
import os

# 切换配置

# from config.sft_config import *
from config.sft_config_lora import *
# from config.sft_config_lora_int4 import *
# from config.sft_config_lora_int8 import *
# from config.sft_config_ptv2 import *

#预处理
if 'rwkv' in train_info_args['tokenizer_name'].lower():
    train_info_args['use_fast_tokenizer'] = True

#lora adalora prompt 使用 deepspeed_offload.json 配置文件
enable_deepspeed = False

def get_deepspeed_config():
    '''
        lora prompt finetuning 使用 deepspeed_offload.json
        普通finetuning 使用deepspeed.json
    '''
    # 是否开启deepspeed
    if not enable_deepspeed:
        return None

    # 选择 deepspeed 配置文件
    is_need_update_config = False
    conf = train_info_args.get('lora',train_info_args.get('adalora',train_info_args.get('prompt',None)))
    if conf is not None and conf.get('with_lora',False) or conf.get('with_prompt',False) :
        is_need_update_config = True
        filename = os.path.join(os.path.dirname(__file__), 'deepspeed_offload.json')
    else:
        filename = os.path.join(os.path.dirname(__file__), 'deepspeed.json')


    with open(filename, mode='r', encoding='utf-8') as f:
        deepspeed_config = json.loads(f.read())

    #lora offload 同步优化器配置
    if is_need_update_config:
        optimizer = deepspeed_config.get('optimizer',None)
        if optimizer:
            optimizer['params']['betas'] = train_info_args.get('optimizer_betas', (0.9, 0.999))
            optimizer['params']['lr'] = train_info_args.get('learning_rate', 2e-5)
            optimizer['params']['eps'] = train_info_args.get('adam_epsilon', 1e-8)
    return deepspeed_config