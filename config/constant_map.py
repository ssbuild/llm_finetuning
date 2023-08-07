# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

train_info_models = {
    'bloom-560m': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    },
    'bloom-1b7': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    },
    'opt-350m': {
        'model_type': 'opt',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
        'config_name': '/data/nlp/pre_models/torch/opt/opt-350m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',
    },

    'llama-7b-hf': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
        'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    },

    'Llama-2-7b-chat-hf':{
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/Llama-2-7b-chat-hf',
        'config_name': '/data/nlp/pre_models/torch/llama/Llama-2-7b-chat-hf/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/Llama-2-7b-chat-hf',
    },

    'chatyuan-7b': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
        'config_name': '/data/nlp/pre_models/torch/llama/ChatYuan-7B/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
    },

    'rwkv-4-430m-pile': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
        'config_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
    },

}


# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    't5': ['qkv_proj'],
    'moss': ['qkv_proj'],
    'chatglm': ['query_key_value'],
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}
