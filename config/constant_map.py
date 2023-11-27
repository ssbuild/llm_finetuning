# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]

MODELS_MAP = {

    'CausalLM-14B': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/CausalLM-14B',
        'config_name': '/data/nlp/pre_models/torch/llama/CausalLM-14B',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/CausalLM-14B',
    },
    'LingoWhale-8B': {
        'model_type': 'lingowhale',
        'model_name_or_path': '/data/nlp/pre_models/torch/lingowhale/LingoWhale-8B',
        'config_name': '/data/nlp/pre_models/torch/lingowhale/LingoWhale-8B',
        'tokenizer_name': '/data/nlp/pre_models/torch/lingowhale/LingoWhale-8B',
    },
    'CausalLM-7B': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/CausalLM-7B',
        'config_name': '/data/nlp/pre_models/torch/llama/CausalLM-7B',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/CausalLM-7B',
    },
    'bloom-560m': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    },
    'bloom-1b7': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    },
    'opt-350m': {
        'model_type': 'opt',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
        'config_name': '/data/nlp/pre_models/torch/opt/opt-350m',
        'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',
    },

    'llama-7b-hf': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
        'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    },

    'Llama-2-7b-chat-hf':{
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/Llama-2-7b-chat-hf',
        'config_name': '/data/nlp/pre_models/torch/llama/Llama-2-7b-chat-hf',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/Llama-2-7b-chat-hf',
    },

    'Llama2-Chinese-7b-Chat':{
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-7b-Chat',
        'config_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-7b-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-7b-Chat',
    },

    'Llama2-Chinese-13b-Chat':{
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-13b-Chat',
        'config_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-13b-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-13b-Chat',
    },

    'chatyuan-7b': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
        'config_name': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
    },
    'tigerbot-13b-chat': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat',
        'config_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat',
    },
    'tigerbot-13b-chat-int4': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat-int4',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat-int4',
    },

    'openbuddy-llama2-70b-v10.1': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16',
        'config_name': '/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16',
    },



    'rwkv-4-430m-pile': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
        'config_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
    },

    'BlueLM-7B-Chat': {
        'model_type': 'BlueLM',
        'model_name_or_path': '/data/nlp/pre_models/torch/bluelm/BlueLM-7B-Chat',
        'config_name': '/data/nlp/pre_models/torch/bluelm/BlueLM-7B-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/bluelm/BlueLM-7B-Chat',
    },
    'BlueLM-7B-Chat-32K': {
        'model_type': 'BlueLM',
        'model_name_or_path': '/data/nlp/pre_models/torch/bluelm/BlueLM-7B-Chat-32K',
        'config_name': '/data/nlp/pre_models/torch/bluelm/BlueLM-7B-Chat-32K',
        'tokenizer_name': '/data/nlp/pre_models/torch/bluelm/BlueLM-7B-Chat-32K',
    },
    'BlueLM-7B-Base': {
        'model_type': 'BlueLM',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/BlueLM-7B-Base',
        'config_name': '/data/nlp/pre_models/torch/opt/BlueLM-7B-Base',
        'tokenizer_name': '/data/nlp/pre_models/torch/opt/BlueLM-7B-Base',
    },

    'BlueLM-7B-Base-32K': {
        'model_type': 'BlueLM',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/BlueLM-7B-Base-32K',
        'config_name': '/data/nlp/pre_models/torch/opt/BlueLM-7B-Base-32K',
        'tokenizer_name': '/data/nlp/pre_models/torch/opt/BlueLM-7B-Base-32K',
    },
    'XVERSE-13B-Chat': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat',
        'config_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B-Chat',
    },

    'xverse-13b-chat-int4': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/xverse-13b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-chat-int4',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-chat-int4',
    },

    'XVERSE-13B': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/XVERSE-13B',
        'config_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/XVERSE-13B',
    },

    'xverse-13b-int4': {
        'model_type': 'xverse',
        'model_name_or_path': '/data/nlp/pre_models/torch/xverse/xverse-13b-int4',
        'config_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-int4',
        'tokenizer_name': '/data/nlp/pre_models/torch/xverse/xverse-13b-int4',
    },
    'Skywork-13B-base': {
        'model_type': 'skywork',
        'model_name_or_path': '/data/nlp/pre_models/torch/skywork/Skywork-13B-base',
        'config_name': '/data/nlp/pre_models/torch/skywork/Skywork-13B-base',
        'tokenizer_name': '/data/nlp/pre_models/torch/skywork/Skywork-13B-base',
    },

    'Yi-6B': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/yi/Yi-6B',
    },

    'Yi-6B-200K': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/yi/Yi-6B-200K',
    },


    'Yi-34B': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/yi/Yi-34B',
    },

    'Yi-34B-200K': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/yi/Yi-34B-200K',
    },

    'Yi-34B-Chat': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/yi/Yi-34B-Chat',
    },


    'zephyr-7b-beta': {
        'model_type': 'mistral',
        'model_name_or_path': '/data/nlp/pre_models/torch/mistral/mistral-7b-sft-beta',
        'config_name': '/data/nlp/pre_models/torch/mistral/mistral-7b-sft-beta',
        'tokenizer_name': '/data/nlp/pre_models/torch/mistral/mistral-7b-sft-beta',
    },
}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING



TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["mistral"] = ["q_proj", "v_proj"]
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["lingowhale"] = ["qkv_proj"]



