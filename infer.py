# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import re
from collections import OrderedDict
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from models import MyTransformer, Generate,LoraArguments,PromptArguments

deep_config = get_deepspeed_config()

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})
    pl_model = MyTransformer(config=config, model_args=model_args)
    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    text= "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    response, history = Generate.chat(model,query=text,tokenizer=tokenizer,max_length=512,
                                        eos_token_id=config.eos_token_id,
                                        do_sample=True, top_p=0.7, temperature=0.95,)
    print('input',text)
    print('output',response)