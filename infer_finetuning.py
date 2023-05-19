# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import re
from collections import OrderedDict
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from models import MyTransformer, Generate,LoraArguments,PromptArguments

deep_config = get_deepspeed_config()

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()

    config = AutoConfig.from_pretrained('./best_ckpt')
    pl_model = MyTransformer(config=config, model_args=model_args, strict=False)

    ###################### 注意 选最新权重
    # 选择最新的权重 ， 根据时间排序 选最新的

    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
        assert os.path.exists(train_weight)

    else:
        # 建议直接使用转换脚本命令 支持 deepspeed stage 0,1,2,3， 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'

    pl_model.load_sft_weight(train_weight)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()

    text= "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    response, history = Generate.chat(model, query=text, tokenizer=tokenizer, max_length=512,
                                      eos_token_id=config.eos_token_id,
                                      do_sample=True, top_p=0.7, temperature=0.95, )
    print('input',text)
    print('output',response)