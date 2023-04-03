# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import os
import re
from collections import OrderedDict

# @Time    : 2023/3/19 18:15
# @Author  : tk
# @FileName: infer
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.lora import LoraArguments
from transformers import HfArgumentParser,AutoConfig,AutoConfig

from data_utils import train_info_args, postprocess, NN_DataHelper, get_deepspeed_config
from models import MyTransformer, Generate

deep_config = get_deepspeed_config()

if __name__ == '__main__':

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)



    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()

    ###################### 注意 选最新权重
    # 选择最新的权重 ， 根据时间排序 选最新的
    config = AutoConfig.from_pretrained('./best_ckpt')
    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
        assert os.path.exists(train_weight)
        pl_model = MyTransformer.load_from_checkpoint(train_weight, config=config, model_args=model_args,
                                                      training_args=training_args, strict=False)
    else:

        # 建议直接使用转换脚本命令 支持 deepspeed stage 0,1,2,3， 生成 ./best_ckpt/last.ckpt/best.pt 权重文件
        # cd best_ckpt/last.ckpt
        # python zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last.ckpt/best.pt'

        # deepspeed stage 0,1,2 不必须执行上面命令
        # train_weight = './best_ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'

        assert os.path.exists(train_weight)
        weights_dict = torch.load(train_weight)
        weights_dict_new = OrderedDict()
        for k, v in (weights_dict['module'] if 'module' in weights_dict else weights_dict).items():
            weights_dict_new[re.sub(r'_forward_module\.', '', k)] = v
        pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args)
        pl_model.load_state_dict(state_dict=weights_dict_new, strict=False)

    model = pl_model.get_llm_model()



    model.eval()
    model.cuda()

    text= "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    output = Generate.generate(model,query=text,tokenizer=tokenizer,max_length=512,
                                        eos_token_id=config.eos_token_id,
                                        do_sample=True, top_p=0.7, temperature=0.95,)
    print('input',text)
    print('output',output)