# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.lora import LoraArguments
from transformers import HfArgumentParser,AutoConfig,PreTrainedTokenizer

from data_utils import train_info_args, NN_DataHelper, postprocess
from models import MyTransformer, Generate


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()

    config = AutoConfig.from_pretrained('./best_ckpt')
    lora_args = LoraArguments.from_pretrained('./best_ckpt')

    assert lora_args.inference_mode == True

    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args)
    # 加载lora权重
    pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path = './best_ckpt', lora_config = lora_args)

    model = pl_model.get_llm_model()

    model.eval()
    model.cuda()

    text = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
    output = Generate.generate(model, query=text, tokenizer=tokenizer, max_length=512,
                               eos_token_id=config.eos_token_id,
                               do_sample=True, top_p=0.7, temperature=0.95, )
    print('input', text)
    print('output', output)