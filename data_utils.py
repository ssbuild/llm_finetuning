# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py

import copy
import json
import os
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from models import LoraArguments,LoraConfig,PromptArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser
from data_processer import DataStrategy, TokenSupervision, TokenUnSupervision,TokenSupervisionRounds,\
    DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

from config import *

data_conf = {
    'strategy': DataStrategy.sup,  # 数据策略选项
    DataStrategy.sup: {
        'stride':  int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.unsup: {
        'stride':  int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.sub_rounds: {
        'stride': int(train_info_args['max_seq_length'] / 3 * 2),
    }

}



def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)
        assert data_conf[DataStrategy.sup]['stride'] > 0
        assert data_conf[DataStrategy.unsup]['stride'] > 0


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        examples = data


        strategy = data_conf['strategy']
        if strategy == DataStrategy.sup:
            ds = TokenSupervision.process(tokenizer, config=config,  max_seq_length=max_seq_length, examples=examples,**data_conf[strategy])
        elif strategy == DataStrategy.unsup:
            ds = TokenUnSupervision.process(tokenizer, config=config,  max_seq_length=max_seq_length, examples=examples, **data_conf[strategy])
        elif strategy == DataStrategy.sub_rounds:
            ds = TokenSupervisionRounds.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                            **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds

    # {
    #     "id": 0, "paragraph": [
    #     # 一轮会话
    #     {
    #         "q": "从南京到上海的路线",
    #         "a": [
    #             "你好，南京到上海的路线如下：",
    #             "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
    #             "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
    #             "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
    #         ]
    #     }
    #     # 二轮....
    # ]
    # }
    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                jd = json.loads(line)
                if not jd:
                    continue
                paragraph = jd['paragraph']
                if i < 10:
                    print(paragraph)
                sub = []
                # 自行做模板
                for session in paragraph:
                    q = session['q']
                    answers = preprocess('\n'.join(session['a']))  if isinstance(session['a'],list) else preprocess(session['a'])
                    # q = preprocess('Human：' + q + '\nAssistant：')
                    assert len(answers),ValueError('answer cannot empty')
                    sub.append((q, answers))
                D.append(copy.deepcopy(sub))
                sub.clear()

        return D

    def collate_fn(self, batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        maxlen = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :maxlen]
        o['attention_mask'] = o['attention_mask'][:, :maxlen]
        o['labels'] = o['labels'][:, :maxlen].long()
        return o






if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments,PromptArguments))
    model_args, training_args, data_args, _,_ = parser.parse_dict(train_info_args)


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})
    config.torch_dtype = "float16"
    config.decoder_start_token_id = config.bos_token_id


    if "llama" in model_args.model_name_or_path.lower() and tokenizer.bos_token_id != DEFAULT_BOS_TOKEN:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
            tokenizer.pad_token_id = tokenizer.eos_token_id


    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False,shuffle=True,mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, shuffle=False,mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, shuffle=False,mode='test')


    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
