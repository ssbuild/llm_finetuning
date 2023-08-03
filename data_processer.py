# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import random
import typing
from enum import Enum
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class DataStrategy(Enum):
    sup = 1
    unsup = 2
    sub_rounds = 3

class TokenIdsFinal:
    @classmethod
    def process(cls,tokenizer,input_ids,labels,max_seq_length):
        seqlen = np.asarray(len(input_ids), dtype=np.int32)
        pad_len = max_seq_length - seqlen
        input_ids = np.asarray(input_ids, dtype=np.int32)
        attention_mask = np.asarray([1] * len(input_ids), dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        if pad_len:
            pad_val = tokenizer.pad_token_id
            input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
            attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))
        d = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'seqlen': seqlen
        }
        return d

class TokenUnSupervision:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride, max_seq_length, examples):
        prefix,examples = examples
        input_ids_all = []
        if len(prefix) > 0:
            input_ids_all += tokenizer.encode(text=prefix)

        for idx, (question, answer) in enumerate(examples):
            text = question + answer
            ids = tokenizer.encode(text=text)
            if len(ids) <= 3:
                continue
            input_ids_all += ids

        # decoder_start_token_id = self.config.decoder_start_token_id
        decoder_start_token_id = config.bos_token_id
        pos = 0
        ds = []
        while pos < len(input_ids_all):
            input_ids = [decoder_start_token_id] + input_ids_all[pos: pos + max_seq_length - 1]
            pos += stride

            if len(input_ids) <= 5:
                continue

            d = TokenIdsFinal.process(tokenizer,input_ids,copy.deepcopy(input_ids),max_seq_length)
            ds.append(d)
        return ds


class TokenSupervision:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride, max_seq_length, examples):
        prefix, examples = examples
        ds = []
        for idx, (question, answer) in enumerate(examples):
            a_ids = tokenizer.encode(text=prefix+question,add_special_tokens=False)[:max_seq_length-3]
            b_ids = tokenizer.encode(text=answer)
            assert len(b_ids)
            input_ids_all = a_ids + b_ids
            labels_all = [-100] * len(a_ids) + b_ids
            pos = 0
            while pos < len(input_ids_all):
                input_ids = [config.bos_token_id] + input_ids_all[pos: pos + max_seq_length - 1]
                labels = [config.bos_token_id] + labels_all[pos: pos + max_seq_length - 1]
                pos += stride
                d = TokenIdsFinal.process(tokenizer, input_ids, labels, max_seq_length)
                ds.append(d)
        return ds

class TokenSupervisionRounds:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride, max_seq_length, examples):
        prefix, examples = examples
        ds = []
        prompt_text = ''
        for idx, (question, answer) in enumerate(examples):
            if idx == 0:
                a_text = question
            else:
                a_text = prompt_text + "[Round {}]\n问：{}\n答：".format(idx, question)

            prompt_text += "[Round {}]\n问：{}\n答：{}".format(idx, question, answer)
            a_ids = tokenizer.encode(text=prefix+a_text,add_special_tokens=False)[:max_seq_length-3]
            b_ids = tokenizer.encode(text=answer)


            assert len(b_ids)
            input_ids_all = a_ids + b_ids
            labels_all = [-100] * len(a_ids) + b_ids
            pos = 0
            while pos < len(input_ids_all):
                input_ids = [config.bos_token_id] + input_ids_all[pos: pos + max_seq_length - 1]
                labels = [config.bos_token_id] + labels_all[pos: pos + max_seq_length - 1]
                pos += stride
                d = TokenIdsFinal.process(tokenizer, input_ids, labels, max_seq_length)
                ds.append(d)
        return ds

