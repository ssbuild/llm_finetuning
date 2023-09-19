# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
from enum import Enum
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class DataStrategy(Enum):
    tunction = 1
    slidding = 2

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





def build_template_default(query, answer = None, history=None):
    prompt = ''
    if history is not None:
        for q,a in history:
            prompt += "User: {}\nAssistant:{}".format(q,a)
    prompt += "User: {}\nAssistant:".format(query)
    if answer is not None:
        prompt += answer
    return prompt

def build_template_tiger(query,answer = None, history=None):
    prompt = ''
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    if history is not None:
        for q,a in history:
            prompt += "{}{}{}{}".format(tok_ins,q,tok_res,a)

    prompt += "{}{}{}".format(tok_ins, query, tok_res)
    if answer is not None:
        prompt += answer
    return prompt


#切换模板
build_template = build_template_default

class TokenTunction:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer, config, sup,ensure_answer_min_length, max_seq_length, examples):
        ds = []
        prefix, examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids, b_ids = [], []
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a)[:max_seq_length - 3 - ensure_answer_min_length] + [config.eos_token_id]
            a_len = max_seq_length - len(b_ids) - 1
            input_ids = a_ids[-a_len:] + b_ids
            if sup:
                labels = [-100] * a_len + input_ids[a_len:]
            else:
                labels = copy.deepcopy(input_ids)
            input_ids = [config.bos_token_id] + input_ids
            labels = [-100] + labels if sup else [config.bos_token_id] + labels

            if len(input_ids) <= 2:
                continue

            ds.append(TokenIdsFinal.process(tokenizer, input_ids, labels, max_seq_length))
        return ds


class TokenSlidding:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride,sup, max_seq_length, examples):
        ds = []
        prefix,examples = examples
        for sid, (q, a) in enumerate(examples):
            a_ids,b_ids = [],[]
            if len(prefix) > 0:
                a_ids += tokenizer.encode(text=prefix, add_special_tokens=False)

            a_ids += tokenizer.encode(text=build_template(q, history=examples[:sid]), add_special_tokens=False)
            b_ids = tokenizer.encode(text=a) + [config.eos_token_id]

            input_ids_all = a_ids + b_ids
            labels_all = [-100] * len(a_ids) + b_ids if sup else copy.deepcopy(input_ids_all)
            if len(input_ids_all) <= 2:
                continue

            pos = 0
            while pos < len(input_ids_all):
                input_ids = [config.bos_token_id] + input_ids_all[pos: pos + max_seq_length - 1]
                labels = [-100] + labels_all[pos: pos + max_seq_length - 1] if sup else [config.bos_token_id] + labels_all[pos: pos + max_seq_length - 1]
                pos += stride
                if np.all(np.asarray(labels) == -100):
                    continue
                ds.append(TokenIdsFinal.process(tokenizer, input_ids, labels, max_seq_length))
        return ds


