# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import random
import typing
from enum import Enum
import numpy as np
from transformers import PreTrainedTokenizer


class DataStrategy(Enum):
    sliding = 1
    supervision = 2





class TokenSliding:

    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride, max_seq_length, examples):
        input_ids_all = []
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
            seqlen = np.asarray(len(input_ids), dtype=np.int32)
            pad_len = max_seq_length - seqlen
            input_ids = np.asarray(input_ids, dtype=np.int32)
            attention_mask = np.asarray([1] * len(input_ids), dtype=np.int32)
            labels = np.asarray(copy.deepcopy(input_ids), dtype=np.int32)
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
            ds.append(d)
        return ds




class TokenSupervision:
    @classmethod
    def process(cls, tokenizer: PreTrainedTokenizer,config,stride, max_seq_length, examples):
        raise NotImplemented
        # input_ids_all = []
        # for idx, (question, answer) in enumerate(examples):
        #     text = question + answer
        #     ids = tokenizer.encode(text=text)
        #     if len(ids) <= 3:
        #         continue
        #     input_ids_all += ids
        #
        # # decoder_start_token_id = self.config.decoder_start_token_id
        # decoder_start_token_id = config.bos_token_id
        # pos = 0
        # ds = []
        # while pos < len(input_ids_all):
        #     input_ids = [decoder_start_token_id] + input_ids_all[pos: pos + max_seq_length - 1]
        #     pos += stride
        #
        #     if len(input_ids) <= 5:
        #         continue
        #     seqlen = np.asarray(len(input_ids), dtype=np.int32)
        #     pad_len = max_seq_length - seqlen
        #     input_ids = np.asarray(input_ids, dtype=np.int32)
        #     attention_mask = np.asarray([1] * len(input_ids), dtype=np.int32)
        #     labels = np.asarray(copy.deepcopy(input_ids), dtype=np.int32)
        #     if pad_len:
        #         pad_val = tokenizer.pad_token_id
        #         input_ids = np.pad(input_ids, (0, pad_len), 'constant', constant_values=(pad_val, pad_val))
        #         attention_mask = np.pad(attention_mask, (0, pad_len), 'constant', constant_values=(0, 0))
        #         labels = np.pad(labels, (0, pad_len), 'constant', constant_values=(-100, -100))
        #     d = {
        #         'input_ids': input_ids,
        #         'attention_mask': attention_mask,
        #         'labels': labels,
        #         'seqlen': seqlen
        #     }
        #     ds.append(d)
        # return ds