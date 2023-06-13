## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

```text
    06-13 fix llama resize_token_embeddings
    06-01 support deepspeed training for lora adalora prompt,0.1.9 和 0.1.10合并
    05-27 add qlora transformers>=4.30
    05-24 fix p-tuning-v2 load weight bugs
    05-12 fix lora int8 多卡训练 , ppo training move to https://github.com/ssbuild/rlhf_llm
    05-02 增加p-tuning-v2
    04-28 deep_training 0.1.3 pytorch-lightning 改名 ligntning ，旧版本 deep_training <= 0.1.2
    04-23 增加lora merge权重（修改infer_lora_finetuning.py enable_merge_weight 选项）
    04-11 升级 lora , 增加adalora
```
   

## install
  - pip install -i https://pypi.org/simple -U -r requirements.txt



## weigtht select one is suitable for you
    
- [bloom预训练模型](https://huggingface.co/bigscience)
- [bloom第三方中文训练模型](https://huggingface.co/Langboat/bloom-6b4-zh)  # 注意 需要修改tokenizer_config.json BloomTokenizer -> BloomTokenizerFast
- [tigerbot](https://huggingface.co/TigerResearch)
- [opt预训练模型](https://huggingface.co/facebook)
- [llama 官方权重转换](https://huggingface.co/decapoda-research) #  llama 词典等下载地址 https://huggingface.co/hf-internal-testing/llama-tokenizer
- [llama vicuna-7B第三方权重1](https://huggingface.co/TheBloke/vicuna-7B-1.1-HF)
- [llama vicuna-7B第三方权重2](https://huggingface.co/Tribbiani/vicuna-7b)
- [cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b)
- [rwkv](https://huggingface.co/BlinkDL) 需要转换权重
- [rwkv](https://huggingface.co/RWKV/rwkv-4-169m-pile) hf 权重


## data sample

[中文第三方数据集](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)


数据示例
```json
 {
     "id": 0, "paragraph": [
         {
             "q": "从南京到上海的路线",
             "a":  "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
         }
     ]
 }
```

多轮会话
```json
 {
     "id": 0, "paragraph": [
        {
           "q": "你好",
           "a": "我是机器人，有什么可以帮助你的？"
        },
         {
             "q": "从南京到上海的路线",
             "a":  "你好，南京到上海的路线如下：1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
         }
     ]
 }

```



## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
    # infer_ptuning.py 推理p-tuning-v2微调模型
     python infer_finetuning.py



## training
```text
    #制作数据
    python data_utils.py
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    #训练
    python train.py
```
   
## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatyuan_finetuning](https://github.com/ssbuild/chatyuan_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [rlhf_llm](https://github.com/ssbuild/rlhf_llm)
- [rlhf_chatglm](https://github.com/ssbuild/rlhf_chatglm)
- [rlhf_chatyuan](https://github.com/ssbuild/rlhf_chatyuan)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)

## 
    纯粹而干净的代码

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/llm_finetuning&type=Date)](https://star-history.com/#ssbuild/llm_finetuning&Date)