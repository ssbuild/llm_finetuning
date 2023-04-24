## 更新详情
    4月23日 增加lora merge权重（修改infer_lora_finetuning.py enable_merge_weight 选项）
    4月11日 升级 lora v2 , 增加adalora


## 1.安装
  - pip install -i https://pypi.org/simple -U deep_training>=0.1.2 transformers>=4.28 deepspeed



## 2.更新详情
- [deep_training](https://github.com/ssbuild/deep_training)

## 3.深度学习常规任务例子

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)



## 4. bloom 预训练模型下载，大小尺寸可自行选择
    
- [bloom预训练模型](https://huggingface.co/bigscience)
- [bloom第三方中文训练模型](https://huggingface.co/Langboat/bloom-6b4-zh)  # 注意 需要修改tokenizer_config.json BloomTokenizer -> BloomTokenizerFast
- [opt预训练模型](https://huggingface.co/facebook)
- [llama预训练模型](https://huggingface.co/decapoda-research) #  llama 词典等下载地址 https://huggingface.co/hf-internal-testing/llama-tokenizer


## 5.数据示例

[中文第三方数据集](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)


数据示例1
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
    或者
```json
 {
     "id": 0, "paragraph": [
         {
             "q": "从南京到上海的路线",
             "a": [
                 "你好，南京到上海的路线如下：",
                 "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
                 "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
                 "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
             ]
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



## 6.生成训练record

    python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## 7.推理
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer_finetuning.py



   


## 8.训练

    python train.py

```text
多机多卡训练 例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


## 9.是否开启lora finetuning

    with_lora


## 10.是否开启deepspeed
    启动则将data_utils.py  修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/llm_finetuning&type=Date)](https://star-history.com/#ssbuild/llm_finetuning&Date)