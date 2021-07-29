# Image-recognition-of-leafy-vegetable-pests-and-diseases



## structre

```
.
├── data
│   ├── test
│   ├── train-all			#用于训练 预测提交答案 的模型
│   ├── train
│   └── val
│       ├── 1
│       ├── 2
│       └── 3
│           └── .jpg
├── model
│   ├── 结果
│   │   ├── .pth
│   │   ├── logger.log
│   │   └── ...
│   └── ...
├── README.md
├── config.py
├── model.py
├── train.py
├── batch_predict.py
├── split_data.py  			#把trian-all分为train和val, 随机
├── class_indices.json
├── main.py
├── resnet50-pre.pth
└── ...
```

## Install

1. git clone

## Usage

train.py for training

batch_predict.py for result


## Links

[比赛官网](https://challenge.xfyun.cn/topic/info?type=pests-diseases)

[腾讯文档](https://docs.qq.com/doc/DQ1FrUGFJeUxaTlhq)

[git/github教程1](https://www.liaoxuefeng.com/wiki/896043488029600/900375748016320)

[pytorch官网](https://pytorch.org/)

