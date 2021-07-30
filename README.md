# Image-recognition-of-leafy-vegetable-pests-and-diseases



## 数据搜集:



> PRCV2019
>
> 1、各参赛队在赛前需签订数据使用协议，承诺本竞赛提供的数据集仅能用于本竞赛，不用于除本竞赛外的任何其他用途，并承诺数据用后即刻删除，不可扩散，主办方保留追究法律责任的权利。









## Todo list(计划, 待补充)

- [x] baseline搭建
- [ ] 数据增强
  - [ ] 数据统计
  - [ ] 切割, 翻转, 等
  - [ ] 网络图片
  - [ ] 合成图片
- [ ] 模型改进
  - [ ] 权重衰减, dropout, bn等
  - [ ] 网络改进
- [ ] 训练策略改进
  - [ ] k折交叉验证
  - [ ] 动态学习率
- [ ] 其他
  - [ ] 算法可复现

## Structre

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
├── crawler2.py 			#爬取数据
├── resnet50-pre.pth
└── ...
```

## Install

1. git clone

## Usage

train.py for training

batch_predict.py for result

crawler2.py 用来 爬取百度图片, 使用时修改结尾几行


## Links

[比赛官网](https://challenge.xfyun.cn/topic/info?type=pests-diseases)

[腾讯文档](https://docs.qq.com/doc/DQ1FrUGFJeUxaTlhq)

[git/github教程1](https://www.liaoxuefeng.com/wiki/896043488029600/900375748016320)

[git-cheat-sheet](https://github.com/flyhigher139/Git-Cheat-Sheet)

[github-cheat-sheet](https://github.com/tiimgreen/github-cheat-sheet)

[pytorch官网](https://pytorch.org/)

