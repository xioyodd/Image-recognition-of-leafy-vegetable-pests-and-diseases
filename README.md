# Image-recognition-of-leafy-vegetable-pests-and-diseases





## Detect

[yolov5](https://github.com/ultralytics/yolov5)

###  标注 

##### 工具

[labelImg](https://github.com/tzutalin/labelImg)

##### 安装(ubuntu)

```
pip3 install labelImg
labelImg
```

##### 策略

打开后选择 300张test图片的目录, 存放目录选择上面的labels/test100, 可以先看看怎么定义各个部位的,特别是茎的标注, 我前100张叶的部分没有标注的很仔细, 导致叶的训练效果没那么好.

把剩下200张图片标一下, 主要是有些叶子不太好标注完整, 看着来. 标注尽量宁缺勿滥, 主要是为了训练出自动检测病变部位或者主要部位的模型, 为图片分类服务.

















## 数据搜集:



> PRCV2019
>
> 1、各参赛队在赛前需签订数据使用协议，承诺本竞赛提供的数据集仅能用于本竞赛，不用于除本竞赛外的任何其他用途，并承诺数据用后即刻删除，不可扩散，主办方保留追究法律责任的权利。









## Todo list(计划, 待补充)

- [x] baseline搭建
- [ ] 数据增强
  - [ ] 数据统计
  - [x] ~~切割, 翻转, 等~~
  - [ ] 网络图片
    - [x] 百度图片
    - [ ] 谷歌图片
  - [ ] 合成图片
  - [ ] 一份原始图片一份手动裁剪
  - [ ] 手动分部位分模型
- [ ] 模型改进
  - [ ] 权重衰减, dropout, bn等
  - [ ] 网络改进
- [ ] 训练策略改进
  - [ ] k折交叉验证
  - [ ] 动态学习率
- [ ] 其他
  - [ ] 算法可复现
  - [ ] 输出训练集准确率
  - [ ] 多模型融合可以尝试
  - [ ] 三种病, 4个部位(5个? 叶的远近景),分12类,分开来看各个类别的分类效果, 具体情况具体分析. 也许有新思路

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

cam_resnet.py 用来生成热力图, pytorch-grad-cam

## 手动处理方法

#### 删除1

```python
lst = [
    'a52d07ac-7d01-429f-84ec-43329a4c474f.jpg',
	'fb337e4e-9d0f-43ec-82e8-5d75fd3a6eb5.jpg',
	'591ed2b3-0092-4210-be2d-26c7779342c4.jpg',
]


```

#### 删除2

```python
lst = [
    '58b73254-54f2-4462-8cad-2c78d0f0a42e.jpg',
	'121d9fb2-a8c9-4b35-a345-9e1f7f73168a.jpg',
	'486b7e20-df1c-4dcc-8a33-8b255ca6b801.jpg',
	'953818b4-2bfe-4371-b810-d49502441d81.jpg',
	'ec92af24-6d47-4f3a-8766-a62833385f0b.jpg',
]

```



#### 删除3

```python
lst = [
    '1c20961d-e8fd-4c1e-a392-b5f3b7765656.jpg',
    '24aa1602-c5d6-45fc-9c55-735376b7efed.jpg',
    '91a61d18-2f9a-4097-8912-de20ad0a73ee.jpg',
    'a957a60b-e4bd-4269-a909-5f10ed9a7080.jpg',
    
]
```

#### 手动裁剪水印等




## Links

[比赛官网](https://challenge.xfyun.cn/topic/info?type=pests-diseases)

[腾讯文档](https://docs.qq.com/doc/DQ1FrUGFJeUxaTlhq)

[git/github教程1](https://www.liaoxuefeng.com/wiki/896043488029600/900375748016320)

[git-cheat-sheet](https://github.com/flyhigher139/Git-Cheat-Sheet)

[github-cheat-sheet](https://github.com/tiimgreen/github-cheat-sheet)

[pytorch官网](https://pytorch.org/)

[图像分类任务中的训练奇技淫巧](https://zhuanlan.zhihu.com/p/149789219)

[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

