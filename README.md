# Image-recognition-of-leafy-vegetable-pests-and-diseases



## 结构

```
.
├── data
│   ├── test
│   ├── train
│   └── val
│       ├── 1
│       ├── 2
│       ├── 3
│           └── .jpg/png
│       └── ...
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

## 安装

1. git clone 

## 用法

config.py 定义参数

train.py 训练模型

batch_predict.py 预测结果

cam_resnet.py 用来生成热力图






## 相关链接

[比赛官网](https://challenge.xfyun.cn/topic/info?type=pests-diseases)

[git/github教程1](https://www.liaoxuefeng.com/wiki/896043488029600/900375748016320)

[git-cheat-sheet](https://github.com/flyhigher139/Git-Cheat-Sheet)

[github-cheat-sheet](https://github.com/tiimgreen/github-cheat-sheet)

[pytorch官网](https://pytorch.org/)

[图像分类任务中的训练奇技淫巧](https://zhuanlan.zhihu.com/p/149789219)

[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

