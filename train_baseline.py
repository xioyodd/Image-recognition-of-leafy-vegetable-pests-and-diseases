import os
import json
from datetime import datetime
from shutil import copyfile

import torchvision.models

from config import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet50
# from torchvision.models import resnet50
import logging
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def main():
    epochs = EPOCH
    msg = 'detectedALL_BSize32'
    save_dir = os.path.join(SAVE_DIR,
                            'Resnet50' + '_' + 'Epoch' + str(epochs) + '_' + datetime.now().strftime(
                                '%Y%m%d_%H%M%S') + '_' + msg)
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    copyfile('./train.py', save_dir + '/train.py')
    copyfile('model.py', save_dir + '/model.py')
    copyfile('config.py', save_dir + '/config.py')

    # 通过下面的方式进行简单配置输出方式与日志级别
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(save_dir, 'logger.log'),
                        level=logging.INFO,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    _print = logging.info

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(224),
            # transforms.RandomCrop(224),
            # transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = DATA_DIR
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train-detected"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = BATCH_SIZE
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    _print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train-detected"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    _print("using {} images for training, {} images for validation.".format(train_num,
                                                                            val_num))

    net = resnet50()
    # load pretrain weights
    model_weight_path = "./resnet50-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 3)
    # net.fc = nn.Sequential(
    #     nn.Linear(in_channel, 512),
    #     nn.BatchNorm1d(512),
    #     nn.Dropout(0.5),
    # )
    # net.fc = nn.Sequential(
    #     nn.Linear(in_channel, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, 3)
    # )
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0003)

    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    all_train_losses = []
    all_valid_losses = []
    all_train_acc = []
    all_valid_acc = []
    for epoch in range(epochs):
        # train
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()

            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
            #                                                          epochs,
            #                                                          loss)
            train_bar.desc = "train epoch[{}/{}]".format(epoch + 1, epochs)

            predict_y1 = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y1, labels.to(device)).sum().item()

        train_accurate = train_acc / train_num

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        _print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
               (epoch + 1, train_loss / train_steps, train_accurate, val_loss / val_steps,  val_accurate))
        all_train_losses.append(train_loss / train_steps)
        all_valid_losses.append(val_loss / val_steps)
        all_train_acc.append(train_accurate)
        all_valid_acc.append(val_accurate)

        # save

        if val_accurate > best_acc:
            best_acc = val_accurate
            if epoch > 15:
                torch.save(net.state_dict(), os.path.join(save_dir, 'best' + str(epoch + 1) + '.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(save_dir, 'best.pth'))
        if (epoch + 1) % SAVE_FREQ == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

    _print(all_train_losses, all_valid_losses, all_train_acc, all_valid_acc, save_dir)
    showResult(all_train_losses, all_valid_losses, all_train_acc, all_valid_acc, save_dir)
    _print('Finished Training')

def showResult(all_train_losses, all_valid_losses, all_train_acc, all_valid_acc, save_dir):
    # 创建第一张画布
    plt.figure()

    # 绘制训练损失曲线
    plt.plot(all_train_losses, label="Train Loss")
    # 绘制验证损失曲线, 颜色为红色
    plt.plot(all_valid_losses, color="red", label="Valid Loss")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(10)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, EPOCH)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig(os.path.join(save_dir, 'loss.png'))

    # 创建第二张画布
    plt.figure()

    # 绘制训练准确率曲线
    plt.plot(all_train_acc, label="Train Acc")
    # 绘制验证准确率曲线, 颜色为红色
    plt.plot(all_valid_acc, color="red", label="Valid Acc")
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(10)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标取值范围
    plt.xlim(1, EPOCH)
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig(os.path.join(save_dir, 'acc.png'))


if __name__ == '__main__':
    main()