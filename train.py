import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import json
from datetime import datetime
from shutil import copyfile
from config import *
from mixup import *

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet50
import logging

def main():
    save_dir = os.path.join(SAVE_DIR, 'resnet50' + '_' + 'epoch'+ str(EPOCH) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
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
        "train": transforms.Compose([#transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),# 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小224*224
                                     transforms.RandomHorizontalFlip(),#以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
                                     transforms.ToTensor(),#将给定图像转为Tensor
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = DATA_DIR
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
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

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    _print("using {} images for training, {} images for validation.".format(train_num,
                                                                            val_num))

    net = resnet50()
    # load pretrain weights
    model_weight_path = "./resnet50-pre.pth"  #官方预训练的权重
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device)) #将权重加载到net中
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 3)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001,weight_decay=1e-3)

    elist=[]
    loss_list=[]
    acc_list=[]
    epochs = EPOCH
    best_acc = 0.0
    save_path = './resNet50.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        elist.append(epoch)
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        # for step, (inputs, targets) in enumerate(train_bar):
        #     inputs,targets=inputs.cuda(),targets.cuda()
        #     inputs,targets_a,targets_b,lam=criterion(inputs,targets,0.5,USE_MIXUP)
        #     outputs=net(inputs)
        #     loss=mixup_criterion(loss_function,outputs,targets_a,targets_b,lam)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        for step,data in enumerate(train_bar):
            images,labels=data
            optimizer.zero_grad()
            logits=net(images.to(device))
            loss=loss_function(logits,labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        _print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
               (epoch + 1, running_loss / train_steps, val_accurate))
        loss_list.append(running_loss / train_steps)
        acc_list.append(val_accurate)
        # save

        if val_accurate > best_acc:
            best_acc = val_accurate
            # torch.save(net.state_dict(), os.path.join(save_dir, 'best' + str(epoch+1) + '.pth'))
            torch.save(net.state_dict(), os.path.join(save_dir, 'best.pth'))
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

    _print('Finished Training')

    plt.title("TrainInfo")
    plt.scatter(elist,loss_list,color="red",label="train_loss")
    plt.scatter(elist, acc_list, color="blue", label="val_accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.savefig(os.path.join(save_dir, 'trainInfo'))
    plt.show()

if __name__ == '__main__':
    main()
