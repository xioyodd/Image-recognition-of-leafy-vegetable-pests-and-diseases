import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import json
from config import SAVE_DIR, DATA_DIR
import torch
from PIL import Image
from torchvision import transforms
from shutil import copy, rmtree

from model import resnet50,resnet34
import pandas as pd

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

def split_test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    test_path = os.path.join(DATA_DIR, 'test')
    img_path_list = os.listdir(test_path)

    print(img_path_list)
    img_list = []
    for img_path in img_path_list:
        img_path = os.path.join(test_path, img_path)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)
        img_list.append(img)

    # batch img
    batch_img = torch.stack(img_list, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=4).to(device)

    # load model weights
    # weights_path = "./resNet50.pth"
    weights_path = os.path.join(SAVE_DIR, 'resnet50_epoch50_20210814_151839', 'epoch20.pth')
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 数据根目录
    data_root = DATA_DIR
    assert os.path.exists(test_path), "path '{}' does not exist.".format(test_path)

    # 建立保存验证集的文件夹
    for cla in range(1,5):
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(data_root, str(cla)))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        for idx,(pro, cla) in enumerate(zip(probs, classes)):
            image_path = os.path.join(test_path, img_path_list[idx])
            new_path = os.path.join(data_root, class_indict[str(cla.numpy())])
            copy(image_path, new_path)

def predict(n_class,path,epoch,save_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    test_path = os.path.join(DATA_DIR, str(n_class))
    img_path_list = os.listdir(test_path)

    print(img_path_list)
    img_list = []
    for img_path in img_path_list:
        img_path = os.path.join(test_path, img_path)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)
        img_list.append(img)

    # batch img
    batch_img = torch.stack(img_list, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=3).to(device)

    # load model weights
    # weights_path = "./resNet50.pth"
    weights_path = os.path.join(SAVE_DIR, path, epoch)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        result = {'image_id': [], 'category_id': []}
        for idx, (pro, cla) in enumerate(zip(probs, classes)):
            result['image_id'].append(img_path_list[idx])
            result['category_id'].append(class_indict[str(cla.numpy())])

        dataframe = pd.DataFrame(result)
        dataframe.to_csv(os.path.join(SAVE_DIR, path, save_name), index=False, sep=',')

if __name__ == '__main__':
    #split_test()
    #predict(1,'resnet50_epoch70_20210814_104309','epoch10.pth','epoch10.csv')
    #predict(2, 'resnet50_epoch30_20210814_111622', 'epoch10.pth','epoch10.csv')
    predict(3, 'resnet50_epoch50_20210814_195629', 'epoch30.pth','epoch10.csv')
    predict(4, 'resnet50_epoch50_20210814_143545', 'epoch20.pth','epoch10.csv')
