import os
import json
from config import *
import torch
from PIL import Image
from torchvision import transforms

from model import resnet50
import pandas as pd


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(INPUT_SIZE),
         # transforms.CenterCrop(INPUT_SIZE),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    test_path = os.path.join(DATA_DIR, 'test210-detected-hsy')
    # test_path = os.path.join(DATA_DIR, 'test210-origin')
    # test_path = '/media/hexiang/4TDisk/python/pyProject/Image-recognition-of-leafy-vegetable-pests-and-diseases/yolov5/runs/detect/test300/crops/fruit'
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
    model = resnet50(num_classes=3).to(device)

    # load model weights
    # weights_path = "./resNet50.pth"
    weights_path = os.path.join(SAVE_DIR, 'Resnet50_Epoch120_20210819_190502_detectedhsy_BSize32', 'best70.pth')
    # weights_path = os.path.join(SAVE_DIR, 'epoch30.pth')
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
            # print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
            #                                                  class_indict[str(cla.numpy())],
            #                                                  pro.numpy()))
            result['image_id'].append(img_path_list[idx])
            result['category_id'].append(class_indict[str(cla.numpy())])

        dataframe = pd.DataFrame(result)
        dataframe.to_csv(os.path.join(SAVE_DIR, 'Resnet50_Epoch120_20210819_190502_detectedhsy_BSize32', 'best70-detected.csv'), index=False, sep=',')


if __name__ == '__main__':
    main()
