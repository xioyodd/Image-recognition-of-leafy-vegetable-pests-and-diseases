import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import json
from config import SAVE_DIR, DATA_DIR
import torch
from PIL import Image
from torchvision import transforms

from model import resnet50
import pandas as pd


def main():
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
    model = resnet50(num_classes=3).to(device)

    # load model weights
    # weights_path = "./resNet50.pth"
    weights_path = os.path.join(SAVE_DIR, 'resnet50_epoch100_20210806_190942', 'epoch30.pth')
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
        dataframe.to_csv(os.path.join(SAVE_DIR, 'resnet50_epoch100_20210806_190942', 'epoch30.csv'), index=False, sep=',')


if __name__ == '__main__':
    main()
