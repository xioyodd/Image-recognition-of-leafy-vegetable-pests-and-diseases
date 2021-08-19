import os

data_path = '/media/hexiang/4TDisk/python/pyProject/Agriculture/database/train-detected-l-4259'

classes = ['1', '2', '3']

for i in classes:
    class_path = os.path.join(data_path, i)
    print(os.listdir(class_path))
