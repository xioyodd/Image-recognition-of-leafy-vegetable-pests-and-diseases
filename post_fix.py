# https://blog.csdn.net/guoziqing506/article/details/52014506


import numpy as np
import csv
from config import *
import os

def main():
    csvFilePath = os.path.join(SAVE_DIR, 'Resnet50_Epoch200_20210818_171826_detected4259_BSize32', 'best98.csv')
    savePath = os.path.join(SAVE_DIR, 'Resnet50_Epoch200_20210818_171826_detected4259_BSize32', 'best98-fixed.csv')
    # csvFilePath = os.path.join(SAVE_DIR, 'Resnet50_Epoch120_20210819_190502_detectedhsy_BSize32', 'best70-origin.csv')
    # savePath = os.path.join(SAVE_DIR, 'Resnet50_Epoch120_20210819_190502_detectedhsy_BSize32', 'best70-origin-fixed.csv')
    allDataPath = os.path.join(DATA_DIR, 'test210-origin')

    allDataList = os.listdir(allDataPath)
    # print(allDataList)
    # print(len(allDataList))
    # print(len('00d5aa3e-9c60-42b3-894d-0b83727e9293'))
    length = 36

    # allDataLen = list(map(lambda x: len(x), allDataList))
    # for i in allDataLen:
    #     if i != 40:
    #         print(i)
    # 有一个长度不对,手动做ba, 0tafwugmn0jm5jfow76q8pjnh1gklicq.jpg
    # allDataList = list(map(lambda x: x[0:length], allDataList))
    # allDataList = list(map(lambda x: x[:-4], allDataList))
    print(allDataList)

    csvFile = open(csvFilePath, 'r')
    reader = csv.reader(csvFile)

    content = {}

    for row in reader:
        if reader.line_num == 1:
            continue
        img = row[0][0:length]
        if img not in content:
            # print(row)
            content[img] = []
        content[img].append(row[1])
    print(content)

    csvFile.close()

    csvFile = open(savePath, 'w')
    writer = csv.writer(csvFile)
    writer.writerow(['image_id', 'category_id'])
    for i in content:
        # print(content[i])
        tmpL = [i+'.jpg', vs(content[i])]
        print(tmpL)
        writer.writerow(tmpL)

    csvFile.close()

def vs(a):
    counts = np.bincount(a)
    res = np.argmax(counts)
    return res


if __name__ == '__main__':
    main()
    # a = [1,2,3,4,5,5,5,5,6]
    # # max_a = max(set(a), key=a.count())
    # # max_a = max(a, key=a.count())
    # # max_a = max(a, key=lambda x: x-1)
    # counts = np.bincount(a)
    # max_a = np.argmax(counts)
    # print(max_a)