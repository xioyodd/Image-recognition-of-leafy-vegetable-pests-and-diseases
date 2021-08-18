import train,detect

if __name__ == '__main__':

    # train
    # data: test300
    train.run(data='data/leaf.yaml', weights='yolov5s.pt', batch=24, epochs=100)

    # data test300 共用机
    # train.run(data='data/leaf.yaml', weights='yolov5l.pt', batch=24, epochs=100)


    # detect
    # detect.run_detect(source='../data/test-detect/', weights='runs/train/exp3/weights/best.pt', save)