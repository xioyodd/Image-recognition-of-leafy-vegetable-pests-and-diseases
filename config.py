import os

WORKSPACE = os.getcwd()
SAVE_DIR = './model/'
DATA_DIR = './data/'
BATCH_SIZE = 32
EPOCH = 120
SEED = 2021
SAVE_FREQ = 10
TEST_FREQ = 1
INPUT_SIZE = (224, 224)  # (w, h)
RESIZE_SIZE = (256, 256)
# INPUT_SIZE = (448, 448)  # (w, h)
# RESIZE_SIZE = (512, 512)


#################################


# RESUME = './model/agriculture_model1_20210721_124231/model.ckpt'
RESUME = ''
MIXUP = 0.




