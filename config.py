import os

WORKSPACE = os.getcwd()
SAVE_DIR = './model/'
DATA_DIR = './data/'
BATCH_SIZE = 32
EPOCH = 50
SEED = 2021


#################################
INPUT_SIZE = (448, 448)  # (w, h)
RESIZE_SIZE = (512, 512)
SAVE_FREQ = 10
TEST_FREQ = 1
# RESUME = './model/agriculture_model1_20210721_124231/model.ckpt'
RESUME = ''
MIXUP = 0.

USE_MIXUP = True,  # 是否使用mixup方法增强数据集
MIXUP_ALPHA = 0.5,  # add mixup alpha ,用于beta分布




