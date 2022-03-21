from transformers import BertConfig
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 数据设置
PRE_TRAIN_PATH = '/data/yangdongquan/pre_trained_model/bert-base-chinese'
bc = BertConfig.from_pretrained(PRE_TRAIN_PATH)
VOCAB_SIZE = bc.vocab_size
MAX_LEN = 100
BATCH_SIZE = 16

# training setting
TRAIN_EPOCH = 10
LEARNING_RATE = 1e-5
DEVICE = 'cuda'
DEVICE = 'cpu'
