import torch
import torchvision.models as models
import torch.nn as nn
import cv2
from transformers import BertTokenizer

import config as cfg


class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(num_embeddings=432, embedding_dim=768)
        self.resnet18 = models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, 768)

    # img_tensor.shape = batch*48*48*3
    def forward(self, img_tensor):
        img_tensor.transpose_(1, 3)
        patch_ebds = self.resnet18(img_tensor)
        patch_ebds2 = self.linear(patch_ebds)

        pos_ebd = self.pos_embedding(torch.arange(432))
        return pos_ebd + patch_ebds2


class TextEmbedding(nn.Module):
    def __init__(self, pre_train_path='/data/yangdongquan/pre_trained_model/bert-base-chinese'):
        super(TextEmbedding, self).__init__()
        sd = torch.load(pre_train_path + '/pytorch_model.bin')
        ebd_dict = sd['bert.embeddings.word_embeddings.weight']
        self.text_embedding = text_embedding = nn.Embedding.from_pretrained(ebd_dict, freeze=False)

    def forward(self, tokenids):
        token_ebd = self.text_embedding(tokenids)
        return token_ebd


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    transformer_model = nn.Transformer(d_model=768)
