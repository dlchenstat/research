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

    def forward(self, img_path):
        patch_list = self.get_patchs(img_path)
        img_tensor = torch.tensor(patch_list, dtype=torch.float32)
        img_tensor.transpose_(1, 3)
        patch_ebds = self.resnet18(img_tensor)
        patch_ebds2 = self.linear(patch_ebds)

        pos_ebd = self.pos_embedding(torch.arange(len(patch_list)))
        return pos_ebd + patch_ebds2

    def get_patchs(self, img_path, fixed_size=(768, 1296), patch_size=48):
        patch_list = []

        matrix = cv2.imread(img_path)
        m = cv2.resize(matrix, fixed_size)

        height = fixed_size[1]
        width = fixed_size[0]

        number_y = int(height / patch_size)
        number_x = int(width / patch_size)

        for y in range(number_y):
            for x in range(number_x):
                patch = m[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size, :]
                patch_list.append(patch)
        return patch_list


class TextEmbedding(nn.Module):
    def __init__(self, pre_train_path='/data/yangdongquan/pre_trained_model/bert-base-chinese'):
        super(TextEmbedding, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pre_train_path)
        # 加载BERT的embeddings
        sd = torch.load(pre_train_path + '/pytorch_model.bin')
        ebd_dict = sd['bert.embeddings.word_embeddings.weight']
        self.text_embedding = text_embedding = nn.Embedding.from_pretrained(ebd_dict, freeze=False)

    def forward(self, text):
        text_list = [t for t in text]
        tokenids = self.tokenizer.encode(text_list, truncation=True, max_length=1000, padding='max_length',
                                         return_tensors='pt')
        tokenids = tokenids[0]
        token_ebd = self.text_embedding(tokenids)

        pad = torch.tensor([self.tokenizer.pad_token_id])
        labels = torch.cat((tokenids[1:], pad))
        return token_ebd, labels


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

    def forward(self):
        pass
