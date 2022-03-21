import torch
import torchvision.models as models
import torch.nn as nn
from data_process.build_dataloader import get_ima_txt_data_loader
import config as cfg


class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(num_embeddings=432, embedding_dim=768)
        self.resnet18 = models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, 768)

    # img_tensor.shape = batch*432(图片拆分块数)*48*48*3
    # return batch*432(图片拆分块数)*768
    def forward(self, img_tensor):
        img_tensor = img_tensor.permute(0, 1, 4, 2, 3)
        batch_len_ebd = torch.zeros((cfg.BATCH_SIZE, 432, 768))
        for i, p in enumerate(img_tensor):
            ebd = self.deal_one_page(p)
            batch_len_ebd[i] = ebd

        return batch_len_ebd

    def deal_one_page(self, patch):
        patch_ebds = self.resnet18(patch)
        patch_ebds2 = self.linear(patch_ebds)

        pos_ids = torch.arange(432).to(cfg.DEVICE)
        pos_ebd = self.pos_embedding(pos_ids)
        return pos_ebd + patch_ebds2


class TextEmbedding(nn.Module):
    def __init__(self):
        super(TextEmbedding, self).__init__()
        sd = torch.load(cfg.PRE_TRAIN_PATH + '/pytorch_model.bin')
        ebd_dict = sd['bert.embeddings.word_embeddings.weight']
        self.text_embedding = nn.Embedding.from_pretrained(ebd_dict, freeze=False)

    # token_ids.shape = batch*len
    # return batch*len*768
    def forward(self, token_ids):
        token_ebd = self.text_embedding(token_ids)
        return token_ebd


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.img_embedding = ImageEmbedding()
        self.text_embedding = TextEmbedding()
        self.tf = nn.Transformer(d_model=768)
        # 21128 词典的size
        self.linear = nn.Linear(768, cfg.VOCAB_SIZE)

    def forward(self, patch, token_ids):
        img_ebd = self.img_embedding(patch)
        txt_ebd = self.text_embedding(token_ids)
        # batch 和 len 交换维度
        img_ebd = img_ebd.transpose(0, 1)
        txt_ebd = txt_ebd.transpose(0, 1)
        out_ebd = self.tf(img_ebd, txt_ebd)
        logits = self.linear(out_ebd)
        return logits


if __name__ == '__main__':
    transformer_model = TransformerModel()
    dl = get_ima_txt_data_loader()
    for batch_patch, batch_tokens, batch_labels in dl:
        out = transformer_model(batch_patch, batch_tokens)
        print(out.shape)
        break
