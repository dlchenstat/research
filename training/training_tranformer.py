from data_process.build_dataloader import get_ima_txt_data_loader
from model_frame.modeling import TransformerModel
import config as cfg
from torch.optim import AdamW
import torch.nn as nn
import tqdm


def training(model, train_dataloader):
    model.to(cfg.DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(cfg.DEVICE)
    for epo in range(cfg.TRAIN_EPOCH):
        for batch_patch, batch_tokens, batch_labels in tqdm.tqdm(train_dataloader):
            batch_patch = batch_patch.to(cfg.DEVICE)
            batch_tokens = batch_tokens.to(cfg.DEVICE)
            batch_labels = batch_labels.to(cfg.DEVICE)
            model.zero_grad()

            out = model(batch_patch, batch_tokens)
            logits = out.view(-1, cfg.VOCAB_SIZE)
            labels = batch_labels.view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    dl = get_ima_txt_data_loader()
    transformer_model = TransformerModel()

    training(transformer_model, dl)
