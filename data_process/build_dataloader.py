from torch.utils.data import Dataset, DataLoader
import os
import cv2
from transformers import BertTokenizer
import config as cfg
import torch


class ImgTxtDataSet(Dataset):

    def __init__(self, data_folder, pre_train_path):
        self.img_list, self.txt_list = self.get_files_path_list(data_folder)
        self.tokenizer = BertTokenizer.from_pretrained(pre_train_path)

    def get_files_path_list(self, data_folder):
        img_list = []
        txt_list = []
        for name in os.listdir(data_folder + '/imgs'):
            img_path = data_folder + '/imgs/' + name

            txt_path = data_folder + '/txts/' + os.path.splitext(name)[0] + '.txt'
            assert os.path.exists(txt_path)
            img_list.append(img_path)
            txt_list.append(txt_path)

        return img_list, txt_list

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

    def get_text_tokens(self, path):
        f = open(path, 'r')
        text = f.read()
        f.close()

        text_list = [t for t in text]
        input_dict = self.tokenizer.encode_plus(
            text_list,
            padding="max_length",
            max_length=cfg.MAX_LEN,
            is_split_into_words=True,
            truncation=True
        )
        input_ids = input_dict['input_ids']
        token_type_ids = input_dict['token_type_ids']
        attention_mask = input_dict['attention_mask']
        label_ids = input_ids[1:]
        input_ids = input_ids[:-1]
        return input_ids, label_ids

    def __getitem__(self, item):
        img_path = self.img_list[item]
        txt_path = self.txt_list[item]
        patch = self.get_patchs(img_path)
        tokens, labels = self.get_text_tokens(txt_path)
        return patch, tokens, labels

    def __len__(self):
        return len(self.img_list)


def batch_dataset_fn(batch):
    batch_patch = torch.tensor([b[0] for b in batch], dtype=torch.float32)
    batch_tokens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    batch_labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return batch_patch, batch_tokens, batch_labels


def get_ima_txt_data_loader():
    itds = ImgTxtDataSet('/data/yangdongquan/research/data', cfg.PRE_TRAIN_PATH)
    data_iter = DataLoader(itds, shuffle=True, batch_size=cfg.BATCH_SIZE, collate_fn=batch_dataset_fn, num_workers=20)
    return data_iter


if __name__ == '__main__':
    dl = get_ima_txt_data_loader()
    for batch_patch, batch_tokens, batch_labels in dl:
        print(batch_patch.shape, batch_tokens.shape, batch_labels.shape)
