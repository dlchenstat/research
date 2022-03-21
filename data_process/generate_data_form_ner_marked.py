import pickle
import shutil
import os

save_path = '/data/yangdongquan/data/20220309_456_170_641_1985.pkl'
early_ocr_rs = pickle.load(open(save_path, 'rb'))

img_save_dir = '/data/yangdongquan/research/data/imgs/'
txt_save_dir = '/data/yangdongquan/research/data/txts/'

num_of_page = 100
index = 0
for img_path in early_ocr_rs:
    v = early_ocr_rs[img_path]

    page_text = ''
    for img_name in v['image_content']:
        shutil.copyfile(img_path, img_save_dir + img_name)
        coor_ste = v['image_content'][img_name]
        for d in coor_ste:
            image_sentence = d['image_sentence']
            page_text += image_sentence

        file_prefix = os.path.splitext(img_name)[0]
        f = open(txt_save_dir + file_prefix + '.txt', 'w')
        f.write(page_text)
        f.close()

    index += 1
    if index > num_of_page:
        break
