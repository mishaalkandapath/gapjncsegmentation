import os, numpy as np, cv2, re
from tqdm import tqdm

base = "E:\\Mishaal\\sem_dauer_2\\sem_dauer_0_50_overlap\\"

train_imgs, valid_imgs = [], []
train_segs, valid_segs = [], []
train_mito_mask, valid_mito_mask = [], []
train_neuron_mask, valid_neuron_mask = [], []

S = set(range(33, 43)) | set(range(3, 13)) | set(range(18, 28))
X = set(range(3, 7)) | set(range(6, 9)) | set(range(6, 9))
Y = set(range(7, 1)) | set(range(2, 5)) | set(range(7, 9))

for file in tqdm(os.listdir(base+"imgs\\")):
    try:
        x, y, s = int(re.findall(r"X\d+", file)[0][1:]), int(re.findall(r"Y\d+_", file)[0][1:-1]), int(re.findall(r"s\d+_", file)[0][1:-1])
    except:
        print(file)
        raise Exception
    if s in S and x in X and y in Y:
        valid_imgs.append(file)
        valid_segs.append(file)
        # valid_mito_mask.append(file[:-4] + ".tiff")
        valid_neuron_mask.append(file)

        assert os.path.isdir(base+"gts\\"+file)
        assert os.path.isdir(base+"masks\\"+file)
        # assert os.path.isfile("E:\\Mishaal\\sem_dauer_2\\jnc_only_dataset\\mito_masks\\shoobedoo\\masks\\"+file[:-4]+".tiff")
    else:
        train_imgs.append(file)
        train_segs.append(file)
        # train_mito_mask.append(file[:-4] + ".tiff")
        train_neuron_mask.append(file)

        assert os.path.isdir(base+"gts\\"+file)
        assert os.path.isdir(base+"masks\\"+file), "E:\\Mishaal\\sem_dauer_2\\seg_export_full_volume\\"+file
        # assert os.path.isfile("E:\\Mishaal\\sem_dauer_2\\jnc_only_dataset\\mito_masks\\shoobedoo\\masks\\"+file[:-4]+".tiff")

print(len(valid_imgs), len(train_imgs))

import shutil
os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\valid_gts")
os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\train_gts")
os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\valid_imgs")
os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\train_imgs")
# os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split\\valid_mito")
# os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split\\train_mito")
os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\valid_neuro")
os.mkdir("E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\train_neuro")

for i in range(len(train_imgs)):
    shutil.copy(base+"imgs\\"+train_imgs[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\train_imgs\\"+train_imgs[i])
    shutil.copy(base+"gts\\"+train_segs[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\train_gts\\"+train_segs[i])
    # shutil.copy("E:\\Mishaal\\sem_dauer_2\\jnc_only_dataset\\mito_masks\\shoobedoo\\masks\\"+train_mito_mask[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split\\train_mito\\"+train_mito_mask[i])
    shutil.copy(base+"masks\\"+train_neuron_mask[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\train_neuro\\"+train_neuron_mask[i])

for i in range(len(valid_imgs)):
    shutil.copy(base+"imgs\\"+valid_imgs[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\valid_imgs\\"+valid_imgs[i])
    shutil.copy(base+"gts\\"+valid_segs[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\valid_gts\\"+valid_segs[i])
    # shutil.copy("E:\\Mishaal\\sem_dauer_2\\jnc_only_dataset\\mito_masks\\shoobedoo\\masks\\"+valid_mito_mask[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split\\valid_mito\\"+valid_mito_mask[i])
    shutil.copy(base+"masks\\"+valid_neuron_mask[i], "E:\\Mishaal\\sem_dauer_2\\final_jnc_only_split3d\\valid_neuro\\"+valid_neuron_mask[i])
