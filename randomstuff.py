import cv2
import os
import numpy as np
from tqdm import tqdm 
import re
import csv
from PIL import Image
import matplotlib.pyplot as plt

BASE = "E:\Mishaal\sem_dauer_2\\"
def only_junc_images_datast():
    #compile a list of images and their corresponding segments that have a gap junction annotation:
    jnc_files = os.listdir(BASE+"seg_export0507_2\\")

    new_img_files, new_seg_files = [], []
    for file in tqdm(jnc_files):
        seg = cv2.cvtColor(cv2.imread(BASE+"seg_export0507_2\\"+file), cv2.COLOR_BGR2GRAY)
        if len(np.unique(seg)) >= 2:
            new_img_files.append(file.replace("sem2dauer_gj_2d_training.vsseg_export", "SEM_dauer_2_image_export"))
            new_seg_files.append(file)

    new_img_files = [file + "\n" for file in new_img_files]
    new_seg_files = [file + "\n" for file in new_seg_files]
    with open(BASE + "jnc_only_images.txt", "w") as f:
        f.writelines(new_img_files)
    with open(BASE + "jnc_only_seg.txt", "w") as f:
        f.writelines(new_seg_files)

def neuron_present_images_dataset():
    jnc_files = os.listdir(BASE+"sem_dauer_2/seg_export/")

    new_img_files = []
    for file in tqdm(jnc_files):
        new_img_files.append(file.replace("sem_dauer_2_gj_segmentation_", "SEM_dauer_2_image_export_"))
    
    new_img_files = [file + "\n" for file in new_img_files]
    with open(BASE + "sem_dauer_2/seg_export_images.txt", "w") as f:
        f.writelines(new_img_files)

def ressample(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return A


def erase_neurons():
    img_files = os.listdir(BASE+"jnc_only_images\\")
    seg_files = os.listdir(BASE+"jnc_only_seg\\")
    neuron_files = os.listdir("E:\Mishaal\GapJunction\seg_export\\")
    os.mkdir(BASE+"erased_neurons\\")
    os.mkdir(BASE+"erased_neurons\\gts\\")
    os.mkdir(BASE+"erased_neurons\\imgs\\")

    for i in tqdm(range(len(neuron_files))):
        img_file = neuron_files[i].replace("20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_", "SEM_dauer_2_image_export_")
        seg_file = neuron_files[i].replace("20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_", "sem2dauer_gj_2d_training.vsseg_export_")
        if not os.path.isfile(BASE+"jnc_only_images\\" + img_file) or not os.path.isfile(BASE+"jnc_only_seg\\" + seg_file): continue
        img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_file), cv2.COLOR_BGR2GRAY)
        neuron_mask = cv2.cvtColor(cv2.imread("E:\Mishaal\GapJunction\seg_export\\"+neuron_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"jnc_only_seg\\"+seg_file), cv2.COLOR_BGR2GRAY)
        seg[seg != 1] = 0
        seg[seg == 1] = 255
        
        img[neuron_mask != 0] = 255

        #write the new image to directory along with a seg directory 
        cv2.imwrite(BASE+"erased_neurons\\imgs\\"+img_file, img)
        cv2.imwrite(BASE+"erased_neurons\\gts\\"+seg_file, seg)

def shorten_pics():
    counter = 0
    img_files = sorted(os.listdir(BASE+"jnc_only_images\\"))
    seg_files = sorted(os.listdir(BASE+"jnc_only_seg\\"))
    # os.mkdir(BASE+"tiny_jnc_only\\")
    os.mkdir(BASE+"tiny_jnc_only\\gts\\")
    os.mkdir(BASE+"tiny_jnc_only\\imgs\\")
    os.mkdir(BASE+"tiny_jnc_only\\masks\\")
    os.mkdir(BASE+"tiny_jnc_only\\mito_masks\\")

    for i in tqdm(range(len(img_files))):
        img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"jnc_only_seg\\"+seg_files[i]), cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(cv2.imread(("E:\\Mishaal\\GapJunction\\seg_export\\"+seg_files[i]).replace("sem2dauer_gj_2d_training.vsseg_export", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export")), cv2.COLOR_BGR2GRAY)
        mito_mask = np.array(Image.open("E:\\Mishaal\\sem_dauer_2\\jnc_only_dataset\\mito_masks\\shoobedoo\\masks\\"+img_files[i]+".tiff"))
        seg[seg != 1] = 0
        seg[seg == 1] = 255

        #split the image into 32x32 bits
        imgs, segs, masks, mito_masks = ressample(img, 128), ressample(seg, 128), ressample(mask, 128), ressample(mito_mask, 128)
        # save these 
        counter=0
        for j in range(len(segs)):
            cv2.imwrite(BASE+"tiny_jnc_only\\gts\\{}.png".format(seg_files[i][:-4] + "_"+str(counter)), segs[j])
            cv2.imwrite(BASE+"tiny_jnc_only\\imgs\\{}.png".format(img_files[i][:-4] + "_"+str(counter)), imgs[j])
            cv2.imwrite(BASE+"tiny_jnc_only\\masks\\{}.png".format(seg_files[i][:-4].replace("sem2dauer_gj_2d_training.vsseg_export", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export")
                                                                    + "_"+str(counter)), masks[j])
            cv2.imwrite(BASE+"tiny_jnc_only\\mito_masks\\{}.png".format(img_files[i][:-4] + "_"+str(counter)), mito_masks[j])
            counter+=1

def stitch_short(tiny_dir,key):
    files = os.listdir(tiny_dir)
    os.mkdir(tiny_dir+"stitched\\")

    assert len(files) % 3 == 0

    img_acc = []
    base_count = None
    for i in range(0, len(files)//3):
        if key not in files[i]: continue
        if i % 16 == 0 and i != 0:
            #np.array(img_acc).reshape((512, 512))
            np.concatenate([np.concatenate([img_acc[:4]], axis=1), np.concatenate([img_acc[4:8]], axis=1), np.concatenate([img_acc[8:12]], axis=1), np.concatenate([img_acc[12:16]], axis=1)], axis=0) 
            cv2.imwrite(tiny_dir+"stitched\\{}.png".format(base_count[:-5]+".png"), img_acc)
            img_acc = []

        img = cv2.cvtColor(cv2.imread(tiny_dir+files[i]), cv2.COLOR_BGR2GRAY)
        img_acc.append(img)
        base_count = files[i]



def mask_acc(gt, pred):
    return np.sum(gt == pred) / (gt.shape[0] * gt.shape[1])

def mask_acc_generous(gt, pred):
    new_gt = np.pad(gt, ((50, 50), (50, 50)), 'constant', constant_values=(0, 0))
    new_gt = np.roll(new_gt, (50, -50, 50, -50), axis=(0, 0, 1, 1))
    new_gt = new_gt[50:-50, 50:-50]
    return np.sum(((new_gt == pred) +(gt == pred)) >= 1) / (gt.shape[0] * gt.shape[1])

def iou_accuracy(gt, pred):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(np.logical_or(gt == 255, pred == 255))

def mask_recall(gt, pred):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(gt == 255)


def mask_acc_split(results_dir, classes=False):
    val_res = sorted(os.listdir(results_dir + "valid_res\\"))
    train_res = sorted(os.listdir(results_dir + "train_res\\"))

    val_classes, val_preds, val_gt, val_gcls = [], [], [], []
    train_classes, train_preds, train_gt, train_gcls = [], [], [], []

    if classes:
        v_cls_file, t_cls_file = open(results_dir + "val_classes.txt", "r"), open(results_dir + "train_classes.txt", "r")
        v_cls, t_cls = csv.reader(v_cls_file), csv.reader(t_cls_file)

    val_acc, val_recall, t_recall, t_acc = [], [], [], []
    val_acc_gen, val_acc_iou, t_acc_gen, t_acc_iou = [], [], [], []

    for i, file in enumerate(val_res):
        #find the _no
        if "pred" in file: continue
        try:
            gt = plt.imread(results_dir + "valid_res\\" + file)
            no = re.findall(r'_\d+', file)[0]
            pred = plt.imread(results_dir + "valid_res\\" + file.replace("sample_gt__", "sample_pred_binary__"))

            gt = gt >= 0.5
        except: continue
        if classes: 
            cls = 1 if "True" in v_cls[no] else 0
            true_cls = np.any(gt == 255).item()

        val_acc.append(mask_acc(gt, pred))
        val_recall.append(mask_recall(gt, pred))
        val_acc_gen.append(mask_acc_generous(gt, pred))
        val_acc_iou.append(iou_accuracy(gt, pred))
        
        val_preds.append(pred)
        val_gt.append(gt)
        if classes:
            val_classes.append(cls)
            val_gcls.append(true_cls)
    
    for i, file in enumerate(train_res):
        if "pred" in file: continue
        try:
            gt = cv2.cvtColor(cv2.imread(results_dir + "train_res\\" + file), cv2.COLOR_BGR2GRAY)
            no = re.findall(r'_\d+', file)[0]
            pred = cv2.cvtColor(cv2.imread(results_dir + "train_res\\" + file.replace("sample_gt__", "sample_pred_binary__")), cv2.COLOR_BGR2GRAY)
        except: continue
        if classes:
            cls = 1 if "True" in t_cls[no] else 0
            true_cls = np.any(gt == 255).item()

        t_acc.append(mask_acc(gt, pred))
        t_recall.append(mask_recall(gt, pred))
        t_acc_gen.append(mask_acc_generous(gt, pred))
        t_acc_iou.append(iou_accuracy(gt, pred))

        
        train_preds.append(pred)
        train_gt.append(gt)
        if classes:
            train_classes.append(cls)
            train_gcls.append(true_cls)
    

    print("Validation set classifier accuracy {}".format(np.mean((np.array(val_classes) == np.array(val_gcls)))))
    print("Training set classifier accuracy {}".format(np.mean((np.array(train_classes) == np.array(train_gcls)))))
    print("Validation set mask accuracy {}".format(np.mean(val_acc)))
    print("Training set mask accuracy {}".format(np.mean(t_acc)))
    print("Validation set mask recall {}".format(np.mean(val_recall)))
    print("Training set mask recall {}".format(np.mean(t_recall)))
    print("Validation set mask accuracy generous {}".format(np.mean(val_acc_gen)))
    print("Training set mask accuracy generous {}".format(np.mean(t_acc_gen)))
    print("Validation set mask accuracy iou {}".format(np.mean(val_acc_iou)))
    print("Training set mask accuracy iou {}".format(np.mean(t_acc_iou)))


if __name__ == "__main__":

    # mask_acc_split("E:\\Mishaal\\GapJunction\\results\\tiny\\", classes=False)


    # erase_neurons()
    shorten_pics()
    # import shutil

    # os.mkdir(BASE+"tiny_jnc_only\\gts1\\")
    # os.mkdir(BASE+"tiny_jnc_only\\imgs1\\")
    # os.mkdir(BASE+"tiny_jnc_only\\masks1\\")
    # os.mkdir(BASE+"tiny_jnc_only\\mito_masks1\\")

    # files = sorted(os.listdir(BASE+"tiny_jnc_only\\gts\\"))
    # images = sorted(os.listdir(BASE+"tiny_jnc_only\\imgs\\"))
    # masks = sorted(os.listdir(BASE+"tiny_jnc_only\\masks\\"))
    # mito_masks = sorted(os.listdir(BASE+"tiny_jnc_only\\mito_masks\\"))

    # #class balancing (65-35)
    # empty_segs, present_segs = [], []
    # for i in tqdm(range(len(files))):
    #     im = cv2.cvtColor(cv2.imread(BASE+"tiny_jnc_only\\gts\\"+files[i]), cv2.COLOR_BGR2GRAY)
    #     if len(np.unique(im)) == 2: present_segs.append(i)
    #     else: empty_segs.append(i)
    # n_present = len(present_segs)
    # total = (n_present *100)//65
    # empty_segs = np.random.choice(np.array(empty_segs), size=7800, replace=False).tolist()
    # all_segs = empty_segs + present_segs

    # for i in tqdm(range(len(files))):
    #     if i not in all_segs: continue
    #     shutil.copyfile(BASE+"tiny_jnc_only\\gts\\"+files[i],BASE+"tiny_jnc_only\\gts1\\"+files[i])
    #     shutil.copyfile(BASE+"tiny_jnc_only\\masks\\"+masks[i],BASE+"tiny_jnc_only\\masks1\\"+masks[i])
    #     shutil.copyfile(BASE+"tiny_jnc_only\\imgs\\"+files[i].replace("sem2dauer_gj_2d_training.vsseg_export", "SEM_dauer_2_image_export"),BASE+"tiny_jnc_only\\imgs1\\"+images[i])
    #     shutil.copyfile(BASE+"tiny_jnc_only\\mito_masks\\"+mito_masks[i],BASE+"tiny_jnc_only\\mito_masks1\\"+mito_masks[i])

    # print("Class balanced tiny has {} empty and {} present with a total of {} points".format(empty_segs, present_segs, len(all_segs)))


    # only_junc_images_datast()
    # neuron_present_images_dataset()

    # from shutil import copyfile 
    # for line in open(BASE+"jnc_only_images.txt", 'r'): 
    #     filename=BASE+"image_export\\" + line.strip() 
    #     dest=BASE+"jnc_only_images\\"+ line.strip() 
    #     copyfile(filename, dest)

    #make a jnc_only_dataset
    # os.mkdir(BASE+"jnc_only_dataset\\")
    # os.mkdir(BASE+"jnc_only_dataset\\gts\\")
    # os.mkdir(BASE+"jnc_only_dataset\\imgs\\")
    # for sec in ["train", "test", "valid"]:
    #     BASE = "E:\\Mishaal\\GapJunction\\small_data\\ground_truth\\"
    #     # img_files = sorted(os.listdir(BASE+"jnc_only_images\\"))
    #     seg_files = sorted(os.listdir(BASE+sec+"\\"))
    #     os.mkdir(BASE+sec+"_new\\")

    #     for i in tqdm(range(len(seg_files))):
    #         # img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_files[i]), cv2.COLOR_BGR2GRAY)
    #         seg = cv2.cvtColor(cv2.imread(BASE+sec+"\\"+seg_files[i]), cv2.COLOR_BGR2GRAY)
    #         if len(np.unique(seg)) > 2: raise Exception(np.unique(seg))
    #         seg[seg == 30] = 0
    #         seg[seg == 215] = 255

    #         cv2.imwrite(BASE+sec+"_new\\{}".format(seg_files[i]), seg)
    #         # cv2.imwrite(BASE+"jnc_only_dataset\\imgs\\{}.png".format(img_files[i]), img)


