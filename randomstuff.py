import cv2
import os
import numpy as np
from tqdm import tqdm 
import re
import csv
from PIL import Image
import matplotlib.pyplot as plt
import threading, random
import torch

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

def stitch_short(train_dir, preds_dir, s_dir):
    pred_files = os.listdir(preds_dir)
    pred_files.remove("classes.csv")
    img_files = sorted(os.listdir(train_dir+"imgs/"), key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))
    seg_files = sorted(os.listdir(train_dir + "gts/"), key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))
    pred_files = sorted(pred_files, key=lambda x: (x.replace(re.findall(r"_\d+\.", x)[0], ""), int(re.findall(r"_\d+\.", x)[0][1:-1])))

    all_saves_list = ThreadSafeList()

    def helper_inside(img_files, seg_files, pred_files, save_dir):
        img_acc, seg_acc, pred_acc = [], [], []
        base_count = None
        for i in tqdm(range(0, len(img_files))):
            if i % 16 == 0 and i != 0:
                #np.array(img_acc).reshape((512, 512))
                img_acc = np.concatenate([np.concatenate(img_acc[:4], axis=1), np.concatenate(img_acc[4:8], axis=1), np.concatenate(img_acc[8:12], axis=1), np.concatenate(img_acc[12:16], axis=1)], axis=0) 
                # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_img."))
                #all_saves_list.append((img_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_img."))))
                cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_img.")), img_acc)
                img_acc = []

                # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_sef."))
                seg_acc = np.concatenate([np.concatenate(seg_acc[:4], axis=1), np.concatenate(seg_acc[4:8], axis=1), np.concatenate(seg_acc[8:12], axis=1), np.concatenate(seg_acc[12:16], axis=1)], axis=0) 
                cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_seg.")), seg_acc)
                #all_saves_list.append((seg_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_seg."))))
                seg_acc = []

                # assert not os.path.isfile(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred."))
                pred_acc = np.concatenate([np.concatenate(pred_acc[:4], axis=1), np.concatenate(pred_acc[4:8], axis=1), np.concatenate(pred_acc[8:12], axis=1), np.concatenate(pred_acc[12:16], axis=1)], axis=0) 
                #all_saves_list.append((pred_acc, save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred."))))
                cv2.imwrite(save_dir+"{}".format(base_count.replace(re.findall(r"_\d+\.", base_count)[0], "_pred.")), pred_acc)
                pred_acc = []

            img = cv2.cvtColor(cv2.imread(train_dir+"imgs/"+img_files[i]), cv2.COLOR_BGR2GRAY)
            img_acc.append(img)
            base_count = img_files[i]

            seg = cv2.cvtColor(cv2.imread(train_dir+"gts/"+seg_files[i]), cv2.COLOR_BGR2GRAY)
            seg_acc.append(seg)

            pred = cv2.cvtColor(cv2.imread(preds_dir+pred_files[i]), cv2.COLOR_BGR2GRAY)
            pred_acc.append(pred)
        print("I did {} files".format(len(img_files)))

    offsets = len(img_files)//16
    threads = []
    for j in range(11):
        t = threading.Thread(target=helper_inside, args=(img_files[j*245*16 : (j+1)*245*16], seg_files[j*245*16 : (j+1)*245*16], pred_files[j*245*16 : (j+1)*245*16], s_dir))
        threads.append(t)

    [t.start() for t in threads]
    [t.join() for t in threads]
    print(all_saves_list.length())



def mask_acc(gt, pred):
    return np.sum(gt == pred) / (gt.shape[0] * gt.shape[1])

def mask_precision(gt, pred):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(pred == 255)

# def mask_precision_generous(gt, pred):
#     new_gt = np.pad(gt, ((50, 50), (50, 50)), 'constant', constant_values=(0, 0))
#     new_gt = np.roll(new_gt, (50, -50, 50, -50), axis=(0, 0, 1, 1))
#     new_gt = new_gt[50:-50, 50:-50]
#     return np.sum((np.logical_and(gt == 255, pred == 255) + np.logical_and(new_gt == 255, pred == 255)) >= 1)/ (gt.shape[0] * gt.shape[1])


def mask_precision_generous(gt, pred):
    # (depth, height, width)
    # Padding the ground truth tensor
    new_gt = torch.nn.functional.pad(gt, (50, 50, 50, 50), mode='constant', value=0)
    # Rolling the tensor 
    new_gt = torch.roll(new_gt, shifts=(50, -50, 50, -50), dims=(1, 1, 2, 2))
    # Removing the padding
    new_gt = new_gt[:, 50:-50, 50:-50]
    
    # Calculating the mask precision
    gt_255 = gt == 255
    pred_255 = pred == 255
    new_gt_255 = new_gt == 255

    intersection = (torch.logical_and(gt_255, pred_255) + torch.logical_and(new_gt_255, pred_255)) >= 1
    precision = torch.sum(intersection).item() / (gt.shape[0] * gt.shape[1])
    
    return precision


def mask_acc_generous(gt, pred):
    new_gt = np.pad(gt, ((50, 50), (50, 50)), 'constant', constant_values=(0, 0))
    new_gt = np.roll(new_gt, (50, -50, 50, -50), axis=(0, 0, 1, 1))
    new_gt = new_gt[50:-50, 50:-50]
    return np.sum(((new_gt == pred) +(gt == pred)) >= 1) / (gt.shape[0] * gt.shape[1])

def iou_accuracy(gt, pred):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(np.logical_or(gt == 255, pred == 255))

def mask_recall(gt, pred):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(gt == 255)

def mask_acc_split(data_dir, results_dir, classes=False):
    segs = sorted(os.listdir(data_dir))
    train_res = sorted(os.listdir(results_dir))
    train_classes, train_preds, train_gt, train_gcls = [], [], [], []

    if classes:
        t_cls_file = open(results_dir + "classes.txt", "r")
        t_cls = csv.reader(t_cls_file)

    t_recall, t_acc = [], []
    t_acc_gen, t_acc_iou = [], []
    t_prec, t_prec_gen = [], []

    for i, file in tqdm(enumerate(train_res), total=len(train_res)):
        if "classes" in file: continue
        # try:
        gt = cv2.cvtColor(cv2.imread(data_dir + segs[i]), cv2.COLOR_BGR2GRAY)
        no = re.findall(r'_\d+', file)[0]
        pred = cv2.cvtColor(cv2.imread(results_dir + file), cv2.COLOR_BGR2GRAY)
    # except: 
            
        #     continue
        if classes:
            cls = 1 if "True" in t_cls[no] else 0
            true_cls = np.any(gt == 255).item()

        t_acc.append(mask_acc(gt, pred))
        t_recall.append(mask_recall(gt, pred))
        t_acc_gen.append(mask_acc_generous(gt, pred))
        t_acc_iou.append(iou_accuracy(gt, pred))
        t_prec_gen.append(mask_precision_generous(gt, pred))
        t_prec.append(mask_precision(gt, pred))

        
        train_preds.append(pred)
        train_gt.append(gt)
        if classes:
            train_classes.append(cls)
            train_gcls.append(true_cls)
    

    print("Training set classifier accuracy {}".format(np.nanmean((np.array(train_classes) == np.array(train_gcls)))))
    print("Training set mask accuracy {}".format(np.nanmean(t_acc)))
    print("Training set mask recall {}".format(np.nanmean(t_recall)))
    print("Training set mask accuracy generous {}".format(np.nanmean(t_acc_gen)))
    print("Training set mask accuracy iou {}".format(np.nanmean(t_acc_iou)))
    print("Training set mask prec {}".format(np.nanmean(t_prec)))
    print("Training set mask prec generous {}".format(np.nanmean(t_prec_gen)))



if __name__ == "__main__":
    # save_dir = "/home/mishaalk/scratch/gapjunc/results/stitched_mito_preds/"
    # left = {'SEM_dauer_2_image_export_s023_Y7_X13_img.png', 'SEM_dauer_2_image_export_s033_Y11_X6_img.png', 'SEM_dauer_2_image_export_s042_Y9_X11_img.png', 'SEM_dauer_2_image_export_s009_Y7_X15_img.png', 'SEM_dauer_2_image_export_s047_Y10_X8_img.png', 'SEM_dauer_2_image_export_s050_Y9_X7_img.png', 'SEM_dauer_2_image_export_s038_Y4_X10_img.png', 'SEM_dauer_2_image_export_s014_Y10_X11_img.png', 'SEM_dauer_2_image_export_s004_Y8_X5_img.png', 'SEM_dauer_2_image_export_s028_Y5_X11_img.png', 'SEM_dauer_2_image_export_s018_Y6_X15_img.png'}

    # for file in tqdm(left):
    #     img_acc, pred_acc, seg_acc = [], [], []
    #     for i in range(16):
    #         im = cv2.cvtColor(cv2.imread("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/imgs/"+file[:-8]+"_{}.png".format(i)), cv2.COLOR_BGR2GRAY)
    #         pred = cv2.cvtColor(cv2.imread("/home/mishaalk/scratch/gapjunc/results/tiny_mito_preds/"+file[:-8]+"_{}.png".format(i)), cv2.COLOR_BGR2GRAY)
    #         seg = cv2.cvtColor(cv2.imread("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/gts/"+file[:-8].replace("SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_")+"_{}.png".format(i)), cv2.COLOR_BGR2GRAY)
    #         img_acc.append(im)
    #         pred_acc.append(pred)
    #         seg_acc.append(seg)
    #     img_acc = np.concatenate([np.concatenate(img_acc[:4], axis=1), np.concatenate(img_acc[4:8], axis=1), np.concatenate(img_acc[8:12], axis=1), np.concatenate(img_acc[12:16], axis=1)], axis=0) 
    #     seg_acc = np.concatenate([np.concatenate(seg_acc[:4], axis=1), np.concatenate(seg_acc[4:8], axis=1), np.concatenate(seg_acc[8:12], axis=1), np.concatenate(seg_acc[12:16], axis=1)], axis=0) 
    #     pred_acc = np.concatenate([np.concatenate(pred_acc[:4], axis=1), np.concatenate(pred_acc[4:8], axis=1), np.concatenate(pred_acc[8:12], axis=1), np.concatenate(pred_acc[12:16], axis=1)], axis=0) 

    #     assert not os.path.isfile(save_dir+"{}".format(file[:-8] + "_pred.png")), file
    #     assert not os.path.isfile(save_dir+"{}".format(file[:-8] + "_img.png")), file
    #     assert not os.path.isfile(save_dir+"{}".format(file[:-8] + "_seg.png")), file

        
    #     cv2.imwrite(save_dir+"{}".format(file[:-8] + "_pred.png"), pred_acc)
    #     cv2.imwrite(save_dir+"{}".format(file[:-8] + "_seg.png"), seg_acc)
    #     cv2.imwrite(save_dir+"{}".format(file[:-8] + "_img.png"), img_acc)



    mask_acc_split("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/gts/", "/home/mishaalk/scratch/gapjunc/results/tiny_mito_preds/", classes=False)
    print("-------")
    mask_acc_split("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/gts/", "/home/mishaalk/scratch/gapjunc/results/tiny_mask_preds/", classes=False)
    # stitch_short("/home/mishaalk/scratch/gapjunc/train_datasets/tiny_jnc_only_full/",\
    #               "/home/mishaalk/scratch/gapjunc/results/tiny_mito_preds/",\
    #               "/home/mishaalk/scratch/gapjunc/results/stitched_mito_preds/")


    # erase_neurons()
    # shorten_pics()
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


