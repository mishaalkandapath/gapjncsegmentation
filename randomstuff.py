import cv2
import os
import numpy as np
from tqdm import tqdm 

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
    os.mkdir(BASE+"tiny_jnc_only1\\")
    os.mkdir(BASE+"tiny_jnc_only1\\gts\\")
    os.mkdir(BASE+"tiny_jnc_only1\\imgs\\")

    for i in tqdm(range(len(img_files))):
        img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"jnc_only_seg\\"+seg_files[i]), cv2.COLOR_BGR2GRAY)
        seg[seg != 1] = 0
        seg[seg == 1] = 255

        #split the image into 32x32 bits
        imgs, segs = ressample(img, 128), ressample(seg, 128)
        # save these 
        count=0
        for i in range(len(segs)):
            cv2.imwrite(BASE+"tiny_jnc_only1\\gts\\{}.png".format(seg_files[i][:-4] + "_"+str(counter)), segs[i])
            cv2.imwrite(BASE+"tiny_jnc_only1\\imgs\\{}.png".format(img_files[i][:-4] + "_"+str(counter)), imgs[i])
            counter+=1
    
if __name__ == "__main__":
    # erase_neurons()
    shorten_pics()
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


