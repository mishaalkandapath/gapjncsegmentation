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

def shorten_pics():
    counter = 0
    img_files = sorted(os.listdir(BASE+"jnc_only_images\\"))
    seg_files = sorted(os.listdir(BASE+"jnc_only_seg\\"))
    os.mkdir(BASE+"tiny_jnc_only\\")
    os.mkdir(BASE+"tiny_jnc_only\\gts\\")
    os.mkdir(BASE+"tiny_jnc_only\\imgs\\")

    for i in tqdm(range(len(img_files))):
        img = cv2.cvtColor(cv2.imread(BASE+"jnc_only_images\\"+img_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"jnc_only_seg\\"+seg_files[i]), cv2.COLOR_BGR2GRAY)

        #split the image into 32x32 bits
        imgs, segs = ressample(img, 32), ressample(seg, 32)
        # save these 
        count=0
        for i in range(len(segs)):
            cv2.imwrite(BASE+"tiny_jnc_only\\gts\\{}.png".format(counter), segs[i])
            cv2.imwrite(BASE+"tiny_jnc_only\\imgs\\{}.png".format(counter), imgs[i])
            counter+=1
    
if __name__ == "__main__":
    shorten_pics()
    # only_junc_images_datast()
    # neuron_present_images_dataset()

    # from shutil import copyfile 
    # for line in open(BASE+"jnc_only_images.txt", 'r'): 
    #     filename=BASE+"image_export\\" + line.strip() 
    #     dest=BASE+"jnc_only_images\\"+ line.strip() 
    #     copyfile(filename, dest)


