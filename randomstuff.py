import cv2
import os
import numpy as np
from tqdm import tqdm 

BASE = "/home/mishaalk/scratch/gapjunc/"
def only_junc_images_datast():
    #compile a list of images and their corresponding segments that have a gap junction annotation:
    jnc_files = os.listdir(BASE+"sem_dauer_2/seg_export_0507/")

    new_img_files, new_seg_files = [], []
    for file in tqdm(jnc_files):
        seg = cv2.cvtColor(cv2.imread(BASE+"sem_dauer_2/seg_export_0507/"+file), cv2.COLOR_BGR2GRAY)
        if len(np.unique(seg)) >= 2:
            new_img_files.append(file.replace("sem_dauer_2_gj_segmentation_", "SEM_dauer_2_image_export_"))
            new_seg_files.append(file)

    new_img_files = [file + "\n" for file in new_img_files]
    with open(BASE + "sem_dauer_2/jnc_only_images.txt", "w") as f:
        f.writelines(new_img_files)
    with open(BASE + "sem_dauer_2/jnc_only_seg.txt", "w") as f:
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
    img_files = os.listdir(BASE+"sem_dauer_2/jnc_only_images/", sorted=True)
    seg_files = os.listdir(BASE+"sem_dauer_2/jnc_only_seg/", sorted=True)
    neuron_files = os.listdir(BASE+"seg_export/", sorted=True)
    os.mkdir(BASE+"sem_dauer_2/erased_neurons/")
    os.mkdir(BASE+"sem_dauer_2/erased_neurons/gts/")
    os.mkdir(BASE+"sem_dauer_2/erased_neurons/imgs/")

    for i in tqdm(range(len(neuron_files))):
        img_file = neuron_files[i].replace()
        seg_file = neuron_files[i].replace()
        if not os.path.isfile(BASE+"sem_dauer_2/jnc_only_images/" + img_file) or not os.path.isfile(BASE+"sem_dauer_2/jnc_only_seg/" + seg_file): continue
        img = cv2.cvtColor(cv2.imread(BASE+"sem_dauer_2/jnc_only_images/"+img_file), cv2.COLOR_BGR2GRAY)
        neuron_mask = cv2.cvtColor(cv2.imread(BASE+"seg_export/"+neuron_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"sem_dauer_2/jnc_only_seg/"+seg_file), cv2.COLOR_BGR2GRAY)
        img[neuron_mask != 0] = 255

        #write the new image to directory along with a seg directory 
        cv2.imwrite(BASE+"sem_dauer_2/erased_neurons/imgs/"+img_file, img)
        cv2.imwrite(BASE+"sem_dauer_2/erased_neurons/gts/"+seg_file, seg)


def shorten_pics():
    counter = 0
    img_files = os.listdir(BASE+"sem_dauer_2/jnc_only_images/", sorted=True)
    seg_files = os.listdir(BASE+"sem_dauer_2/jnc_only_seg/", sorted=True)
    os.mkdir(BASE+"sem_dauer_2/tiny_jnc_only/")
    os.mkdir(BASE+"sem_dauer_2/tiny_jnc_only/gts/")
    os.mkdir(BASE+"sem_dauer_2/tiny_jnc_only/imgs/")

    for i in tqdm(range(len(img_files))):
        img = cv2.cvtColor(cv2.imread(BASE+"sem_dauer_2/jnc_only_images/"+img_files[i]), cv2.COLOR_BGR2GRAY)
        seg = cv2.cvtColor(cv2.imread(BASE+"sem_dauer_2/jnc_only_seg/"+seg_files[i]), cv2.COLOR_BGR2GRAY)

        #split the image into 32x32 bits
        imgs, segs = ressample(img, 32), ressample(seg, 32)
        # save these 
        for i in range(len(segs)):
            cv2.imwrite(BASE+"sem_dauer_2/tiny_jnc_only/gts/{}.png".format(seg_files[i][:-4] + str(counter)), segs[i])
            cv2.imwrite(BASE+"sem_dauer_2/tiny_jnc_only/imgs/{}.png".format(img_files[i][:-4] + str(counter)), imgs[i])
            counter+=1
    
if __name__ == "__main__":
    only_junc_images_datast()
    # neuron_present_images_dataset()


