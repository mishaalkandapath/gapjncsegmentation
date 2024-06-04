import os
import re
import cv2, numpy as np, shutil
from tqdm import tqdm
BASE = "/Volumes/Normal/gapjnc/"
base = "/Volumes/Normal/gapjnc/final_jnc_only_split/"

def get_seg(imgname):
        newname = imgname.replace("SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_")
        return base+"train_gts/"+newname

def get_neuron(imgname):
     newname = imgname.replace("SEM_dauer_2_image_export_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_")
     if os.path.isfile(base+"train_neuro/"+newname):
         return base+"train_neuro/"+newname
     else: 
        newname = newname.replace(".png.png", ".png")
        assert os.path.isfile("/Volumes/Normal/gapjnc/seg_export_full_volume/"+newname), newname
        return "/Volumes/Normal/gapjnc/seg_export_full_volume/"+newname
     
pattern = r's\d\d\d'
def get_another(filename, i):
    i = int(re.findall(pattern, filename)[0][1:]) + i
    assert i>=0, i
    return filename.replace(re.findall(pattern, filename)[0], "s"+("0"*(3 - len(str(i)))) + str(i))

imgs = os.listdir(base+"train_imgs/")

flat_imgs, flat_segs, flat_masks = [], [], []
seq_imgs, seq_segs, seq_masks = [], [], []

for img in tqdm(imgs):
    seg = cv2.cvtColor(cv2.imread(get_seg(img)), cv2.COLOR_BGR2GRAY)
    neuron = cv2.cvtColor(cv2.imread(get_neuron(img)), cv2.COLOR_BGR2GRAY)
    depth = int(re.findall(pattern, img)[0][1:])
    if depth == 0 or depth > 48: continue
    if len(np.unique(seg)) >= 2:
            
        # flat_masks += [get_mask(img)]
        flat_segs += [get_seg(img)]
        
        # seq_masks.append([get_another(get_mask(img), -1), get_mask(img), get_another(get_mask(img), 1), get_another(get_mask(img), 2)])
        seq_segs.append([get_another(get_seg(img), -1), get_seg(img), get_another(get_seg(img), 1), get_another(get_seg(img), 2)])

        flat_masks += [get_neuron(img)]
        seq_masks.append([get_another(get_neuron(img), -1), get_neuron(img), get_another(get_neuron(img), 1), get_another(get_neuron(img), 2)])
        
        img = (base+"train_imgs/"+img)

        flat_imgs += [img]
        seq_imgs.append([get_another(img, -1), img, get_another(img, 1), get_another(img, 2)])

for i in tqdm(range(len(seq_imgs))):
    # os.mkdir(BASE+"3d_test\\masks\\"+os.path.split(seq_imgs[i][1])[-1][:-4])
    os.mkdir(BASE+"3d_train/train_gts/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", ""))
    os.mkdir(BASE+"3d_train/train_imgs/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", ""))
    os.mkdir(BASE+"3d_train/train_neuro/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", ""))

    for j in range(4):
        try:
            shutil.copy(seq_imgs[i][j], BASE+"3d_train/train_imgs/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", "")+"/"+os.path.split(seq_imgs[i][j])[-1].replace(".png.png", ".png"))
        except:
            shutil.copy(seq_imgs[i][j].replace("final_jnc_only_split/train_imgs", "image_export").replace(".png.png", ".png"), BASE+"3d_train/train_imgs/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", "")+"/"+os.path.split(seq_imgs[i][j])[-1].replace(".png.png", ".png"))
        # shutil.copy(flat_segs[i], BASE+"3d_jnc_only\\imgs\\"+os.path.split(flat_segs[i]))
        try:
            seg = cv2.cvtColor(cv2.imread(seq_segs[i][j]), cv2.COLOR_BGR2GRAY)
        except:
            print(seq_segs[i][j])
            seg = np.zeros((512, 512))
        seg *= 255
        cv2.imwrite(BASE+"3d_train/train_gts/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", "")+"/"+os.path.split(seq_segs[i][j])[-1].replace(".png.png", ".png"), seg)

        shutil.copy(seq_masks[i][j], BASE+"3d_train/train_neuro/"+os.path.split(seq_imgs[i][1])[-1].replace(".png", "")+"/"+os.path.split(seq_masks[i][j])[-1])


def sanity_check():
    dirs = os.listdir(BASE+"3d_train/train_gts/")
    #check if ther is any directoryu where all files inside ar eempty
    for d in dirs:
        files = os.listdir(BASE+"3d_train/train_gts/"+d)
        for f in files:
            im = cv2.cvtColor(cv2.imread(BASE+"3d_train/train_gts/"+d+"/"+f))
            print(np.unique(im))
            if len(np.unique(im)) == 1:
                print(f)
        raise Exception
        