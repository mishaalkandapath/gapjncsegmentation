import numpy as np
import cv2
import multiprocessing as mp
import re
import os
from tqdm import tqdm
import glob
from collections import defaultdict

NUM_SECTIONS=50
IMG_FILE="/home/mishaalk/scratch/gapjunc/seg_export/"

def shift_img(im_classes, im_path, X, Y, mode="left"):
    max_cover_path, max_cover, new_classes = im_path, None, None
    
    for i in range(3):
        match mode:
            case "left": new_img_path = im_path.replace("X{}".format(X), "X{}".format(X-i-1))
            case "right": new_img_path = im_path.replace("X{}".format(X), "X{}".format(X+i+1))
            case "top": new_img_path = im_path.replace("X{}".format(Y), "X{}".format(Y-i-1))
            case "bottom": new_img_path = im_path.replace("X{}".format(X), "X{}".format(Y+i+1))
            case _: raise Exception
        
        if not os.path.isfile(new_img_path): return max_cover_path, max_cover, new_classes

        new_img = cv2.imread(new_img_path)[0]
        new_classes = np.unique(new_img)
        if new_classes[0]: new_classes = new_classes[0:]
        new_classes = set(new_classes)
        
        #check the intersection on this new image:
        if max_cover is None or len(im_classes.intersection(new_classes))/len(im_classes) > max_cover:
            max_cover_path, max_cover = new_img_path, len((im_classes.intersection(new_classes)))/len(im_classes)
    return max_cover_path, max_cover, new_classes


def gen_from_image(image_path):
    global NUM_SECTIONS
    
    struc_chain = [image_path]
    
    im = cv2.imread(image_path)
    im = im[0]
    initial_classes = np.unique(im)
    if initial_classes[0] == 0: initial_classes = initial_classes[1:]
    if len(initial_classes) == 0: raise Exception("empty tile for file {}".format(image_path))
    initial_classes = set(initial_classes)
    X, Y = int(re.findall(r'X\d*.', image_path)[0][1:-1]), int(re.findall(r'Y\d*_', image_path)[0][1:-1])
    for i in tqdm(range(NUM_SECTIONS)):
        new_img_path = image_path.replace("s00{}".format(i+1), "s00{}".format(i+2) if i < 8 else "s0{}".format(i+2))
        new_img = cv2.imread(new_img_path)[0]
        new_classes = np.unique(new_img)
        if new_classes[0] == 0: new_classes = new_classes[1:]
        new_classes = set(new_classes)
        if len(initial_classes.intersection(new_classes)) == len(initial_classes): 
            initial_classes = new_classes
            image_path = new_img_path
            struc_chain.append(new_img_path)
            continue
        cur_cover = len(initial_classes.intersection(new_classes))/len(initial_classes)
        
        for i, mode in enumerate(("left", "right", "top", "bottom")):
            best_path, best_cover, best_classes = shift_img(initial_classes, new_img_path, X, Y, mode = mode)
            if best_cover is None: continue
            if best_cover > cur_cover:
                new_img_path, cur_cover, new_classes = best_path, best_cover, best_classes
                
        initial_classes = new_classes
        image_path = new_img_path
        struc_chain.append(new_img_path)
    return struc_chain
        
        
if __name__ == "__main__":
    # how many different structures:
    file = IMG_FILE+"20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_s001_Y7_X6.png"
    files = glob.glob(IMG_FILE+"20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_s001*")

    for file in files:
        
    # deviations = defaultdict(lambda: 0)
    # for j, file in enumerate(files):
    #     try:
    #         chain = gen_from_image(file)
    #     except Exception as e:
    #         print("exception ", e)
    #         deviations[-1]+=1
    #         continue
    #     deviation_index = 0
    #     X, Y = int(re.findall(r'X\d*.', file)[0][1:-1]), int(re.findall(r'Y\d*_', file)[0][1:-1])
    #     assert len(chain[1:]) == NUM_SECTIONS, len(chain)
    #     chain = chain[1:]
    #     for i in range(NUM_SECTIONS):
    #         file = chain[i]
    #         new_X, new_Y = int(re.findall(r'X\d*.', file)[0][1:-1]), int(re.findall(r'Y\d*_', file)[0][1:-1])
    #         if new_X != X or new_Y != Y: deviation_index+=1
    #         X, Y = new_X, new_Y
            
    #     print("for file index ", j, " deviation is ", deviation_index)
    #     deviations[deviation_index] +=1
    # print(deviations)