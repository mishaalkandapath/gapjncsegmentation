""" 

Sample usage:
START_Z=0
END_Z=3
START_X=2048
END_X=7168
START_Y=2048
END_Y=6144
PRED_FP="/Users/huayinluo/Documents/stitchedpreds"
GT_FP="/Users/huayinluo/Documents/gj_seg"
python eval.py --pred_fp $PRED_FP \
               --gt_fp $GT_FP \
               --start_z $START_Z \
               --start_x $START_X \
               --start_y $START_Y \
               --end_z $END_Z \
               --end_x $END_X \
               --end_y $END_Y
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as mpatches
from utilities.utilities import visualize_3d_slice
import numpy as np
import time
from scipy.ndimage import label

def get_precision_recall(pred_vol, gt_vol, entity_thresh=0.5):
    labeled_array, num_features = label(pred_vol)
    print(f"Number of entities found in pred: {num_features}")
    # Extract each labeled component into a separate 3D array
    true_positive_components = []
    true_positive_gt_components = []
    false_positive_components = []
    false_negative_components = []
    all_components = []
    tp = 0
    fp = 0
    fn = 0
    for i in range(1, num_features + 1):
        component = (labeled_array == i).astype(np.uint8) * 255
        
        # check intersection of component with gt
        num_intersect = np.count_nonzero(component[gt_vol != 0]) # nonzero pixels in component where gt is also nonzero
        num_pred = np.count_nonzero(component)
        proportion_intersect = num_intersect/num_pred
        
        if proportion_intersect >= entity_thresh:
            true_positive_components.append(component)
            tp += 1
        else:
            false_positive_components.append(component)
            fp += 1
        all_components.append(component)

    # Label connected components
    labeled_array, num_gt_features = label(gt_vol)
    print(f"Number of entities found in gt: {num_gt_features}")
    tp_intersect_gt = 0
    for i in range(1, num_gt_features + 1):
        component = (labeled_array == i).astype(np.uint8) * 255
        num_intersect = np.count_nonzero(component[pred_vol != 0])
        num_gt = np.count_nonzero(component)
        if num_intersect/num_gt >= entity_thresh:
            tp_intersect_gt += 1
            true_positive_gt_components.append(component)
        else:
            fn += 1
            false_negative_components.append(component)
    print("---------------------------------")
    print(f"(criteria: >{entity_thresh*100:.2f}% of pred component intersect with gt component)")
    print(f"Entity precision ({tp}/{tp+fp}): {(tp/(tp+fp)):.3f}")
    print(f"Entity recall ({tp}/{tp+fn}): {(tp/(tp+fn)):.3f}")
    print("---------------------------------")
    print(f"(criteria: pred component covers >{(entity_thresh*100):.2f}% of gt component)")
    print(f"Entity precision ({tp_intersect_gt}/{tp_intersect_gt+fp}): {(tp_intersect_gt/(tp_intersect_gt+fp)):.3f}")
    print(f"Entity recall ({tp_intersect_gt}/{tp_intersect_gt+fn}): {(tp_intersect_gt/(tp_intersect_gt+fn)):.3f}")
    precision_gt = tp_intersect_gt/(tp_intersect_gt+fp)
    recall_gt = tp_intersect_gt/(tp_intersect_gt+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision_gt, recall_gt, precision, recall, true_positive_components, true_positive_gt_components, false_positive_components, false_negative_components, all_components

def get_precision_recall_2d(pred_vol, gt_vol, entity_thresh=0.5):
    total_tp=0
    total_fp=0
    total_fn=0
    total_tp_gt=0
    true_positive_components = []
    true_positive_gt_components = []
    false_positive_components = []
    false_negative_components = []
    for k in range(depth):
        pred_img = pred_vol[k]
        gt_img = gt_vol[k]
        gt_img[gt_img != 0] = 255

        labeled_array, num_features = label(pred_img)
        tp = 0
        tp_gt = 0
        fp = 0
        fn = 0
        for i in range(1, num_features + 1):
            component = (labeled_array == i).astype(np.uint8) * 255
            # check intersection of component with gt
            num_intersect = np.count_nonzero(component[gt_img != 0]) # nonzero pixels in component where gt is also nonzero
            num_pred = np.count_nonzero(component)
            proportion_intersect = num_intersect/num_pred
            if proportion_intersect >= entity_thresh:
                true_positive_components.append(component)
                tp += 1
            else:
                false_positive_components.append(component)
                fp += 1

        labeled_array, num_gt_features = label(gt_img)
        for i in range(1, num_gt_features + 1):
            component = (labeled_array == i).astype(np.uint8) * 255
            num_intersect = np.count_nonzero(component[pred_img != 0])
            num_gt = np.count_nonzero(component)
            if num_intersect/num_gt >= entity_thresh:
                true_positive_gt_components.append(component)
                tp_gt += 1
            else:
                false_negative_components.append(component)
                fn += 1
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tp_gt += tp_gt
        print(f"--------Slice {k}--------")
        print(f"Number of entities found | pred: {num_features} | gt: {num_gt_features}")
        print(f"Entity precision ({tp}/{tp+fp}): {tp/(tp+fp):.3f}")
        print(f"Entity recall ({tp}/{tp+fn}): {tp/(tp+fn):.3f}")
    print(f"------------------------Total------------------------")
    print(f"Total Entity precision ({total_tp}/{total_tp+total_fp}): {total_tp/(total_tp+total_fp):.3f}")
    print(f"Total Entity recall ({total_tp}/{total_tp+total_fn}): {total_tp/(total_tp+total_fn):.3f}")
    precision = total_tp/(total_tp+total_fp)
    recall = total_tp/(total_tp+total_fn)
    precision_gt = total_tp_gt/(total_tp_gt+total_fp)
    recall_gt = total_tp_gt/(total_tp_gt+total_fn)
    return precision_gt, recall_gt, precision, recall, true_positive_components, true_positive_gt_components, false_positive_components, false_negative_components


parser = argparse.ArgumentParser()
parser.add_argument("--pred_fp", type=str, help="Path to the directory containing predicted images")
parser.add_argument("--gt_fp", type=str, help="Path to the directory containing ground truth images")
parser.add_argument("--start_z", type=int, help="Starting z-coordinate")
parser.add_argument("--start_x", type=int, help="Starting x-coordinate")
parser.add_argument("--start_y", type=int, help="Starting y-coordinate")
parser.add_argument("--end_z", type=int, help="Ending z-coordinate")
parser.add_argument("--end_x", type=int, help="Ending x-coordinate")
parser.add_argument("--end_y", type=int, help="Ending y-coordinate")


args = parser.parse_args()
pred_fp = args.pred_fp
gt_fp = args.gt_fp
start_z = args.start_z
start_x = args.start_x
start_y = args.start_y
end_z = args.end_z
end_x = args.end_x
end_y = args.end_y
depth, height, width = end_z-start_z, end_y-start_y, end_x-start_x
suffix=f"100_103_x{start_x}-{end_x}_y{start_y}-{end_y}_height{height}xwidth{width}"

# generate section that you're processing
pred_imgs = os.listdir(pred_fp)
pred_imgs = sorted(pred_imgs)
gt_imgs = os.listdir(gt_fp)
gt_imgs = sorted(gt_imgs)
print(len(pred_imgs), len(gt_imgs))

tmp_fp=os.path.join(pred_fp, pred_imgs[0])
tmp_gt_fp=os.path.join(gt_fp, gt_imgs[0])
tmp = cv2.imread(tmp_fp, cv2.IMREAD_GRAYSCALE)
tmp_gt = cv2.imread(tmp_gt_fp, cv2.IMREAD_GRAYSCALE)
print(tmp.shape, tmp_gt.shape)
tmp_gt[start_y:end_y, start_x:end_x] = 255 # draw a red box from start_x to end_x, start_y to end_y
plt.imshow(tmp_gt, cmap="gray")
plt.imshow(tmp, alpha=0.5)
plt.savefig(f"section.png")
plt.close("all")

# slices 100-110
pred_vol = np.zeros((depth, height, width), dtype=np.uint8)
for i in range(depth):
    fp=os.path.join(pred_fp, pred_imgs[i])
    tmp = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    pred_vol[i] = tmp[start_y:end_y, start_x:end_x]
    print("read pred", i, "from ", fp)
gt_vol = np.zeros((depth, height, width), dtype=np.uint8)
for i in range(depth):
    fp=os.path.join(gt_fp, gt_imgs[i])
    tmp = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    gt_vol[i] = tmp[start_y:end_y, start_x:end_x]
    print("read gt", i, "from ", fp)  
gt_vol[gt_vol != 0] = 255
print(np.unique(gt_vol, return_counts=True))

entity_threshes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
precision_dict = {}
recall_dict = {}
precision_gt_dict = {}
recall_gt_dict = {}
start_time = time.time()
for entity_thresh in entity_threshes:
# Label connected components
    precision_gt, recall_gt, precision, recall, true_positive_components, true_positive_gt_components, false_positive_components, false_negative_components, all_components = get_precision_recall(pred_vol, gt_vol, entity_thresh)
    precision_dict[entity_thresh] = precision
    recall_dict[entity_thresh] = recall
    precision_gt_dict[entity_thresh] = precision_gt
    recall_gt_dict[entity_thresh] = recall_gt
    print("(TP)", len(true_positive_components), "(TP gt)", len(true_positive_gt_components), "(FP)", len(false_positive_components), "(FN)", len(false_negative_components))
    print(f"=========================================== entity thresh: {entity_thresh} (time: {(time.time()-start_time):.2f}s)===========================================")


    # plot all pred components
    all_components_vol = np.zeros((depth, height, width, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, (len(all_components), 3))
    for i, component in enumerate(all_components):
        for j in range(3):
            all_components_vol[:, :, :, j] += component * colors[i][j]
        print(f"added component {i}/{len(all_components)}", end="\r")
    suffix=f"100_103_x{start_x}-{end_x}_y{start_y}-{end_y}_height{height}xwidth{width}"
    fig, ax = plt.subplots(1, depth, figsize=(15, 5), num=1)
    visualize_3d_slice(all_components_vol, ax)
    plt.savefig(f"all_components_{suffix}.png")
    plt.close("all")

    # plot the pred components (different colors for tp, fp, fn)
    combined_img = np.zeros((depth, height, width, 3), dtype=np.uint8)
    for i, component in enumerate(true_positive_components):
        combined_img[component != 0] = [0, 255, 0]
    for i, component in enumerate(false_positive_components):
        combined_img[component != 0] = [255, 0, 0]
    for i, component in enumerate(false_negative_components):
        combined_img[component != 0] = [0, 0, 255]    
    fig, ax = plt.subplots(1, depth, figsize=(15, 5), num=1)
    visualize_3d_slice(combined_img, ax)
    plt.suptitle(f"Predicted Entities (TP: pred is >{entity_thresh*100:.2f}% gt)")
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='red', label='False Positive')
    fn_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='upper left')
    plt.savefig(f"tp_{suffix}_{entity_thresh}.png")
    plt.close("all")
    
    # plot the pred gt components (different colors for tp, fp, fn)
    combined_img = np.zeros((depth, height, width, 3), dtype=np.uint8)
    for i, component in enumerate(true_positive_gt_components):
        combined_img[component != 0] = [0, 255, 0]
    for i, component in enumerate(false_positive_components):
        combined_img[component != 0] = [255, 0, 0]
    for i, component in enumerate(false_negative_components):
        combined_img[component != 0] = [0, 0, 255]              
    fig, ax = plt.subplots(1, depth, figsize=(15, 5), num=1)
    visualize_3d_slice(combined_img, ax)
    plt.suptitle(f"Predicted Entities (TP: gt is >{entity_thresh*100:.2f}% pred)")
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='red', label='False Positive')
    fn_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='upper left')
    plt.savefig(f"tp_gt_{suffix}_{entity_thresh}.png")
    plt.close("all")
    
# plot chnge
plt.plot(precision_dict.keys(), precision_dict.values(), label="precision (>x% of pred is gt)")
plt.plot(recall_dict.keys(), recall_dict.values(), label="recall (>x% of pred is gt)")
plt.title(f"Precision and Recall vs Entity Threshold \n {suffix}")
plt.plot(precision_gt_dict.keys(), precision_gt_dict.values(), label="precision (>x% of gt is pred)")
plt.plot(recall_gt_dict.keys(), recall_gt_dict.values(), label="recall (>x% of gt is pred)")
plt.xlabel("Entity Threshold")
plt.ylabel("Value")
plt.legend()
plt.savefig(f"graph_{suffix}.png")

# ===== 2d =====
print("==========================================")
print("====================2D====================")
print("==========================================")
entity_threshes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
precision_dict = {}
recall_dict = {}
precision_gt_dict = {}
recall_gt_dict = {}
start_time = time.time()
for entity_thresh in entity_threshes:
    precision_gt, recall_gt, precision, recall, true_positive_components, true_positive_gt_components, false_positive_components, false_negative_components = get_precision_recall_2d(pred_vol, gt_vol, entity_thresh)
    precision_dict[entity_thresh] = precision
    recall_dict[entity_thresh] = recall
    precision_gt_dict[entity_thresh] = precision_gt
    recall_gt_dict[entity_thresh] = recall_gt
    print(f"=========================================== entity thresh: {entity_thresh} (time: {(time.time()-start_time):.2f}s)===========================================")
    print(len(true_positive_components), len(true_positive_gt_components), len(false_positive_components), len(false_negative_components))
    
    # plot the components (different colors for tp, fp, fn)
    combined_img = np.zeros((depth, height, width, 3), dtype=np.uint8)
    for k in range(depth):
        for i, component in enumerate(true_positive_components):
            combined_img[k][component != 0] = [0, 255, 0]
        for i, component in enumerate(false_positive_components):
            combined_img[k][component != 0] = [255, 0, 0]
        for i, component in enumerate(false_negative_components):
            combined_img[k][component != 0] = [0, 0, 255]
    fig, ax = plt.subplots(1, depth, figsize=(15, 5), num=1)
    visualize_3d_slice(combined_img, ax)
    plt.suptitle(f"2d Predicted Entities (TP: pred is >{entity_thresh*100:.2f}% gt)")
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='red', label='False Positive')
    fn_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='upper left')
    plt.savefig(f"2dtp_{suffix}_{entity_thresh}.png")
    plt.close("all")

    # plot the components (different colors for tp, fp, fn)
    combined_img = np.zeros((depth, height, width, 3), dtype=np.uint8)
    for k in range(depth):
        for i, component in enumerate(true_positive_gt_components):
            combined_img[k][component != 0] = [0, 255, 0]
        for i, component in enumerate(false_positive_components):
            combined_img[k][component != 0] = [255, 0, 0]
        for i, component in enumerate(false_negative_components):
            combined_img[k][component != 0] = [0, 0, 255]
    fig, ax = plt.subplots(1, depth, figsize=(15, 5), num=1)
    visualize_3d_slice(combined_img, ax)
    plt.suptitle(f"Predicted Entities (TP: gt is >{entity_thresh*100:.2f}% pred)")
    tp_patch = mpatches.Patch(color='green', label='True Positive')
    fp_patch = mpatches.Patch(color='red', label='False Positive')
    fn_patch = mpatches.Patch(color='blue', label='False Negative')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='upper left')
    plt.savefig(f"tp_gt_{suffix}_{entity_thresh}.png")
    plt.close("all")

suffix=f"100_103_x{start_x}-{end_x}_y{start_y}-{end_y}_height{height}xwidth{width}"
plt.plot(precision_dict.keys(), precision_dict.values(), label="precision (>x% of pred is gt)")
plt.plot(recall_dict.keys(), recall_dict.values(), label="recall (>x% of pred is gt)")
plt.title(f"2D Precision and Recall vs Entity Threshold \n {suffix}")
plt.plot(precision_gt_dict.keys(), precision_gt_dict.values(), label="precision (>x% of gt is pred)")
plt.plot(recall_gt_dict.keys(), recall_gt_dict.values(), label="recall (>x% of gt is pred)")
plt.xlabel("Entity Threshold")
plt.ylabel("Value")
plt.legend()
plt.savefig(f"graph2d_{suffix}.png")
