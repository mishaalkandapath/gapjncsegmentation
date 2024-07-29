import cv2, numpy as np, re, os, sys, shutil, glob, traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

#assembling function for full image assembly
"""
Function for assembling the prediction tiles into a single EM image.
@param img_dir: directory containing the original images
@param gt_dir: directory containing the ground truth images
@param pred_dir: directory containing the predicted images
@param save_dir: directory to save the assembled images
@param overlap: whether to overlap the predictions on the EM images
@param missing_dir: directory containing the missing images - when your dataset was created without certain images (e.g. images without any neurons in them)
@param img_templ: template for the image names
@param seg_templ: template for the ground truth image names
@param s_range: range of s values
@param x_range: range of x values
@param y_range: range of y values
@param offset: offset for the overlap (default 256)
@returns: None (saves the images in the save_dir)
"""
def assemble_overlap(img_dir, gt_dir, pred_dir, save_dir, extend_dir=None, overlap=True, missing_dir=None, img_templ="SEM_dauer_2_em_",seg_templ="sem_dauer_2_gj_gt_" , s_range=range(101, 109), x_range=range(0, 19), y_range=range(0, 17), offset=256, fn=None, fn_mask_dir=None, plot_legend=True):
    offset = int(offset)
    if fn: fn_stats = []
    for s in s_range:
        s_acc_img, s_acc_pred, s_acc_gt = [], [], []
        if offset: s_acc_img1, s_acc_pred1, s_acc_gt1 = [], [], []
        for y in tqdm(y_range):
            y_acc_img, y_acc_pred, y_acc_gt = [], [], []
            if offset: y_acc_img1, y_acc_pred1, y_acc_gt1 = [], [], []
            for x in x_range:
                suffix = r"s{}_Y{}_X{}".format('0'*(3-len(str(s)))+str(s), y, x)

                if not os.path.isfile(os.path.join(img_dir, img_templ +suffix + ".png")): 
                    assert missing_dir is not None, f"Missing image {img_templ +suffix + '.png'}"
                    shutil.copy(os.path.join(missing_dir, img_templ + suffix + ".png"), os.path.join(img_dir, img_templ + suffix + ".png"))
                im =cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_templ+ suffix + ".png")), cv2.COLOR_BGR2GRAY)

                try:
                    if gt_dir: gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, seg_templ + suffix + ".png")), cv2.COLOR_BGR2GRAY) 
                    if True:
                        pred = np.load(os.path.join(pred_dir, img_templ + suffix + ".png.npy"))
                        #softmax on channel 1 of size 3
                        pred = 1/(1+np.exp(-pred)) >= 0.5
                    else:
                        pred = cv2.cvtColor(cv2.imread(os.path.join(pred_dir, img_templ + suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                        pred = pred == 255
                    if extend_dir:
                        ext_pred = cv2.imread(os.path.join(extend_dir, img_templ + suffix + ".png"))
                        ext_pred = cv2.cvtColor(ext_pred, cv2.COLOR_BGR2GRAY)
                        ext_pred[ext_pred != 0] = 255
                        pred = pred | (ext_pred == 255)

                    if gt_dir and gt.shape != pred.shape: pred = np.zeros_like(gt)  
                except Exception as e:
                    print(e)
                    if gt_dir: gt = np.zeros_like(im)
                    pred = np.zeros_like(im)

                #just append as is
                y_acc_img.append(im)
                y_acc_pred.append(pred)
                if gt_dir: y_acc_gt.append(gt)

                if offset:
                    try:
                        im1 = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_templ +suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                        if gt_dir: gt1 = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, seg_templ + suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                        if True:
                            pred1 = np.load(os.path.join(pred_dir, img_templ + suffix + "off.png.npy"))
                            pred1 = 1/(1+np.exp(-pred1)) >= 0.5
                            # pred1 = np.argmax(pred, axis=0)
                            # pred1[pred1 != 1] = 0
                        else:
                             pred1 = cv2.cvtColor(cv2.imread(os.path.join(pred_dir, img_templ + suffix + "off.png")), cv2.COLOR_BGR2GRAY)
                             pred1 = pred1 == 255
                        if extend_dir:
                            try:
                                ext_pred1 = cv2.imread(os.path.join(extend_dir, img_templ + suffix + "off.png"))
                                ext_pred1 = cv2.cvtColor(ext_pred1, cv2.COLOR_BGR2GRAY)
                                ext_pred1[ext_pred1 != 0] = 255
                                pred1 = pred1 | (ext_pred1 == 255)
                            except:
                                pass

                        # if im1 is None or gt1 is None or pred1 is None: raise Exception
                        if gt_dir and gt1.shape != pred1.shape: pred1 = np.zeros_like(gt1)  
                        y_acc_img1.append(im1)
                        y_acc_pred1.append(pred1)
                        if gt_dir: y_acc_gt1.append(gt1)
                    except Exception as e:
                        print(e)
                        gt1 = np.zeros_like(im1)
                        pred1 = np.zeros_like(im1)
                        # continue

            s_acc_img.append(np.concatenate(y_acc_img, axis=1))
            s_acc_pred.append(np.concatenate(y_acc_pred, axis=1))
            if gt_dir: s_acc_gt.append(np.concatenate(y_acc_gt, axis=1))
            if offset:
                try:
                    s_acc_img1.append(np.concatenate(y_acc_img1, axis=1))
                    s_acc_pred1.append(np.concatenate(y_acc_pred1, axis=1))
                    if gt_dir: s_acc_gt1.append(np.concatenate(y_acc_gt1, axis=1))
                except Exception as e:
                    print(e)
                    continue

        new_img = np.concatenate(s_acc_img, axis=0)
        new_pred = np.concatenate(s_acc_pred, axis=0)
        if gt_dir: new_gt = np.concatenate(s_acc_gt, axis=0)

        if offset:
            new_img1 = np.concatenate(s_acc_img1, axis=0)
            new_pred1 = np.concatenate(s_acc_pred1, axis=0)
            if gt_dir: new_gt1 = np.concatenate(s_acc_gt1, axis=0)

        #calculate and color statsitic
        if gt_dir:
            new_gt[(new_gt == 2) | (new_gt == 15)] = 0
            new_gt_conf = new_gt.copy()
            new_gt[new_gt != 0] = 1

        if offset and gt_dir:
            new_gt1[(new_gt1 == 2) | (new_gt1 == 15)] = 0
            new_gt1[new_gt1 != 0] = 1
        
        new_pred[new_pred !=0 ] = 1
        if offset: new_pred1[new_pred1 !=0 ] = 1
        if offset:
            if gt_dir: new_gt[offset:-offset, offset:-offset] = new_gt[offset:-offset, offset:-offset] | new_gt1
            try:
                new_pred[offset:-offset, offset:-offset] = new_pred[offset:-offset, offset:-offset] | new_pred1
            except:
                print(new_pred.shape, new_pred1.shape)
                sys.exit()

        new_img1 = np.repeat(new_img[:, :, np.newaxis], 3, axis=-1)
        if gt_dir: new_gt1 = np.repeat(new_gt[:, :, np.newaxis], 3, axis=-1)
        new_pred1 = np.repeat(new_pred[:, :, np.newaxis], 3, axis=-1).astype(np.uint8)            

        if not fn and plot_legend:
            # else: 
            #color statistics now
            red = (0, 0, 255)
            green = (0, 255, 0)
            blue = (255, 0, 0)
            for m in range(3):
                new_pred1[(new_pred == 1) & (new_gt == 1)] = green
                new_pred1[(new_pred == 1) & (new_gt == 0)] = red
                new_pred1[(new_pred == 0) & (new_gt == 1)] = blue
            
            #color confidence levels
            conf5 = (255, 0, 255)
            conf4 = (255, 255, 0)
            conf3 = (0, 255, 255)
            conf2 = (0, 165, 255)
            conf1 = (255, 165, 0)

            #make labels
            for m in range(3):
                new_gt1[new_gt_conf == 3] = conf1
                new_gt1[new_gt_conf == 5] = conf2
                new_gt1[new_gt_conf == 7] = conf3
                new_gt1[new_gt_conf == 9] = conf4
                new_gt1[new_gt_conf == 11] = conf5

            #make legends
            def write_legend(text, color, img, org):
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 

                
                # fontScale 
                fontScale = 10
                
                # Line thickness of 2 px 
                thickness = 20

                return cv2.putText(img, text, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)

            if overlap:
                new_pred1 = cv2.addWeighted(new_img1, 0.5, new_pred1, 0.6, 0)
                new_gt1 = cv2.addWeighted(new_img1, 0.5, new_gt1, 0.6, 0)
            
            for color in [conf1, conf2, conf3, conf4, conf5]:
                new_gt = write_legend(f"Confidence {[conf1, conf2, conf3, conf4, conf5].index(color)}", color, new_gt1, (450, 450+250*[conf1, conf2, conf3, conf4, conf5].index(color)))
            
            for color in [green, red, blue]:
                new_pred = write_legend(f"{['TP', 'FP', 'FN'][[green, red, blue].index(color)]}", color, new_pred1, (450, 450+250*[green, red, blue].index(color)))
            
            #write them all in
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            assert cv2.imwrite(os.path.join(save_dir, "SEM_dauer_2_image_export_" + suffix + "_img.png"), new_img1)
            assert cv2.imwrite(os.path.join(save_dir, "SEM_dauer_2_image_export_" + suffix + "_pred.png"), new_pred1)
            assert cv2.imwrite(os.path.join(save_dir, "SEM_dauer_2_image_export_" + suffix + "_gt.png"), new_gt1)
        elif not fn and not plot_legend:
            if gt_dir: new_gt1[new_gt == 1] = (255, 255, 255)
            new_pred1[new_pred == 1] = (255, 0, 0)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            assert cv2.imwrite(os.path.join(save_dir, "SEM_dauer_2_image_export_" + suffix + "_img.png"), new_img1)
            assert cv2.imwrite(os.path.join(save_dir, "SEM_dauer_2_image_export_" + suffix + "_pred.png"), new_pred1)
            if gt_dir: assert cv2.imwrite(os.path.join(save_dir, "SEM_dauer_2_image_export_" + suffix + "_gt.png"), new_gt1)
        else: 
            mask = glob.glob(fn_mask_dir+f"*s{'0'*(3-len(str(s)))}{s}*")[0]
            mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY)  
            mask = mask == 21
            fn_stats.append(fn(new_gt_conf, new_pred, nr_mask=mask))

    if fn: return fn_stats

#statistics
def mask_acc(gt, pred, mask=None):
    gt = gt.flatten()[mask]
    pred = pred.flatten()[mask]
    gt[gt != 0] = 255  

    return np.sum(gt == pred) / (gt.shape[0])
    # return np.sum(gt == pred) / (gt.shape[0] * gt.shape[1])

def mask_precision(gt, pred, mask=None):

    gt = gt.flatten()[mask]
    pred = pred.flatten()[mask]
    gt[gt != 0] = 255  

    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(pred == 255)

def move_generous(img, dims, itr):
    accum = img.copy()
    for i in range(min(itr, 0), max(itr, 0)):
        accum = accum | np.roll(img, i+1, dims)
    return accum

def mask_precision_generous(gt, pred, mask=None):
    new_gt = np.pad(gt, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
    new_gt = move_generous(new_gt, 0, 10) | move_generous(new_gt, 1, 10) | move_generous(new_gt, 1, -10) | move_generous(new_gt, 0, -10)
    new_gt = new_gt[10:-10, 10:-10]

    gt = gt.flatten()[mask]
    pred = pred.flatten()[mask]
    new_gt = new_gt.flatten()[mask]
    new_gt[new_gt != 0] = 255
    gt[gt != 0] = 255     

    num = (np.logical_and(gt == 255, pred == 255) | np.logical_and(new_gt == 255, pred == 255)) >= 1
    denom = np.sum(pred == 255)
    #remove mask items.
    # print(num.flatten().shape, mask.flatten().shape)
    # if mask is not None:
    #     num = num.flatten()[mask.flatten()]
    #     denom = np.count_nonzero(pred.flatten()[mask.flatten()] == 255)
    return np.sum(num)/denom 

def mask_acc_generous(gt, pred, mask=None):
    new_gt = np.pad(gt, ((10, 10), (10, 10)), 'constant', constant_values=(0, 0))
    new_gt = move_generous(new_gt, 0, 10) | move_generous(new_gt, 1, 10) | move_generous(new_gt, 1, -10) | move_generous(new_gt, 0, -10)
    new_gt = new_gt[10:-10, 10:-10]
    
    gt = gt.flatten()[mask]
    pred = pred.flatten()[mask]
    new_gt = new_gt.flatten()[mask]
    new_gt[new_gt != 0] = 255
    gt[gt != 0] = 255         

    num = ((new_gt == pred) +(gt == pred)) >= 1
    denom = (gt.shape[0])
    return np.sum(num)/denom

def iou_accuracy(gt, pred, mask=None):
    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(np.logical_or(gt == 255, pred == 255))

def mask_recall(gt, pred, mask=None):
    print(gt.shape, pred.shape, mask.shape)
    gt = gt.flatten()[mask]
    pred = pred.flatten()[mask]
    gt[gt != 0] = 255   

    return np.sum(np.logical_and(gt == 255, pred == 255)) / np.sum(gt == 255)
    


    
"""
@params: gt_image: ground truth image
@params: preds_image: predicted image
@params: nr_mask: mask for the non-relevant areas
@params: seg_bads: tuple of segmentation labels to ignore in statistics (e.g GJs you are not sure about)
@returns: recall, precision, precision_gen, acc, acc_gen
"""
def assembled_stats(gt_image, preds_image, nr_mask=None, seg_bads=(2, 15)):
    if seg_bads:
        mask = gt_image != seg_bads[0]
        for i in seg_bads[1:]:
            mask = mask & (mask != i)
        mask = mask.flatten()

    good_mask = np.logical_and(gt_image != 2, gt_image != 15).flatten() # TODO: figure this out for general case
    assert np.count_nonzero(good_mask) == np.count_nonzero(mask)

    if nr_mask is not None:
        nr_mask = np.logical_and(good_mask, nr_mask.flatten())
    else: nr_mask = np.ones_like(good_mask)

    if 1 in np.unique(preds_image):
        preds_image *= 255

    recall = mask_recall(gt_image, preds_image, mask=nr_mask)
    precision = mask_precision(gt_image, preds_image, mask=nr_mask)
    precision_gen = mask_precision_generous(gt_image, preds_image, mask=nr_mask)
    acc = mask_acc(gt_image, preds_image, mask=nr_mask)
    acc_gen = mask_acc_generous(gt_image, preds_image, mask=nr_mask)

    return recall, precision, precision_gen, acc, acc_gen

#get stats on a split dataset:
def mask_acc_split(seg_dir, results_dir, nr_mask_dir=None, td=False, breakdown=False, preds_to_seg_map=lambda x:x, preds_to_nr_map=lambda x:x):
    segs = sorted(os.listdir(seg_dir))
    preds = sorted(os.listdir(results_dir))

    t_recall, t_acc = [], []
    t_acc_gen = []
    t_prec, t_prec_gen = [], []

    t_45_recall, t_3_recall, t_2_recall, t_1_recall = [], [], [], []

    weird_shape_count = 0

    for i, file in tqdm(enumerate(preds), total=len(preds)): 
        if "DS" in file: continue #villainous
        try:
            segs[i] = os.path.join(seg_dir, preds_to_seg_map(file))
            gt = cv2.cvtColor(cv2.imread(segs[i]), cv2.COLOR_BGR2GRAY)
            no = re.findall(r'_\d+', file)[0]
            pred = cv2.cvtColor(cv2.imread(os.path.join(results_dir, file)), cv2.COLOR_BGR2GRAY)

            #area mask 
            if nr_mask_dir:
                mask = cv2.cvtColor(cv2.imread(os.path.join(nr_mask_dir, preds_to_nr_map(file))), cv2.COLOR_BGR2GRAY)
                mask = mask == 21
            else: mask = np.ones_like(gt)
            if breakdown:
                test_gt = gt.copy()

            t_acc.append(mask_acc(gt, pred, mask=mask))
            t_recall.append(mask_recall(gt, pred, mask=mask))
            t_acc_gen.append(mask_acc_generous(gt, pred, mask=mask))
            t_prec_gen.append(mask_precision_generous(gt, pred, mask=mask))
            t_prec.append(mask_precision(gt, pred, mask=mask))

            if breakdown:
                t_45_recall.append(mask_recall((np.where(test_gt == 11, 1, 0) | np.where(test_gt == 9, 1, 0)) * 255, pred, mask=mask))
                t_3_recall.append(mask_recall(np.where(test_gt == 7, 255, 0), pred, mask=mask))
                t_2_recall.append(mask_recall(np.where(test_gt == 5, 255, 0), pred, mask=mask))
                t_1_recall.append(mask_recall(np.where(test_gt == 3, 255, 0), pred, mask=mask))

        except Exception:
            segs[i] = os.path.join(seg_dir, preds_to_seg_map(file))
            gt = cv2.cvtColor(cv2.imread(segs[i]), cv2.COLOR_BGR2GRAY) 
            assert not np.count_nonzero(gt[(gt != 2) & (gt != 15)])
            weird_shape_count +=1

            print(traceback.format_exc())
    
    print("Training set mask accuracy {}".format(np.nanmean(t_acc)))
    print("Training set mask recall {}".format(np.nanmean(t_recall)))
    print("Training set mask accuracy generous {}".format(np.nanmean(t_acc_gen)))
    print("Training set mask prec {}".format(np.nanmean(t_prec)))
    print("Training set mask prec generous {}".format(np.nanmean(t_prec_gen)))

    print("Training set mask recall 45 {}".format(np.nanmean(t_45_recall)))
    print("Training set mask recall 3 {}".format(np.nanmean(t_3_recall)))
    print("Training set mask recall 2 {}".format(np.nanmean(t_2_recall)))
    print("Training set mask recall 1 {}".format(np.nanmean(t_1_recall)))

def center_img(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.zeros_like(gray, dtype=np.uint32)


    # Apply thresholding
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate center of each contour
    centers = []
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx, cy))
            color = random.randint(0, 255)
            gray = cv2.drawContours(gray, contours, i, color=30, thickness=-1)
            out[gray == 30] = i+1
            gray[gray == 30] = 1
            assert not np.count_nonzero(gray == 30)
    return centers, out

def entity_recall(gt_contour, pred):
    contours = np.unique(gt_contour)
    contours = contours[contours != 0]
    num_right = 0
    for i , cnt in enumerate(contours):
        if cnt == 0: continue
        num = np.count_nonzero(gt_contour == cnt)
        num_intersect = np.count_nonzero(pred[gt_contour == cnt] == 255)
        if num_intersect/num >= 0.5: 
            num_right +=1
    return num_right/(len(contours)-1)


if __name__ == "__main__":
    base = "/Volumes/Normal/gapjnc/resuklts/assembled_2d_ent_recall_test/"
    # img = cv2.imread(base+"SEM_dauer_2_image_export_s101_Y16_X18_gt.png")

    files = glob.glob(base+"SEM_dauer_2_image_export_*pred.png")
    rec = []
    for file in tqdm(files):
        gt = cv2.imread(file.replace("pred", "gt"))
        _, gt_cont = center_img(gt)
        pred = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        pred[pred != 0] = 255
        recall = entity_recall(gt_cont, pred)
        print(recall)
        rec.append(recall)
    print(np.nanmean(rec))

    #test overlay:
    # img1 = "/Volumes/Normal/gapjnc/resuklts/assembled_finetuned_aug_2d_sem_dauer_1/SEM_dauer_2_image_export_s113_Y16_X18_pred.png"
    # img2 = "/Volumes/Normal/gapjnc/resuklts/assembled_finetuned_aug_2d_sem_dauer_1/SEM_dauer_2_image_export_s113_Y16_X18_img.png"

    # #overlap img1 on img2
    # img1 = cv2.imread(img1)
    # img2 = cv2.imread(img2)
    # img1 = cv2.addWeighted(img2, 0.5, img1, 0.6, 0)
    # cv2.imwrite("overlap1.png", img1)

