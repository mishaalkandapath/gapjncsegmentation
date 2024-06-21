import cv2, re, os, sys, shutil, numpy as np
from tqdm import tqdm
from postprocessing import center_img


"""
Split a full EM image into tiles of size @tile_size, with an optional offset from the beginning of an image of @offset pixels
@param img: the image to split
@param offset: the offset from the beginning of the image
@param tile_size: the size of the tiles
@param names: whether to return the names of the tiles
@return: a list of the tiles, and optionally the names of the tiles
"""
def split_img(img, offset=256, tile_size=512, names=False):
    if offset:
        img = img[offset:-offset, offset:-offset]
    imgs = []
    names_list = []
    for i in range(img.shape[0]//tile_size+1):
        for j in range(img.shape[1]//tile_size +1):
            imgs.append(img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size])
            names_list.append("Y{}_X{}".format(i, j))
    return (imgs, names_list) if names else imgs 

"""
Given an image at a certain cross_section, get corresponding area at a different cross-section at offset i from current depth
@param filename: the filename of the image
@param i: the offset from the current depth: <0 for before, >0 for after
@param pattern: the pattern to match the depth
@return: the filename of the image at the new depth
"""
def get_another(filename, i, pattern=r's\d\d\d'):
        i = int(re.findall(pattern, filename)[0][1:]) + i
        assert i>=0
        return filename.replace(re.findall(pattern, filename)[0], "s"+("0" if i <=9 else 
    "") + str(i))

"""
Function to create a 3d dataset from a 2d, aka flat, dataset
@param flat_dataset_dir: the directory of the flat dataset
@param output_dir: the directory to output the 3d dataset
@param window: the context window to use for the 3d dataset, only one dimension can be set to 1
@param image_to_seg_name_map: the mapping from image to segmentation name
@param add_dir: the additional directories to include in the dataset
@param add_dir_maps: the mapping from image to additional data directory
@param depth_pattern: the pattern to match the depth
@param test: whether to run in test mode - just an images dataset
@return: None (creates and saves your dataset to specified directory)
"""
def create_dataset_3d(flat_dataset_dir, output_dir, window=(0, 1, 0, 0), image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, depth_pattern=r's\d\d\d', test=False):
    assert window.count(1) == 1, "Only one dimension can be set to 1"

    imgs = os.listdir(os.path.join(flat_dataset_dir, "imgs"))

    flat_imgs, flat_segs, flat_adds = [], [], []
    seq_imgs, seq_segs, seq_adds = [], [], []

    min_depth = min([int(re.findall(depth_pattern, img)[0][1:]) for img in imgs])
    max_depth = max([int(re.findall(depth_pattern, img)[0][1:]) for img in imgs])

    central_pos = window.index(1)

    def helper_for_another(img, name_map):
        temp_seq = []
        for i in range(len(window)):
            if i < central_pos:
                temp_seq.append(get_another(name_map(img), i-central_pos))
            elif i == central_pos:
                temp_seq.append(name_map(img))
            else:
                temp_seq.append(get_another(name_map(img), i-central_pos))
        return temp_seq

    for img in tqdm(imgs):
        depth = int(re.findall(depth_pattern, img)[0][1:])
        if depth < min_depth + central_pos or depth > max_depth - len(window) + central_pos+1: continue
                
        # flat_masks += [get_mask(img)]
        if not test: flat_segs += [image_to_seg_name_map(img)]

        img = (os.path.join(flat_dataset_dir, "imgs", img))
        flat_imgs += [img]

        if not test: seq_segs.append(helper_for_another(img, image_to_seg_name_map))
        seq_imgs.append(helper_for_another(img, lambda x: x))
        
        if not test:
            for i in add_dir_maps:
                flat_adds.append(add_dir_maps[i](img))
                seq_adds.append(helper_for_another(img, add_dir_maps[i]))

    for i in tqdm(range(len(seq_imgs))):
        os.makedirs(os.path.join(output_dir, "imgs", os.path.split(seq_imgs[i][1])[-1][:-4]))
        if not test:
            os.makedirs(os.path.join(output_dir, "gts", os.path.split(seq_imgs[i][1])[-1][:-4]))
            for j in add_dir_maps:
                os.makedirs(os.path.join(output_dir, os.path.split(j)[-1], os.path.split(seq_imgs[i][1])[-1][:-4]))

        for j in range(4):
            shutil.copy(os.path.join(flat_dataset_dir, "imgs", seq_imgs[i][j]), os.path.join(output_dir, "imgs", os.path.split(seq_imgs[i][1])[-1][:-4], os.path.split(seq_imgs[i][j])[-1]))
            if not test:
                shutil.copy(os.path.join(flat_dataset_dir, "gts", seq_segs[i][j]), os.path.join(output_dir, "gts", os.path.split(seq_imgs[i][1])[-1][:-4], os.path.split(seq_segs[i][j])[-1]))
                for k in add_dir:
                    shutil.copy(os.path.join(flat_dataset_dir, os.path.split(k)[-1], seq_adds[i][j]), os.path.join(output_dir, os.path.split(k)[-1], os.path.split(seq_imgs[i][1])[-1][:-4], os.path.split(seq_adds[i][j])[-1]))
    
"""
Function to create a 2d dataset from a dataset of full EM images
@param imgs_dir: the directory of the full EM images
@param output_dir: the directory to output the 2d dataset
@param seg_dir: the directory of the segmentations
@param img_size: the size of the images
@param image_to_seg_name_map: the mapping from image to segmentation name
@param add_dir: the additional directories to include in the dataset
@param add_dir_maps: the mapping from image to additional data directory
@param create_overlap: whether to create overlapping tiles - an offset of img_size//2 is used -- helps with predictions since every GJ is given a centered context
@param seg_ignore: the values to ignore in the segmentation
@param test: whether to run in test mode - just an images dataset
@return: None (creates and saves your dataset to specified directory)
"""
def create_dataset_2d_from_full(imgs_dir, output_dir, seg_dir=None, img_size=512, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, create_overlap=False, seg_ignore=(2, 15), test=False):

    assert (add_dir is None and add_dir_maps is None or add_dir is not None and add_dir_maps is not None), "Missing additional directory name mapping for additional data directories, or vice versa"
    if not test and image_to_seg_name_map is None:
        print("WARNING: No image to segmentation name mapping provided, assuming the default naming convention")
        image_to_seg_name_map = lambda x: x.replace('img', 'seg')

    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, deleting it")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -rf {output_dir}")
        else:
            sys.exit(0)

    os.makedirs(output_dir)
    #make subdirs
    os.makedirs(os.path.join(output_dir, "imgs"))
    if not test:
        os.makedirs(os.path.join(output_dir, "gts"))
        if add_dir:
            for i in (add_dir):
                os.makedirs(os.path.join(output_dir, os.path.split(i)[-1]))

    imgs = sorted(os.listdir(imgs_dir))

    def split_subroutine(img, gt, offset=0):

            img_imgs, img_names = split_img(img, offset=offset, tile_size=img_size, names=True)
            if not test: 
                gt_imgs = split_img(gt, offset=offset, tile_size=img_size)
                add_data = [] if add_dir else None
                if add_dir:
                    for j in range(len(add_dir)):
                        dat = cv2.imread(os.path.join(add_dir[j], add_dir_maps[add_dir[j]](imgs[i])))
                        dat = split_img(dat, offset=offset, tile_size=img_size)
                        add_data.append(dat)

            for j in range(len(img_imgs)):
                cv2.imwrite(os.path.join(output_dir, f"imgs/{imgs[i].replace('.png', '_'+img_names[j] + ('' if not offset else 'off'))}.png"), img_imgs[j])
                if not test:
                    cv2.imwrite(os.path.join(output_dir, f"gts/{imgs[i].replace('.png', '_'+img_names[j] + ('' if not offset else 'off'))}.png"), gt_imgs[j])
                    if add_dir:
                        for k in range(len(add_data)):
                            assert cv2.imwrite(os.path.join(output_dir, f"{os.path.split(add_dir[k])[-1]}/{imgs[i].replace('.png', '_'+img_names[j] + ('' if not offset else 'off'))}.png"), add_data[k][j])

    for i in tqdm(range(len(imgs))):
        img = cv2.imread(os.path.join(imgs_dir, imgs[i]))
        if not test: 
            gt = cv2.imread(os.path.join(seg_dir, image_to_seg_name_map(imgs[i])))

            #make changes in the gt
            for ig in seg_ignore:
                gt[gt == ig] = 0

            gt[gt!=0] = 255
        else: gt = None

        split_subroutine(img, gt)
        
        # make the test overlapping mode 
        if create_overlap:
            split_subroutine(img, gt, offset=img_size//2)
    
"""
Function to create a train-val split of a dataset. Given the highly-shared structure of C. elegans, we split the dataset using defined chunks to reduce
the amount of structure shared between the train and validation sets.

@param dataset_dir: the directory of the dataset
@param output_dir: the directory to output the split
@param td: whether to copy the directories as trees
@param filter_neurons: whether to filter out images without neurons
@param filter_gj: whether to filter out images without GJs
@param image_to_seg_name_map: the mapping from image to segmentation name
@param add_dir: the additional directories to include in the dataset
@param add_dir_maps: the mapping from image to additional data directory
@return: None (creates and saves your dataset to specified directory)
"""
def create_train_val_split(dataset_dir, output_dir, td=False, filter_neurons=False, filter_gj=False, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None): 
    
    # TODO: how to specify the splitting of the worm?
    s_set = set()
    x_set = set()
    y_set = set()

    imgs = sorted(os.listdir(os.path.join(dataset_dir, "imgs/")))

    extras = add_dir
    assert len(extras) == len(add_dir_maps)

    train_imgs, train_segs, train_adds = [], [], {}
    valid_imgs, valid_segs, valid_adds = [], [], {}
    for img in imgs:
        s, y, x = re.findall("s\d\d\d", img), re.findall("Y\d+_", img), re.findall("X\d+", img)
        #open the seg and check if any gjs:
        
        if filter_gj:
            seg = cv2.imread(image_to_seg_name_map(img), cv2.COLOR_BGR2GRAY)
            if not np.any(seg != 0): continue
        
        if filter_neurons:
            key = [k for k in extras if "neur" in k][0]
            neur = cv2.imread(add_dir_maps[key](img), cv2.COLOR_BGR2GRAY)
            if not np.any(neur != 0): continue
        
        if s in s_set and y in y_set and x in x_set:
            #validation mode
            valid_imgs += [img]
            valid_segs += [image_to_seg_name_map(img)]
        else:
            train_imgs +=[img]
            train_segs+=[image_to_seg_name_map(img)]
    for j in extras:
        train_extra_dirs, valid_extra_dirs = []
        for img in imgs:
            s, y, x = re.findall("s\d\d\d", img), re.findall("Y\d+_", img), re.findall("X\d+", img)
            if s in s_set and y in y_set and x in x_set:
                #validation mode
                valid_extra_dirs += [add_dir_maps[j](img)]
            else:
                train_extra_dirs+=[add_dir_maps[j](img)]
        train_adds[j] = (train_extra_dirs)
        valid_adds[j] = (valid_extra_dirs)
    
    #write stuff
    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, deleting it")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -rf {output_dir}")
        else:
            sys.exit(0)
    
    os.mkdir(os.path.join(output_dir, "train_imgs/"))
    os.mkdir(os.path.join(output_dir, "train_segs/"))
    os.mkdir(os.path.join(output_dir, "valid_imgs/"))
    os.mkdir(os.path.join(output_dir, "valid_segs/"))
    for j in extras:
        os.mkdir(os.path.join(output_dir, f"train_{j}/"))
        os.mkdir(os.path.join(output_dir, f"valid_{j}/"))

    copyfn = lambda src, dest: shutil.copy(src, dest) if not td else shutil.copytree(src, dest)
    
    for i in range(len(train_imgs)):
        copyfn(os.path.join(dataset_dir, "imgs", train_imgs[i]), os.path.join(output_dir, "imgs", train_imgs[i]))
        copyfn(os.path.join(dataset_dir, "segs", train_segs[i]), os.path.join(output_dir, "segs", train_imgs[i]))
        for j in extras:
            copyfn(os.path.join(dataset_dir, f"{j}", train_adds[j][i]), os.path.join(output_dir, f"{j}", train_adds[j][i]))
    for i in range(len(valid_imgs)):
        copyfn(os.path.join(dataset_dir, "imgs", valid_imgs[i]), os.path.join(output_dir, "imgs", valid_imgs[i]))
        copyfn(os.path.join(dataset_dir, "segs", valid_segs[i]), os.path.join(output_dir, "segs", valid_imgs[i]))
        for j in extras:
            copyfn(os.path.join(dataset_dir, f"{j}", valid_adds[j][i]), os.path.join(output_dir, f"{j}", valid_adds[j][i]))
    
def create_entity_sequence_dataset(imgs_dir, output_dir, seg_dir=None, img_size=512, image_to_seg_name_map=None, add_dir=None, add_dir_maps=None, seg_ignore=(2, 15)):
    assert (add_dir is None and add_dir_maps is None or add_dir is not None and add_dir_maps is not None), "Missing additional directory name mapping for additional data directories, or vice versa"
    if image_to_seg_name_map is None:
        print("WARNING: No image to segmentation name mapping provided, assuming the default naming convention")
        image_to_seg_name_map = lambda x: x.replace('img', 'seg')

    if os.path.isdir(output_dir):
        print("WARNING: Output directory already exists, deleting it")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"rm -rf {output_dir}")
        else:
            sys.exit(0)
    
    os.makedirs(output_dir)
    #make subdirs
    os.makedirs(os.path.join(output_dir, "imgs"))
    gj_set = set()
    imgs = sorted(os.listdir(imgs_dir))
    
    prev_gt = None
    for i in tqdm(range(len(imgs))):
        img = cv2.imread(os.path.join(imgs_dir, imgs[i]))
        gt = cv2.imread(os.path.join(seg_dir, image_to_seg_name_map(imgs[i])))
         
        # get the contours and centers:
        gt_contours, gt_centers = center_img(gt)
        
        if i != 0:
            #trace 
            new_gt_contours = np.zeros_like(gt_contours)
            for c in gj_set:
                temp_gt = gt_contours != 0
                temp_prev_gt = prev_gt == c
                
                #which center do they belong to?
                curr_no = np.unique(gt_contours[temp_gt == temp_prev_gt])[1:]
                for element in curr_no:
                    new_gt_contours[gt_contours == element] = c
            
            #everything has been assigned, what is leftover?
            leftover = np.unique(gt_contours * (new_gt_contours == 0))
            leftover = leftover[leftover != 0]
            if len(leftover) >= 0:
                for element in leftover:
                    gj_set.add(len(gj_set) + 1)
                    new_gt_contours[gt_contours == element] = len(gj_set) + 1
            cv2.imwrite(os.path.join(output_dir, f"imgs/{imgs[i]}"), img)

        else:
            contours = np.unique(gt_contours)
            contours = contours[contours != 0]
            gj_set = set(contours)
            prev_gt = gt_contours
            cv2.imwrite(os.path.join(output_dir, f"imgs/{imgs[i]}"), img)
        