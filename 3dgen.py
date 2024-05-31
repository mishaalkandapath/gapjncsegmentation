 base = "E:\\Mishaal\\sem_dauer_2"
    def get_mask(imgname):
         return base+"\\seg_export_full_volume\\"+imgname.replace("SEM_dauer_2_image_export_", "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_")

    def get_seg(imgname):
         return base+"\\seg_export0507_2\\"+imgname.replace("SEM_dauer_2_image_export_", "sem2dauer_gj_2d_training.vsseg_export_")
    pattern = r's\d\d\d'
    def get_another(filename, i):
        i = int(re.findall(pattern, filename)[0][1:]) + i
        assert i>=0
        return filename.replace(re.findall(pattern, filename)[0], "s0"+("0" if i <=9 else 
"") + str(i))
    
    imgs = os.listdir(base+"\\image_export")

    flat_imgs, flat_segs, flat_masks = [], [], []
    seq_imgs, seq_segs, seq_masks = [], [], []

    for img in tqdm(imgs):
        seg = cv2.cvtColor(cv2.imread(get_seg(img)), cv2.COLOR_BGR2GRAY)
        depth = int(re.findall(pattern, img)[0][1:])
        if depth == 0 or depth >48: continue
        if len(np.unique(seg)) >= 2:
            
            flat_masks += [get_mask(img)]
            flat_segs += [get_seg(img)]
            
            seq_masks.append([get_another(get_mask(img), -1), get_mask(img), get_another(get_mask(img), 1), get_another(get_mask(img), 2)])
            seq_segs.append([get_another(get_seg(img), -1), get_seg(img), get_another(get_seg(img), 1), get_another(get_seg(img), 2)])

            img = (base+"\\image_export\\"+img)

            flat_imgs += [img]
            seq_imgs.append([get_another(img, -1), img, get_another(img, 1), get_another(img, 2)])

    for i in tqdm(range(len(seq_imgs))):
        os.mkdir(BASE+"3d_jnc_only\\masks\\"+os.path.split(seq_imgs[i][1])[-1][:-4])
        os.mkdir(BASE+"3d_jnc_only\\gts\\"+os.path.split(seq_imgs[i][1])[-1][:-4])
        os.mkdir(BASE+"3d_jnc_only\\imgs\\"+os.path.split(seq_imgs[i][1])[-1][:-4])

        for j in range(4):
            shutil.copy(seq_imgs[i][j], BASE+"3d_jnc_only\\imgs\\"+os.path.split(seq_imgs[i][1])[-1][:-4]+"\\"+os.path.split(seq_imgs[i][j])[-1])
            # shutil.copy(flat_segs[i], BASE+"3d_jnc_only\\imgs\\"+os.path.split(flat_segs[i]))
            seg = cv2.cvtColor(cv2.imread(seq_segs[i][j]), cv2.COLOR_BGR2GRAY)
            seg *= 255
            cv2.imwrite(BASE+"3d_jnc_only\\gts\\"+os.path.split(seq_imgs[i][1])[-1][:-4]+"\\"+os.path.split(seq_segs[i][j])[-1], seg)
            shutil.copy(seq_masks[i][j], BASE+"3d_jnc_only\\masks\\"+os.path.split(seq_imgs[i][1])[-1][:-4]+"\\"+os.path.split(seq_masks[i][j])[-1].replace(
                "20240325_SEM_dauer_2_nr_vnc_neurons_head_muscles.vsseg_export_", "SEM_dauer_2_image_export_"
            ))

