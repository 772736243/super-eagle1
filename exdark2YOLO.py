# author: wujiahao
# last modified: 2021/11/5
#       exdark has some .JPEG or .JPG or .jpg or .png image ,but the responding ann txt is sometimes not correct in name
#       so we should convert the image name first to unified .png, and also convert the txt name
#       and select low light image w.r.t. imagecalsslist.txt
#       and convert annotation format

import os
import cv2
import shutil
import glob

data_root = 'E:\PytorchPro\SuperYOLO-main\dark'  # anndir  imgdir
dst_path = 'E:\PytorchPro\SuperYOLO-main\dark\exdark_txt'
annfold = 'ann'
imgfold = 'img'
txtfile = 'imageclasslist.txt'
class_folders = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People']
class_idx_dict = {
    'Bicycle': 1,
    'Boat': 2,
    'Bottle': 3,
    'Bus': 4,
    'Car': 5,
    'Cat': 6,
    'Chair': 7,
    'Cup': 8,
    'Dog': 9,
    'Motorbike': 10,
    'People': 11,
}
ignore_class = 'Cup'

with open(txtfile, 'r') as ftxt:
    lines = ftxt.readlines()[1:]
for line in lines:  # each img
    words = line.split()
    img = words[0]
    class_folder = class_folders[int(words[1]) - 1]
    file_prefix = img.split('.')[0]
    # copy img and change suffix
    img_file_path = glob.glob(os.path.join(data_root, imgfold, class_folder, file_prefix + '*'))[0]
    file_suffix = img_file_path.split('.')[-1]
    h_img, w_img, c_img = cv2.imread(img_file_path).shape
    if file_suffix in ['jpg', 'JPG', 'JPEG']:
        shutil.copy(img_file_path, os.path.join(dst_path, file_prefix + '.jpg'))
    else:
        shutil.copy(img_file_path, os.path.join(dst_path, file_prefix + '.png'))
    # convert ann
    ann_file_path = glob.glob(os.path.join(data_root, annfold, class_folder, file_prefix + '*'))[0]
    with open(ann_file_path, 'r') as f_in:
        ann_lines = f_in.readlines()[1:]
    dst_file_path = os.path.join(dst_path, file_prefix + '.txt')
    with open(dst_file_path, 'w') as f_out:
        for ann_line in ann_lines:  # each obj in the img
            ann_words = ann_line.split()
            if ann_words[0] == ignore_class:
                continue
            class_idx = class_idx_dict[ann_words[0]]
            bbox = ann_words[1:5]
            l, t, w, h = [float(bbox[i]) for i in range(4)]
            x_center, y_center, w_bbox, h_bbox = (l + w / 2) / w_img, (t + h / 2) / h_img, w / w_img, h / h_img
            outline = f'{class_idx} {x_center} {y_center} {w_bbox} {h_bbox}\n'
            f_out.write(outline)
    print(img_file_path)


