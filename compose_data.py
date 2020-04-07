import os
import cv2
import numpy as np
import copy
import json
background_dir = 'F:\dataset\HuiKe_data\ComposeData\\background'#背景文件夹
block_dir = 'F:\dataset\HuiKe_data\ComposeData\\block'#泡沫文件夹
save_img_dir = 'F:\dataset\HuiKe_data\ComposeData\ComposedData\img'
save_json_dir = 'F:\dataset\HuiKe_data\ComposeData\ComposedData\json'

background_types = ["desk", "testtable", "floor"]#背景的类型
scales = [1/7, 1/5]#缩放尺度

def combine(background, block, point):

    bg = background[:, :, :]


    x, y = point
    rows, cols, channels = block.shape
    roi = bg[x:(rows+x), y:(cols+y)]  # 获得bg的ROI

    img2gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)  # 颜色空间的转换
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)  # 掩码 黑色
    mask_inv = cv2.bitwise_not(mask)# 掩码取反 白色

    img1_bg = cv2.bitwise_and(block, block, mask=mask_inv)
    img2_fg = cv2.bitwise_and(roi, roi, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)

    bg[x:(rows+x), y:(cols+y)] = dst
    return bg
i = 0
for type in background_types:#对不同的类型循环
    type_dir = os.path.join(background_dir, type)
    for bg_name in os.listdir(type_dir):
        bg_path = os.path.join(type_dir, bg_name)
        background = cv2.imread(bg_path)
        # bg = background[:]
        try:
            bg_height, bg_width, bg_channel = background.shape
        except:
            os.remove(bg_path)
            continue
        assert bg_channel==3
        midx = int(bg_height/2)
        midy = int(bg_width/2)
        point = [midx, midy]
        for scale in scales:#对不同尺度循环
            img_anno = {}
            for block_name in os.listdir(block_dir):#对不同的泡沫循环
                i += 1
                print("type: {}, background: {}, block: {}, scale: {} ".format(type, bg_name, block_name, scale))
                bg = copy.deepcopy(background)

                block_path = os.path.join(block_dir, block_name)
                block = cv2.imread(block_path)
                block_height, block_width, block_channel = block.shape
                assert block_channel == 3
                try:
                    new_block = cv2.resize(block, (int(bg_height * scale), int(bg_width * scale)))#resize block
                except:
                    print(block_name)
                    continue


                bg = combine(bg, new_block, point)
                save_name = "%05d.jpg" % (i)
                # save_name = "{}_{}_{}_{}".format(type, block_name.split('.')[0], "%.2f"%scale, i)
                img_path = os.path.join(save_img_dir, save_name + '.jpg')

                img_anno['type'] = type
                img_anno['background'] = bg_name
                img_anno['block'] = block_name
                img_anno['scale'] = scale
                img_anno['point'] = point
                img_anno['shape'] = background.shape
                boundingbox = {
                    'xmin': midx,
                    'ymin': midy,
                    'xmax': midx +int(bg_width * scale),
                    'ymax': midy + int(bg_height * scale)
                }
                img_anno['boundingbox'] = boundingbox


                json_path = os.path.join(save_json_dir, save_name + '.json')#保存json
                with open(json_path, 'w') as f:
                    json.dump(img_anno, f, indent=4)
                cv2.imwrite(img_path, bg)#保存图片











