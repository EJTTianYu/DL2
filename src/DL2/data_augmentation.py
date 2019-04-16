# coding: utf-8

from PIL import Image
import os
import random
import os.path

# rootdir = r'D:\用户目录\我的图片\From Yun\背景图\背景图'  # 指明被遍历的文件夹
rootdir = r'/data/gouwang/DL2/DLdataset/train'

for parent, dirnames, filenames in os.walk(rootdir):  # 遍历图片

    for filename in filenames:
        tmp = random.random()
        if (tmp > 0.75):
            print('parent is :' + parent)
            print('filename is :' + filename)
            currentPath = os.path.join(parent, filename)
            print('the fulll name of the file is :' + currentPath)

            im = Image.open(currentPath)
            out = im.transpose(Image.FLIP_LEFT_RIGHT)  # 实现翻转
            # newname=r"D:\用户目录\我的图片\From Yun\背景图\背景图反转"+'\\'+filename+"(1).jpg"
            newname = parent + "/data_arg" + filename  # 重新命名
            out.save(newname)  # 保存结束
