import os
import json
from numpy.random import permutation
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model
import shutil


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataurl="./data/train/"
    totrainurl="./data/split/train"
    totesturl = "./data/split/test"
    num_train=1600
    num_test=400
    for classname in permutation(os.listdir(dataurl)):
        i=0
        to_trainpath = os.path.join(totrainurl, classname)
        to_testpath = os.path.join(totesturl, classname)
        if not os.path.exists( to_trainpath):
            os.mkdir( to_trainpath)
        if not os.path.exists( to_testpath):
            os.mkdir( to_testpath)
        for img_name in permutation(os.listdir(os.path.join(dataurl,classname))):
            i+=1
            img_path = os.path.join(dataurl,classname,img_name)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            if i>1600:
                shutil.copy(img_path,to_testpath)
                if i==2000:
                    break
                continue
            shutil.copy(img_path, to_trainpath)
            # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
            #                                              predict[predict_cla].numpy())
            # plt.title(print_res)
            # for i in range(len(predict)):
            #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
            #                                               predict[i].numpy()))
            # plt.show()

            print(i)



if __name__ == '__main__':
    main()
