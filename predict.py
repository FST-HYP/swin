import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model
import shutil


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataurl="./data/train/"
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    i = 0
    # create model
    model = create_model(num_classes=8).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # load image
    for classname in os.listdir(dataurl):
        for img_name in os.listdir(os.path.join(dataurl,classname)):
            i+=1
            img_path = os.path.join(dataurl,classname,img_name)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            # plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # read class_indict


            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
            #                                              predict[predict_cla].numpy())
            # plt.title(print_res)
            # for i in range(len(predict)):
            #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
            #                                               predict[i].numpy()))
            # plt.show()

            del img
            pre_class=class_indict[str(predict_cla)]
            if pre_class!=classname:
                pre_imgpath = os.path.join(dataurl, pre_class,img_name)
                shutil.move(img_path,pre_imgpath)
            print(i)



if __name__ == '__main__':
    main()
