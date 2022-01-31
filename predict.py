import os

import numpy as np
import torch
import cv2

from model import Rec_Transformer
from utils.data_utils import get_loader

def load_model(load_model_dir, model):
    loaded_dict = torch.load(load_model_dir)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)


def setup(load_model_dir,input_size, output_size):
    model = Rec_Transformer(input_size=input_size, rec_size=output_size)
    load_model(load_model_dir, model)
    return model

def predict(color, input_size, output_size, model,test_dir,rec_dir):
    test_in = np.load(test_dir)
    if color:
        b, g, r = cv2.split(test_in)
        test_in = np.dstack((r, g, b))
        test_out = np.zeros((output_size, output_size, 3))
        for c in range(3):
            test_in_one_channel = test_in[:, :, c]
            test_in_one_channel = np.reshape(test_in_one_channel,(1, 1, input_size, input_size))
            test_in_one_channel = (test_in_one_channel - test_in_one_channel.mean()) / test_in_one_channel.std()
            test_in_one_channel.astype(float)
            test_in_one_channel = torch.from_numpy(test_in_one_channel)
            test_in_one_channel.cuda()
            test_out_one_channel = model(test_in_one_channel)
            test_out_one_channel = test_out_one_channel.to('cpu').detach().numpy().copy()
            test_out_one_channel = np.reshape(test_out_one_channel,(output_size, output_size))
            test_out[:, :, c] = test_out_one_channel
    else:
        test_in=np.reshape(test_in,(1, 1, input_size, input_size))
        test_in=(test_in - test_in.mean()) / test_in.std()
        test_in.astype(float)
        test_in = torch.from_numpy(test_in)
        test_in.cuda()
        test_out = model(test_in)
        test_out = test_out.to('cpu').detach().numpy().copy()
    cv2.imwrite(rec_dir,cv2.normalize(test_out, None, 0, 255, cv2.NORM_MINMAX))



def main():
    input_size=1600
    output_size=500
    test_dir='/home/pan/Desktop/vit_rec/Rec_Transformer0824/datasets/final_val/pattern/'
    files=os.listdir(test_dir)
    rec_dir='/home/pan/Desktop/vit_rec/Rec_Transformer0824/predict/'

    load_model_dir = '/home/pan/Desktop/vit_rec/Rec_Transformer0824/checkpoints_best/best.pth'
    model = setup(load_model_dir,input_size,output_size)
    model.cuda()
    for file in files:
        predict(True, input_size, output_size, model, test_dir+file, rec_dir+'best/'+file[:-3]+'bmp')

if __name__ == "__main__":
    main()
