import argparse
import os

import numpy as np
import torch
import cv2

from lensless_imaging_transformer.model import RecTransformer


def load_model(load_model_dir, model):
    loaded_dict = torch.load(load_model_dir)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)


def setup(load_model_dir, input_size, output_size):
    model = RecTransformer(input_size=input_size, rec_size=output_size)
    load_model(load_model_dir, model)
    return model


def predict(input_size, output_size, model, test_dir, rec_dir):
    test_in = np.load(test_dir)
    b, g, r = cv2.split(test_in)
    test_in = np.dstack((r, g, b))
    test_out = np.zeros((output_size, output_size, 3))
    for c in range(3):
        channel_in = test_in[:, :, c]
        channel_in = np.reshape(channel_in, (1, 1, input_size, input_size))
        channel_in = (channel_in - channel_in.mean()) / channel_in.std()
        channel_in = torch.from_numpy(channel_in).float().cuda()
        channel_out = model(channel_in)
        channel_out = channel_out.detach().cpu().numpy()
        test_out[:, :, c] = np.reshape(channel_out, (output_size, output_size))
    cv2.imwrite(rec_dir, cv2.normalize(test_out, None, 0, 255, cv2.NORM_MINMAX))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--input-dir', required=True, help='Directory of input pattern .npy files')
    parser.add_argument('--output-dir', required=True, help='Directory to write reconstructed .bmp files')
    parser.add_argument('--input-size', type=int, default=1600)
    parser.add_argument('--output-size', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = setup(args.checkpoint, args.input_size, args.output_size)
    model.cuda()
    model.eval()

    files = os.listdir(args.input_dir)
    for file in files:
        in_path = os.path.join(args.input_dir, file)
        out_path = os.path.join(args.output_dir, file[:-3] + 'bmp')
        predict(args.input_size, args.output_size, model, in_path, out_path)


if __name__ == "__main__":
    main()
