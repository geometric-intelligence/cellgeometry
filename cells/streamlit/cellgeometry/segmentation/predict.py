import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Modified3DUNet
from celldataset import cell_testing_inter
from utils import Parser, criterions
from skimage.transform import resize


def readprops(proppath):
    props = dict()
    with open(proppath) as file:
        for line in file:
            entry = line.strip().split(":", 1)
            props[entry[0].strip()] = entry[1].strip()
    print(props)
    return props


def saveimage(image, filename):
    data = sitk.GetImageFromArray(image)
    sitk.WriteImage(data, filename)


def main(model_path, cell_hist_datadir, prob_map_datadir):
    props = readprops(os.path.join(model_path, "cfg.txt"))

    model = Modified3DUNet(in_channels=1, n_classes=2, base_n_filter=16)

    # Load model
    model_file = os.path.join(model_path, "model_last.tar")
    checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    criterion = getattr(criterions, props["criterion"])
    batch_size = int(props["batch_size"])
    workers = int(props["workers"])

    dset = cell_testing_inter(cell_hist_datadir)
    print(dset.__len__())
    test_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.eval()
    torch.set_grad_enabled(False)

    for i, sample in enumerate(test_loader):
        input1 = sample["data"]
        img_size = sample["image_size"]
        file_name = os.path.splitext(sample["name"][0])[0]

        _, _, z, x, y = input1.shape
        seg = np.zeros((z, x, y))
        for j in range(z // 16):
            input_temp = input1[0, 0, j * 16 : (j + 1) * 16].float()
            input_temp = input_temp[None, None, ...]
            output_temp = model(input_temp)
            output_temp = output_temp.detach().numpy()
            seg_temp = output_temp.argmax(0)
            seg[j * 16 : (j + 1) * 16] = seg_temp

        data = input1.detach().numpy()[0, 0, :, :, :]
        prob_map = (seg[0 : 5 * img_size[0]] * 255).astype("uint8")
        prob_map = prob_map.astype("float32") / 255.0
        prob_map = np.multiply(data[0 : 5 * img_size[0]], prob_map)
        prob_map = resize(
            prob_map, (prob_map.shape[0] // 5, prob_map.shape[1], prob_map.shape[2])
        )
        prob_map_img = sitk.GetImageFromArray(prob_map.astype("uint8"))
        sitk.WriteImage(
            prob_map_img, os.path.join(prob_map_datadir, f"{file_name}-prob.tif")
        )


if __name__ == "__main__":
    model_path = "model/regression"
    cell_hist_datadir = "hist_match/"
    prob_map_datadir = "prob_map/"
    main(model_path, cell_hist_datadir, prob_map_datadir)
