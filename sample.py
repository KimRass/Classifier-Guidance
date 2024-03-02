# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import argparse

from utils import get_device, image_to_grid, save_image
from unet import UNet
from classifier import Classifier
from classifier_guidance import ClassifierGuidedDiffusion


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--classifier_scale", type=float, required=True)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def main():
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()
    print(f"[ DEVICE: {DEVICE} ]")

    N_CLASSES = 10
    unet = UNet(
        n_classes=N_CLASSES,
        channels=128,
        channel_mults=[1, 2, 2, 2],
        attns=[False, True, False, False],
        n_res_blocks=2,
    )
    classifier = Classifier(
        n_classes=N_CLASSES,
        channels=64,
        channel_mults=[1, 2, 2, 2],
        attns=[False, True, False, False],
        n_res_blocks=2,
    ).to(DEVICE)
    IMG_SIZE = 32
    model = ClassifierGuidedDiffusion(
        unet=unet,
        classifier=classifier,
        img_size=IMG_SIZE,
        classifier_scale=args.CLASSIFIER_SCALE,
        device=DEVICE,
    )
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    test_label = torch.arange(
        N_CLASSES, dtype=torch.int32, device=DEVICE,
    ).repeat_interleave(args.BATCH_SIZE)
    gen_image = model.sample(batch_size=test_label.size(0), label=test_label)
    gen_grid = image_to_grid(gen_image, n_cols=args.BATCH_SIZE)
    gen_grid.show()
    save_image(gen_grid, save_path=args.SAVE_PATH)


if __name__ == "__main__":
    main()
