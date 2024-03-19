# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import gc
import argparse
from pathlib import Path
import re

from utils import get_device, image_to_grid, save_image
from unet import UNet
from classifier import Classifier
from classifier_guidance import ClassifierGuidedDiffusion


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--classifier_scale", type=float, required=True)

    parser.add_argument("--img_size", type=int, required=True)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


def get_sample_num(x, pref):
    match = re.search(pattern=rf"{pref}-\s*(.+)", string=x)
    return int(match.group(1)) if match else -1


def get_max_sample_num(samples_dir, pref):
    stems = [path.stem for path in Path(samples_dir).glob("**/*") if path.is_file()]
    if stems:
        return max([get_sample_num(stem, pref=pref) for stem in stems])
    else:
        return -1


def pref_to_save_path(samples_dir, pref, suffix):
    max_sample_num = get_max_sample_num(samples_dir, pref=pref)
    save_stem = f"{pref}-{max_sample_num + 1}"
    return str((Path(samples_dir)/save_stem).with_suffix(suffix))


def get_save_path(samples_dir, classifier_scale):
    pref = f"classifier_scale={classifier_scale}"
    return pref_to_save_path(samples_dir=samples_dir, pref=pref, suffix=".jpg")


def main():
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()
    print(f"[ DEVICE: {DEVICE} ]")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

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
    model = ClassifierGuidedDiffusion(
        unet=unet,
        classifier=classifier,
        img_size=args.IMG_SIZE,
        classifier_scale=args.CLASSIFIER_SCALE,
        device=DEVICE,
    )
    state_dict = torch.load(str(args.MODEL_PARAMS), map_location=DEVICE)
    model.load_state_dict(state_dict)

    SAMPLES_DIR = Path(__file__).resolve().parent/"samples"
    test_label = torch.arange(
        N_CLASSES, dtype=torch.int32, device=DEVICE,
    ).repeat_interleave(args.BATCH_SIZE)
    gen_image = model.sample(batch_size=test_label.size(0), label=test_label)
    gen_grid = image_to_grid(gen_image, n_cols=args.BATCH_SIZE)
    gen_grid.show()
    save_path = get_save_path(
        samples_dir=SAMPLES_DIR, classifier_scale=args.CLASSIFIER_SCALE,
    )
    save_image(gen_grid, save_path=save_path)


if __name__ == "__main__":
    main()
