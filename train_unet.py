import torch
from torch.optim import AdamW
import gc
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from copy import deepcopy
import wandb

from utils import (
    set_seed,
    get_device,
    get_grad_scaler,
    get_elapsed_time,
    modify_state_dict,
    print_n_params,
    image_to_grid,
    save_image,
)
from data import get_train_and_val_dls
from unet import UNet
from classifier import Classifier
from classifier_guidance import ClassifierGuidedDiffusion


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--classifier_params", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--n_warmup_steps", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=32, required=False)

    parser.add_argument("--seed", type=int, default=223, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


class Trainer(object):
    def __init__(self, n_classes, train_dl, val_dl, save_dir, device):
        self.n_classes = n_classes
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.save_dir = Path(save_dir)
        self.device = device

        # self.run = wandb.init(project="Classifier-Guidance-Classifier")

        self.ckpt_path = self.save_dir/"ckpt.pth"

        self.test_label = torch.arange(
            self.n_classes, dtype=torch.int32, device=self.device,
        ).repeat_interleave(10)

    def train_for_one_epoch(self, epoch, model, optim, scaler):
        train_loss = 0
        pbar = tqdm(self.train_dl, leave=False)
        for step_idx, (ori_image, label) in enumerate(pbar):
            pbar.set_description("Training...")

            ori_image = ori_image.to(self.device)
            label = label.to(self.device)

            loss = model.get_unet_loss(ori_image=ori_image, label=label)
            train_loss += (loss.item() / len(self.train_dl))

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            self.scheduler.step((epoch - 1) * len(self.train_dl) + step_idx)
        return train_loss

    @torch.inference_mode()
    def validate(self, model):
        val_loss = 0
        pbar = tqdm(self.val_dl, leave=False)
        for ori_image, label in pbar:
            pbar.set_description("Validating...")

            ori_image = ori_image.to(self.device)
            label = label.to(self.device)

            loss = model.get_unet_loss(ori_image=ori_image, label=label)
            val_loss += (loss.item() / len(self.val_dl))
        return val_loss

    @staticmethod
    def save_model_params(model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(modify_state_dict(model.state_dict()), str(save_path))
        print(f"Saved model params as '{str(save_path)}'.")

    def save_ckpt(self, epoch, model, optim, min_val_loss, scaler):
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model": modify_state_dict(model.state_dict()),
            "optimizer": optim.state_dict(),
            "min_val_loss": min_val_loss,
        }
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, str(self.ckpt_path))

    # @torch.inference_mode()
    def test_sampling(self, epoch, model):
        gen_image = model.sample(batch_size=self.test_label.size(0), label=self.test_label)
        gen_grid = image_to_grid(gen_image, n_cols=int(self.test_label.size(0) ** 0.5))
        sample_path = self.save_dir/f"epoch={epoch}-sample.jpg"
        save_image(gen_grid, save_path=sample_path)
        wandb.log({"Samples": wandb.Image(sample_path)}, step=epoch)

    def train(self, n_epochs, model, optim, scaler, n_warmup_steps):
        model = torch.compile(model)

        self.scheduler = CosineLRScheduler(
            optimizer=optim,
            t_initial=n_epochs * len(self.train_dl),
            warmup_t=n_warmup_steps,
            warmup_lr_init=optim.param_groups[0]["lr"] * 0.1,
            warmup_prefix=True,
            t_in_epochs=False,
        )

        init_epoch = 0
        min_val_loss = math.inf
        for epoch in range(init_epoch + 1, n_epochs + 1):
            start_time = time()
            train_loss = self.train_for_one_epoch(
                epoch=epoch, model=model, optim=optim, scaler=scaler,
            )
            val_loss = self.validate(model)
            if val_loss < min_val_loss:
                model_params_path = str(
                    self.save_dir/f"epoch={epoch}-val_loss={val_loss:.4f}.pth"
                )
                self.save_model_params(model=model, save_path=model_params_path)
                min_val_loss = val_loss

            self.save_ckpt(
                epoch=epoch,
                model=model,
                optim=optim,
                min_val_loss=min_val_loss,
                scaler=scaler,
            )

            self.test_sampling(epoch=epoch, model=model)

            log = f"[ {get_elapsed_time(start_time)} ]"
            log += f"[ {epoch}/{n_epochs} ]"
            log += f"[ Train loss: {train_loss:.4f} ]"
            log += f"[ Val loss: {val_loss:.4f} | Best: {min_val_loss:.4f} ]"
            wandb.log(
                {"Train loss": train_loss, "Val loss": val_loss, "Min val loss": min_val_loss},
                step=epoch,
            )
            print(log)


def main():
    torch.set_printoptions(linewidth=70)

    DEVICE = get_device()
    args = get_args()
    set_seed(args.SEED)
    print(f"[ DEVICE: {DEVICE} ]")

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_dl, val_dl = get_train_and_val_dls(
        data_dir=args.DATA_DIR,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
        seed=args.SEED,
    )
    N_CLASSES = 10
    trainer = Trainer(
        n_classes=N_CLASSES,
        train_dl=train_dl,
        val_dl=val_dl,
        save_dir=args.SAVE_DIR,
        device=DEVICE,
    )

    unet = UNet(
        n_classes=N_CLASSES,
        channels=64,
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
    state_dict = torch.load(str(args.CLASSIFIER_PARAMS), map_location=DEVICE)
    classifier.load_state_dict(state_dict)
    model = ClassifierGuidedDiffusion(
        unet=unet,
        classifier=classifier,
        img_size=args.IMG_SIZE,
        device=DEVICE,
    )
    print_n_params(model)
    optim = AdamW(model.parameters(), lr=args.LR)
    scaler = get_grad_scaler(device=DEVICE)

    trainer.train(
        n_epochs=args.N_EPOCHS,
        model=model,
        optim=optim,
        scaler=scaler,
        n_warmup_steps=args.N_WARMUP_STEPS,
    )


if __name__ == "__main__":
    main()
