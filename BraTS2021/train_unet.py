import os
import shutil

import torch
import torch.nn as nn
from dataset.brats_data_utils_multi_label import get_loader_brats
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.utils import set_determinism
from unet.basic_unet import BasicUNet

set_determinism(123)

data_dir = "/repos/datasets/brats2021/"
logdir = "./logs_unet"

model_save_path = os.path.join(logdir, "model")

env = "pytorch"  # or env = "pytorch" if you only have one gpu.

max_epoch = 3
batch_size = 1
val_every = 1
num_gpus = 1
device = "cuda:0"
num_workers = 1
save_every = 1
fast_dev_run = True

number_modality = 4
number_targets = 3  # WT, TC, ET


class BraTSTrainer(Trainer):
    def __init__(
        self,
        env_type,
        max_epochs,
        batch_size,
        device="cpu",
        val_every=1,
        num_gpus=1,
        logdir="./logs/",
        master_ip="localhost",
        master_port=17750,
        training_script="train.py",
        num_workers=8,
    ):
        super().__init__(
            env_type,
            max_epochs,
            batch_size,
            device,
            val_every,
            num_gpus,
            logdir,
            master_ip,
            master_port,
            training_script,
            num_workers,
        )
        if fast_dev_run:
            overlap = 0.1
        else:
            overlap = 0.25
        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96], sw_batch_size=1, overlap=overlap
        )
        self.model = BasicUNet(
            spatial_dims=3,
            in_channels=number_modality,
            out_channels=number_targets,
            features=[64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
            return_embeddings=False,
        )

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer, warmup_epochs=30, max_epochs=max_epochs
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

        self.save_every = save_every

    def training_step(self, batch):
        image, label = self.get_input(batch)
        output = self.model(image)

        loss_dice = self.dice_loss(output, label)
        loss_bce = self.bce(output, label)

        output_sigmoid = torch.sigmoid(output)
        loss_mse = self.mse(output_sigmoid, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train/loss", loss, step=self.global_step)
        self.log("train/loss_dice", loss_dice, step=self.global_step)
        self.log("train/loss_bce", loss_bce, step=self.global_step)
        self.log("train/loss_mse", loss_mse, step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.window_infer(image, self.model)

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()

        # peritumoral edema
        o = output[:, 1]
        t = target[:, 1]
        wt = dice(o, t)

        # necrotic tumor core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)

        # enhancing tumor
        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)

        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        self.log("val/wt_dice", wt, step=self.epoch)
        self.log("val/tc_dice", tc, step=self.epoch)
        self.log("val/et_dice", et, step=self.epoch)

        self.log("val/mean_dice", (wt + tc + et) / 3, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        # if mean_dice > self.best_mean_dice:
        #     self.best_mean_dice = mean_dice
        #     save_new_model_and_delete_last(self.model,
        #                                     os.path.join(model_save_path,
        #                                     f"best_model_{mean_dice:.4f}.pt"),
        #                                     delete_symbol="best_model")

        if self.epoch % self.save_every == 0 and self.epoch != 0:
            save_new_model_and_delete_last(
                self.model,
                os.path.join(
                    model_save_path, f"epoch_{self.epoch}_model_{mean_dice:.4f}.pt"
                ),
                delete_symbol="epoch",
            )
            print(f"save model at epoch {self.epoch}")

        if self.epoch == self.max_epochs - 1:
            save_new_model_and_delete_last(
                self.model,
                os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
                delete_symbol="final_model",
            )
            print(f"save final model at epoch {self.epoch}")

        print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")


if __name__ == "__main__":
    if fast_dev_run:
        print("fast_dev_run is True!")

    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        print(f"logdir has existed, remove {logdir}!")

    train_ds, val_ds, _ = get_loader_brats(
        data_dir=data_dir, batch_size=batch_size, fold=0, fast_dev_run=fast_dev_run
    )

    trainer = BraTSTrainer(
        env_type=env,
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        logdir=logdir,
        val_every=val_every,
        num_gpus=num_gpus,
        training_script=__file__,
        num_workers=num_workers,
    )

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
