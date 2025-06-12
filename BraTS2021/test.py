import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from dataset.brats_data_utils_multi_label import (
    get_loader_brats,
    save_brats_pred_seg,
    save_brats_uncer,
)
from guided_diffusion.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from guided_diffusion.resample import UniformSampler
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from light_training.evaluation.metric import dice, fscore, hausdorff_distance_95, recall
from light_training.trainer import Trainer
from light_training.utils.files import save_metrics_to_csv
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe

set_determinism(123)

data_dir = "/repos/datasets/brats2021/"

max_epoch = 10
batch_size = 1
val_every = 1
device = "cuda:0"
num_workers = 1
metric_file = "validation_metrics.csv"
fast_dev_run = True

number_modality = 4
number_targets = 3  # WT, TC, ET


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = -pred_out * torch.log(pred_out)
    return uncer_out


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            3, number_modality, number_targets, [64, 64, 128, 256, 512, 64]
        )

        self.model = BasicUNetDe(
            3,
            number_modality + number_targets,
            number_targets,
            [64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
        )

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [1000]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )

        self.sample_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, [10]),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
        )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 4
            if fast_dev_run:
                uncer_step = 2
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(
                    self.sample_diffusion.ddim_sample_loop(
                        self.model,
                        (1, number_targets, 96, 96, 96),
                        model_kwargs={"image": image, "embeddings": embeddings},
                    )
                )

            sample_return = torch.zeros((1, number_targets, 96, 96, 96))
            all_uncers = []

            for index in range(10):
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()
                all_uncers.append(uncer)

                w = torch.exp(
                    torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer)
                )

                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()

            # concat uncers, change shape to fit SlidingWindowInferer
            stacked_uncers = torch.cat(all_uncers, dim=1)

            # return sample_return/(uncer_step * 10) # keep logits scale to -1~1
            return sample_return, stacked_uncers  # larger logits scale


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
        self.metric_file = os.path.join(logdir, metric_file)
        if os.path.exists(self.metric_file):
            os.remove(self.metric_file)
            print(f"metric file {self.metric_file} removed!")

        if fast_dev_run:
            overlap = 0.1
        else:
            overlap = 0.5
        self.window_infer = SlidingWindowInferer(
            roi_size=[96, 96, 96], sw_batch_size=1, overlap=overlap
        )

        self.model = DiffUNet()

    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]

        label = label.float()
        return image, label

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output, uncers = self.window_infer(image, self.model, pred_type="ddim_sample")

        # reshape uncers back to (10, 3, D, H, W)
        uncers = uncers.view(-1, 10, 3, *uncers.shape[2:])

        uncers = uncers.cpu().numpy()
        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()
        target = label.cpu().numpy()

        # list to store metrics for each sample in the batch
        batch_metrics = {
            "wt": [],
            "tc": [],
            "et": [],
            "wt_hd": [],
            "tc_hd": [],
            "et_hd": [],
            "wt_recall": [],
            "tc_recall": [],
            "et_recall": [],
            "wt_fscore": [],
            "tc_fscore": [],
            "et_fscore": [],
        }

        for i in range(output.shape[0]):
            # get data for single sample
            output_i = output[i]
            target_i = target[i]
            uncer_i = uncers[i]

            file_identifier = batch["file_identifizer"][i]
            data_path = batch["data_path"][i]

            original_shape = batch["original_shape"][i].cpu().numpy()
            foreground_start_coord = batch["foreground_start_coord"][i].cpu().numpy()
            foreground_end_coord = batch["foreground_end_coord"][i].cpu().numpy()

            # save segmentation prediction results
            save_brats_pred_seg(
                output_i,
                data_path,
                file_identifier,
                "DiffUNet",
                original_shape,
                foreground_start_coord,
                foreground_end_coord,
            )
            save_brats_uncer(
                uncer_i,
                data_path,
                file_identifier,
                "DiffUNet",
                original_shape,
                foreground_start_coord,
                foreground_end_coord,
            )

            # calculate metrics for single sample
            # peritumoral edema (WT)
            wt_i = dice(output_i[1], target_i[1])
            wt_hd_i = hausdorff_distance_95(output_i[1], target_i[1])
            wt_recall_i = recall(output_i[1], target_i[1])
            wt_fscore_i = fscore(output_i[1], target_i[1])

            # necrotic and non-enhancing tumor core (TC)
            tc_i = dice(output_i[0], target_i[0])
            tc_hd_i = hausdorff_distance_95(output_i[0], target_i[0])
            tc_recall_i = recall(output_i[0], target_i[0])
            tc_fscore_i = fscore(output_i[0], target_i[0])

            # enhancing tumor (ET)
            et_i = dice(output_i[2], target_i[2])
            et_hd_i = hausdorff_distance_95(output_i[2], target_i[2])
            et_recall_i = recall(output_i[2], target_i[2])
            et_fscore_i = fscore(output_i[2], target_i[2])

            # prepare metrics to save
            metrics_to_save = OrderedDict(
                [
                    ("filename", file_identifier),
                    ("wt", wt_i),
                    ("tc", tc_i),
                    ("et", et_i),
                    ("wt_hd", wt_hd_i),
                    ("tc_hd", tc_hd_i),
                    ("et_hd", et_hd_i),
                    ("wt_recall", wt_recall_i),
                    ("tc_recall", tc_recall_i),
                    ("et_recall", et_recall_i),
                    ("wt_fscore", wt_fscore_i),
                    ("tc_fscore", tc_fscore_i),
                    ("et_fscore", et_fscore_i),
                ]
            )

            # save metrics to CSV file
            save_metrics_to_csv(self.logdir, metric_file, metrics_to_save)

            # add current sample's metrics to batch list
            for key, value in metrics_to_save.items():
                if key != "filename":
                    batch_metrics[key].append(value)

        # calculate average metrics for the batch
        mean_metrics = [
            np.mean(batch_metrics[key])
            for key in [
                "wt",
                "tc",
                "et",
                "wt_hd",
                "tc_hd",
                "et_hd",
                "wt_recall",
                "tc_recall",
                "et_recall",
                "wt_fscore",
                "tc_fscore",
                "et_fscore",
            ]
        ]

        print(
            f"wt is {mean_metrics[0]}, tc is {mean_metrics[1]}, et is {mean_metrics[2]}"
        )
        return mean_metrics


if __name__ == "__main__":
    if fast_dev_run:
        print("fast_dev_run is True!")

    _, _, test_ds = get_loader_brats(
        data_dir=data_dir, batch_size=batch_size, fold=0, fast_dev_run=fast_dev_run
    )

    trainer = BraTSTrainer(
        env_type="pytorch",
        max_epochs=max_epoch,
        batch_size=batch_size,
        device=device,
        val_every=val_every,
        num_gpus=1,
        training_script=__file__,
        num_workers=num_workers,
    )

    logdir = "./logs/model/final_model_0.0473.pt"
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")
