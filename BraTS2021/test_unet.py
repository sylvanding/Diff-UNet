import os
from collections import OrderedDict

import numpy as np
import torch
from dataset.brats_data_utils_multi_label import get_loader_brats, save_brats_pred_seg
from light_training.evaluation.metric import dice, fscore, hausdorff_distance_95, recall
from light_training.trainer import Trainer
from light_training.utils.files import save_metrics_to_csv
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from unet.basic_unet import BasicUNet

set_determinism(123)

data_dir = "/repos/datasets/brats2021/"
logdir = "./logs_unet"

max_epoch = 10
batch_size = 1
val_every = 1
device = "cuda:0"
num_workers = 1
metric_file = "validation_metrics.csv"
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

        self.model = BasicUNet(
            spatial_dims=3,
            in_channels=number_modality,
            out_channels=number_targets,
            features=[64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
            return_embeddings=False,
        )

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
                "BasicUNet",
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

    logdir = "./logs_unet/model/final_model_0.0430.pt"
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")
