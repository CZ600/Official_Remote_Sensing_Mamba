import logging
import os
import random
import sys
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from rs_mamba_ss import RSM_SS
from utils.data_loading import BasicDataset
from utils.losses import FCCDN_loss_without_seg
from utils.path_hyperparameter import ph
from utils.utils import train_val_test


def random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def build_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    num_workers = min(ph.num_workers, os.cpu_count() or 1)
    loader_args = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_args["persistent_workers"] = True
        loader_args["prefetch_factor"] = ph.prefetch_factor
    return DataLoader(dataset, **loader_args)


def auto_experiment() -> None:
    random_seed(seed=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info("Interrupt")
        sys.exit(0)


def train_net(dataset_name: str) -> None:
    train_dataset = BasicDataset(
        images_dir=f"{ph.root_dir}/{dataset_name}/train/{ph.image_dir_name}/",
        labels_dir=f"{ph.root_dir}/{dataset_name}/train/{ph.label_dir_name}/",
        train=True,
    )
    val_dataset = BasicDataset(
        images_dir=f"{ph.root_dir}/{dataset_name}/val/{ph.image_dir_name}/",
        labels_dir=f"{ph.root_dir}/{dataset_name}/val/{ph.label_dir_name}/",
        train=False,
    )

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = build_dataloader(train_dataset, batch_size=ph.batch_size, shuffle=True)
    val_loader = build_dataloader(
        val_dataset,
        # batch_size=ph.batch_size * ph.inference_ratio,
        batch_size=1,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    log_dir = os.path.join(ph.log_dir, f"{ph.project_name}_{localtime}")
    writer = SummaryWriter(log_dir=log_dir)

    logging.info(
        "Starting training:\n"
        f"    Epochs:          {ph.epochs}\n"
        f"    Batch size:      {ph.batch_size}\n"
        f"    Learning rate:   {ph.learning_rate}\n"
        f"    Training size:   {n_train}\n"
        f"    Validation size: {n_val}\n"
        f"    Checkpoints:     {ph.save_checkpoint}\n"
        f"    Save best model: {ph.save_best_model}\n"
        f"    Device:          {device.type}\n"
        f"    TensorBoard:     {os.path.abspath(log_dir)}"
    )

    net = RSM_SS(
        dims=ph.dims,
        depths=ph.depths,
        ssm_d_state=ph.ssm_d_state,
        ssm_dt_rank=ph.ssm_dt_rank,
        ssm_ratio=ph.ssm_ratio,
        mlp_ratio=ph.mlp_ratio,
    ).to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate, weight_decay=ph.weight_decay)
    warmup_lr = np.arange(
        1e-7,
        ph.learning_rate,
        (ph.learning_rate - 1e-7) / ph.warm_up_step,
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=ph.amp and device.type == "cuda")

    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        state_dict = checkpoint["net"] if "net" in checkpoint else checkpoint
        net.load_state_dict(state_dict)
        logging.info(f"Model loaded from {ph.load}")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = ph.learning_rate

    total_step = 0
    lr = ph.learning_rate
    criterion = FCCDN_loss_without_seg
    best_metrics = {
        "best_road_iou": float("-inf"),
        "lowest_loss": float("inf"),
        "best_epoch": -1,
    }
    checkpoint_dir = os.path.join(ph.checkpoint_dir, ph.project_name)

    for epoch in range(ph.epochs):
        logging.info(f"Epoch {epoch + 1}/{ph.epochs} - train")
        net, optimizer, grad_scaler, total_step, lr, train_metrics = train_val_test(
            mode="train",
            dataloader=train_loader,
            device=device,
            writer=writer,
            net=net,
            optimizer=optimizer,
            total_step=total_step,
            lr=lr,
            criterion=criterion,
            epoch=epoch,
            warmup_lr=warmup_lr,
            grad_scaler=grad_scaler,
        )

        if (epoch + 1) >= ph.evaluate_epoch and (epoch + 1) % ph.evaluate_inteval == 0:
            logging.info(f"Epoch {epoch + 1}/{ph.epochs} - val")
            with torch.no_grad():
                net, optimizer, total_step, lr, best_metrics, _, val_metrics = train_val_test(
                    mode="val",
                    dataloader=val_loader,
                    device=device,
                    writer=writer,
                    net=net,
                    optimizer=optimizer,
                    total_step=total_step,
                    lr=lr,
                    criterion=criterion,
                    epoch=epoch,
                    best_metrics=best_metrics,
                    checkpoint_path=checkpoint_dir,
                )
            logging.info(
                "Validation summary - "
                f"loss: {val_metrics['loss']:.6f}, "
                f"miou: {val_metrics['miou']:.6f}, "
                f"road_iou: {val_metrics['road_iou']:.6f}, "
                f"precision: {val_metrics['precision']:.6f}, "
                f"recall: {val_metrics['recall']:.6f}, "
                f"f1: {val_metrics['f1']:.6f}"
            )
        else:
            logging.info(
                "Training summary - "
                f"loss: {train_metrics['loss']:.6f}, "
                f"miou: {train_metrics['miou']:.6f}, "
                f"road_iou: {train_metrics['road_iou']:.6f}"
            )

    writer.close()


if __name__ == "__main__":
    auto_experiment()
