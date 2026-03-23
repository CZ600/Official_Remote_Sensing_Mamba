import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.path_hyperparameter import ph


class BinarySegmentationMeter:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.tn = 0.0
        self.fn = 0.0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).long()
        targets = (labels >= 0.5).long()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        self.tp += torch.sum((preds == 1) & (targets == 1)).item()
        self.fp += torch.sum((preds == 1) & (targets == 0)).item()
        self.tn += torch.sum((preds == 0) & (targets == 0)).item()
        self.fn += torch.sum((preds == 0) & (targets == 1)).item()

    def compute(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        road_iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        bg_iou = self.tn / (self.tn + self.fp + self.fn + self.eps)
        miou = 0.5 * (road_iou + bg_iou)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + self.eps)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "road_iou": road_iou,
            "miou": miou,
        }


def save_model(model, path, epoch, mode, optimizer=None, metrics=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    state_dict = {
        "net": model.state_dict(),
        "epoch": epoch + 1,
        "mode": mode,
    }
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    if metrics is not None:
        state_dict["metrics"] = metrics

    filename = f"{mode}_epoch_{epoch + 1}.pth"
    torch.save(state_dict, os.path.join(path, filename))
    logging.info(f"Saved {mode} model at epoch {epoch + 1}")


def train_val_test(
    mode,
    dataloader,
    device,
    writer,
    net,
    optimizer,
    total_step,
    lr,
    criterion,
    epoch,
    warmup_lr=None,
    grad_scaler=None,
    best_metrics=None,
    checkpoint_path=None,
):
    assert mode in ["train", "val"], "mode should be train or val"

    epoch_loss = 0.0
    epoch_dice_loss = 0.0
    epoch_bce_loss = 0.0
    meter = BinarySegmentationMeter(threshold=ph.threshold)

    if mode == "train":
        net.train()
    else:
        net.eval()
    logging.info(f"SET model mode to {mode}")

    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=max(n_iter, 1))
    sample_image = None
    sample_label = None
    sample_pred = None

    for i, (image, labels, _) in enumerate(tbar):
        tbar.set_description(f"epoch {epoch + 1} {mode} {i + 1}/{n_iter}")
        total_step += 1

        if mode == "train":
            optimizer.zero_grad()
            if total_step < ph.warm_up_step:
                for group in optimizer.param_groups:
                    group["lr"] = float(warmup_lr[total_step])

        image = image.float().to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        b, c, h, w = image.shape
        image = F.interpolate(
            image,
            size=(h // ph.downsample_raito, w // ph.downsample_raito),
            mode="bilinear",
            align_corners=False,
        )
        labels = F.interpolate(
            labels.unsqueeze(1),
            size=(h // ph.downsample_raito, w // ph.downsample_raito),
            mode="nearest",
        ).squeeze(1)

        crop_size = ph.image_size
        image_patches = image.unfold(2, crop_size, crop_size).unfold(3, crop_size, crop_size)
        _, _, _, _, _, _ = image_patches.size()
        image = image_patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, crop_size, crop_size).contiguous()

        labels_patches = labels.unfold(1, crop_size, crop_size).unfold(2, crop_size, crop_size)
        labels = labels_patches.reshape(-1, crop_size, crop_size).contiguous()

        with torch.cuda.amp.autocast(enabled=ph.amp and device.type == "cuda"):
            preds = net(image)
            loss_change, diceloss, foclaloss = criterion(preds, labels)
            cd_loss = loss_change.mean()

        if mode == "train":
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), ph.max_norm, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()

        epoch_loss += cd_loss.item()
        epoch_dice_loss += float(diceloss.item())
        epoch_bce_loss += float(foclaloss.item())
        meter.update(preds.detach(), labels.detach())

        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=image.shape[0])
            sample_image = image[sample_index].detach().cpu()
            sample_label = labels[sample_index].detach().cpu().unsqueeze(0)
            sample_pred = (torch.sigmoid(preds[sample_index]).detach().cpu() >= ph.threshold).float()

        batch_metrics = meter.compute()
        writer.add_scalar(f"{mode}/step_loss", cd_loss.item(), total_step)
        writer.add_scalar(f"{mode}/step_precision", batch_metrics["precision"], total_step)
        writer.add_scalar(f"{mode}/step_recall", batch_metrics["recall"], total_step)
        writer.add_scalar(f"{mode}/step_f1", batch_metrics["f1"], total_step)
        writer.add_scalar(f"{mode}/step_miou", batch_metrics["miou"], total_step)
        writer.add_scalar(f"{mode}/step_road_iou", batch_metrics["road_iou"], total_step)
        if mode == "train":
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], total_step)

    metrics = meter.compute()
    metrics["loss"] = epoch_loss / max(n_iter, 1)
    metrics["dice_loss"] = epoch_dice_loss / max(n_iter, 1)
    metrics["bce_loss"] = epoch_bce_loss / max(n_iter, 1)

    logging.info(
        f"{mode} epoch {epoch + 1}: "
        f"loss={metrics['loss']:.6f}, "
        f"miou={metrics['miou']:.6f}, "
        f"road_iou={metrics['road_iou']:.6f}, "
        f"precision={metrics['precision']:.6f}, "
        f"recall={metrics['recall']:.6f}, "
        f"f1={metrics['f1']:.6f}"
    )

    for key, value in metrics.items():
        writer.add_scalar(f"{mode}/epoch_{key}", value, epoch + 1)

    if sample_image is not None:
        writer.add_image(f"{mode}/image", sample_image, epoch + 1)
        writer.add_image(f"{mode}/label", sample_label, epoch + 1)
        writer.add_image(f"{mode}/prediction", sample_pred, epoch + 1)

    if mode == "val":
        if metrics["road_iou"] > best_metrics["best_road_iou"]:
            best_metrics["best_road_iou"] = metrics["road_iou"]
            best_metrics["best_epoch"] = epoch + 1
            if ph.save_best_model:
                save_model(net, checkpoint_path, epoch, "best_road_iou", optimizer=optimizer, metrics=metrics)

        if metrics["loss"] < best_metrics["lowest_loss"]:
            best_metrics["lowest_loss"] = metrics["loss"]

        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, "checkpoint", optimizer=optimizer, metrics=metrics)

        writer.add_scalar("val/best_road_iou", best_metrics["best_road_iou"], epoch + 1)

    if mode == "train":
        return net, optimizer, grad_scaler, total_step, lr, metrics
    return net, optimizer, total_step, lr, best_metrics, None, metrics
