import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from .metrics import AverageMeter

log = logging.getLogger(__name__)

def train_one_epoch(
        student: nn.Module,
        loader: DataLoader,
        class_loss: nn.Module,
        dist_loss: nn.Module,
        optimizer: Optimizer,
        scheduler: CosineAnnealingLR,
        epoch: int,
        freq: int,
        query_desc_num: int = 400,
        db_desc_num: int = 400,
        beta: float = 0.,
        teacher: Optional[nn.Module] = None) -> None:

    student.train()
    device = next(student.parameters()).device
    to_device = lambda x: x.to(device, non_blocking=True)
    loader_length = len(loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    train_accs = AverageMeter(device=device, length=loader_length)
    pbar = tqdm(loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))

    for i, entry in enumerate(pbar):
        anchors, positives, negatives = entry
        anchors = list(map(to_device, anchors))
        positives = list(map(to_device, positives))
        negatives = list(map(to_device, negatives))

        if teacher is not None:
            q_len, db_len = np.random.randint(10, query_desc_num), np.random.randint(10, db_desc_num)
            trim_q = lambda x: (torch.split(x, [q_len, x.shape[1] - q_len], dim=1)[0] if x is not None else x)
            trim_db = lambda x: (torch.split(x, [db_len, x.shape[1] - db_len], dim=1)[0] if x is not None else x)

            with torch.no_grad():
                p_sims, p_logits = teacher(*(anchors + positives), return_logits=True)
                n_sims, n_logits = teacher(*(anchors + negatives), return_logits=True)
                L = anchors[1].shape[1]
                idx = np.arange(1 + q_len).tolist() + np.arange(1 + L, 1 + L + db_len).tolist()
                teacher_logits = torch.cat([p_logits, n_logits], 0)[:, idx]

            anchors = list(map(trim_q, anchors))
            positives = list(map(trim_db, positives))
            negatives = list(map(trim_db, negatives))

        p_sims, p_logits = student(*(anchors + positives), return_logits=True)
        n_sims, n_logits = student(*(anchors + negatives), return_logits=True)
        student_logits = torch.cat([p_logits, n_logits], 0)
        student_sims = torch.cat([p_sims, n_sims], 0)

        labels = student_sims.new_ones(student_sims.size()).float()
        labels[p_sims.shape[0]:] = 0
        loss = class_loss(student_sims, labels).mean()
        loss = loss + beta * dist_loss(student_logits, teacher_logits).mean() if beta > 0 else 0
        acc = ((torch.sigmoid(student_sims) > 0.5).long() == labels.long()).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(acc)
        if not (i + 1) % freq:
            step = epoch + i / loader_length
            log.info(f'step/loss/accu/lr: {step:.3f}, {train_losses.last_avg.item():.3f}, {train_accs.last_avg.item():.3f}, {scheduler.get_last_lr()[0]:.6f}')