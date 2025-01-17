"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午10:50
@Author  : Yang "Jan" Xiao 
@Description : train_utils
"""
import numpy as np
import torch
import torch_optimizer
from torch import optim
from torch import nn
from networks.bcresnet import BCResNet
from networks.tcresnet import TCResNet
from networks.matchboxnet import MatchboxNet
from networks.kwt import kwt_from_name
from networks.convmixer import KWSConvMixer
from torchaudio.transforms import MFCC
import random
from typing import Optional, Tuple

def mixup(data: torch.Tensor, target: Optional[torch.Tensor] = None, alpha: float = 0.2, beta: float = 0.2, mixup_label_type: str = "soft") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mixup data augmentation by permuting the data.

    Args:
        data: input tensor, must be a batch so data can be permuted and mixed.
        target: tensor of the target to be mixed, if None, do not return targets.
        alpha: float, the parameter to the np.random.beta distribution
        beta: float, the parameter to the np.random.beta distribution
        mixup_label_type: str, the type of mixup to be used choice between {'soft', 'hard'}.
    Returns:
        torch.Tensor of mixed data and labels if given
    """
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if target.dim() == 1:
                target = target.unsqueeze(1)
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target.squeeze(1) if mixed_target.size(1) == 1 else mixed_target
        else:
            return mixed_data
    


def _spec_augmentation(x, num_time_mask=1, num_freq_mask=1, max_time=25, max_freq=25):
    """perform spec augmentation 
    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
    Returns:
        augmented feature
    """
    max_freq_channel, max_frames = x.size()[-2], x.size()[-1]

    # time mask
    for i in range(num_time_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_time)
        end = min(max_frames, start + length)
        x[:, start:end] = 0

    # freq mask
    for i in range(num_freq_mask):
        start = random.randint(0, max_freq_channel - 1)
        length = random.randint(1, max_freq)
        end = min(max_freq_channel, start + length)
        x[start:end, :] = 0

    return x
    
class MFCC_KWS_Model(nn.Module):
    def __init__(self, model) -> None:
        super(MFCC_KWS_Model,self).__init__()
        self.mfcc = MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64, "f_min": 20, "f_max": 8000},
        )
        self.model = model
    def forward(self, x, label=None):
        x = self.mfcc(x)
        batch_size, num_channels, num_mfcc, num_frames = x.shape
        x = x.view(batch_size, num_mfcc, num_frames)  # 确保 x 是二维的
        if self.training:
            if 0.5 < np.random.rand():
                x, label = mixup(x, label, alpha=0.2, beta=0.2, mixup_label_type="soft")
            x = _spec_augmentation(x, num_time_mask=1, num_freq_mask=1, max_time=25, max_freq=25)
        x = self.model(x)
        if label is not None:
            label = label.long()  # 确保标签是长整型
        return x if label is None else (x, label)


def select_optimizer(opt_name, lr, model, sched_name="cos"):
    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    elif opt_name == "radam":
        opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=0.00001)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    elif opt_name == "NovoGrad":
        opt = torch_optimizer.NovoGrad(model.parameters(), lr=0.05, betas=(0.95, 0.5), weight_decay=0.001)
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")

    if sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2, eta_min=lr * 0.01
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, list(range(5, 26)), gamma=0.85
        )
    else:
        raise NotImplementedError(
            "Please select the sched_name [cos, anneal, multistep]"
        )

    return opt, scheduler


def select_model(model_name, total_class_num=None):
    # load the model.
    config = {
        "tcresnet8": [16, 24, 32, 48],
        "tcresnet14": [16, 24, 24, 32, 32, 48, 48]
    }

    if model_name == "tcresnet8" or model_name == "tcresnet14":
        model = MFCC_KWS_Model(TCResNet(bins=40, n_channels=[int(cha * 1) for cha in config[model_name]],
                              n_class=total_class_num))
    elif "bcresnet" in model_name:
        scale = int(model_name[-1])
        model = MFCC_KWS_Model(BCResNet(n_class=total_class_num, scale=scale))
    elif "matchboxnet" in model_name:
        b, r, c = model_name.split("_")[1:]
        model = MFCC_KWS_Model(MatchboxNet(B=int(b), R=int(r), C=int(c), bins=40, kernel_sizes=None,num_classes=total_class_num))
    elif "kwt" in model_name:
        model = MFCC_KWS_Model(kwt_from_name(model_name, total_class_num))
    elif "convmixer" in model_name:
        model = MFCC_KWS_Model(KWSConvMixer(input_size=[101, 40],num_classes=total_class_num))
    else:
        model = None
    print(model)
    return model


if __name__ == "__main__":
    inputs = torch.randn(8, 1, 16000)
    # inputs = padding(inputs, 128)
    model = select_model("bcresnet2", 15)
    outputs = model(inputs)
    print(outputs.shape)
    print('num parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

