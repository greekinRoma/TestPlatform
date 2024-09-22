from torch import nn
from .RetinaNet.Model.retainnet import RetainNet
import torch
from easydict import EasyDict as edict
from ..utils import LRScheduler
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - w / 2), (cy - h / 2),
         (cx + w / 2), (cy + h / 2)]
    return torch.stack(b, dim=-1)

class GTBoxes(nn.Module):
    def __init__(self,tensor) -> None:
        super().__init__()
        self.tensor = box_cxcywh_to_xyxy(tensor)
class torch_dict(nn.Module):
    def __init__(self,target):
        super().__init__()
        mas = target[...,0]>0
        self.gt_boxes = GTBoxes(target[mas,1:])
        self.gt_classes = target[mas,0].to(torch.long)
def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            print(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            print(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model
class RetainNet_network(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = RetainNet()
        checkpoint = torch.load(r'/home/greek/files/test/Test_platfrom/Weights/YOLOF/YOLOF_CSP_D_53_DC5_9x.pth', map_location='cuda')
        load_ckpt(self.model,checkpoint['model'])
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.basic_lr_per_img = 0.01 / 64.0
        self.batch_size = args['batch_size']
        self.max_epoch = args['max_epoch']
        self.warmup_epochs = 1
        self.warmup_lr = 0
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.scheduler = "yoloxwarmcos"
        self.lr_scheduler = self.get_lr_scheduler(self.basic_lr_per_img*self.batch_size,150)
        self.optimizer = self.get_optimizer(args['batch_size'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
    def train(self,inputs,targets,iter):
        dict_targets = []
        for target in targets:
            dict_target = torch_dict(target)
            dict_targets.append(dict_target)
        losses = self.model(inputs,dict_targets,training=True)
        total_loss = 0
        for k in losses.keys():
            total_loss += losses[k]
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        lr = self.lr_scheduler.update_lr(iter)
        for k,param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        return losses
    def get_optimizer(self, batch_size):
        if self.warmup_epochs > 0:
            lr = self.warmup_lr
        else:
            lr = self.basic_lr_per_img * batch_size
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        optimizer = torch.optim.SGD(
            pg0, lr=lr, momentum=self.momentum, nesterov=True
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.weight_decay}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params":pg2})
        return optimizer
    def get_lr_scheduler(self,lr,iters_per_epoch):
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio)
        return scheduler
    def forward(self,inputs):
        outputs = self.model(inputs,None,False)
        boxes = outputs[0]['instances'].pred_boxes.tensor.unsqueeze(0).detach().cpu().numpy()
        score = outputs[0]['instances'].scores.unsqueeze(0).unsqueeze(-1).detach().cpu().numpy()
        return boxes,score

        

