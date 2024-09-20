import argparse
from .models.detr import build
import torch
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks',default=False, action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser
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
class DETR(torch.nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        self.config = config
        self.args = parser.parse_args()
        self.model, self.criterion, self.postprocessors = build(self.args)
        self.model = self.model.cuda()
        if self.args.resume:
            if self.args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.args.resume, map_location='cuda', check_hash=True)
            else:
                checkpoint = torch.load(self.args.resume, map_location='cpu')
            load_ckpt(self.model,checkpoint['model'])
            if not self.args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.args.start_epoch = checkpoint['epoch'] + 1
        param_dicts = [
        {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": self.args.lr_backbone,
        },]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)
    # def get_optimizer(self):
    #     param_dicts = [
    #     {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": self.args.lr_backbone,
    #     },]
    #     self.optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr,weight_decay=self.args.weight_decay)
    #     self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_drop)
    def train(self,inputs,targets):
        outputs = self.model(inputs)
        loss_dict = self.criterion(outputs,targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict['total_loss'] = losses
        print(loss_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_dict
    def forward(self,inputs):
        batch = len(inputs)
        img_sizes = torch.tensor([self.config['img_size']]*batch)
        if self.config['use_cuda']:
            img_sizes = img_sizes.cuda()
        outputs = self.model(inputs)
        outputs = self.postprocessors['bbox'](outputs,img_sizes)
        return outputs[..., 0:4].cpu().numpy(),(outputs[..., 4:5]).cpu().numpy()