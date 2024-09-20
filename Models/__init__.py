import torch
from .DETR import DETR
def Get_Network(model_name:str,args):
    if model_name == 'detr':
        detr = DETR(args)
    return detr