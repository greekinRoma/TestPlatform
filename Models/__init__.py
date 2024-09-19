import torch
from .DETR import DETR
def Get_Network(model_name:str):
    if model_name == 'detr':
        detr = DETR()
    return detr