from .detr_label import detr_encoder
def get_encoder(name:str):
    if name =='detr':
        return detr_encoder
    return None