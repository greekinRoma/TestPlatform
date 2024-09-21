import torch
def detr_encoder(targets,args):
    new_targets = []
    for target in targets:
        new_target = dict()
        mas = target[...,0]>0
        new_target['labels'] = target[mas,0].to(torch.long)*0
        new_target['boxes'] = target[mas,1:]/args['input_size']
        new_targets.append(new_target)
    return new_targets
