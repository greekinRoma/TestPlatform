from new_evaluator.voc_eval import *
import torch
from utils.boxes import postprocess
from tqdm import tqdm
from new_evaluator.coco import coco
class Evaluator():
    def __init__(self,dataloader,need_change):
        super(Evaluator, self).__init__()
        self.dataloader=dataloader
        self.net=None
        self.num_classes=1
        self.test_conf=0.001
        self.nmsthre=0.5
        self.need_change=need_change
        self.coco = coco(save_dir='./', image_set='test', year='2007')
        self.save_file='./content.json'
    def eval(self,network):
        assert network.model is not None,'you should push a model,and the model is None!'
        all_boxes = [[]]
        for i,(imgs,_,_,targets,names) in enumerate(tqdm(self.dataloader)):
            outcomes, scores = network(imgs)
            for i,(outcome,score) in enumerate(zip(outcomes,scores)):
                if len(score)<=0:
                    all_boxes[0].append(np.array([[0,0,0,0,0]]))
                else:
                    det = np.concatenate([score, outcome], -1)
                    all_boxes[0].append(det)
        self.coco._write_coco_results_file(all_boxes=all_boxes, res_file=self.save_file)
        stats,precision = self.coco._do_detection_eval(res_file=self.save_file,output_dir='./')
        return stats