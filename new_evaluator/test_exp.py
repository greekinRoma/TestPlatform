import os.path
import torch
from DataLoader.dataloader import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from DataLoader.dataset.valtransform import ValTransform
from DataLoader.dataset.testdataset import TestDataset
from new_evaluator.coco import coco
from utils import *
from DataLoader.dataset.data_cache import DataCache
from thop import profile
import time
class TestExp():
    def __init__(self,
                 args,
                 datacache:DataCache,
                 use_cuda,
                 data_dir,
                 save_dir,
                 use_tide):
        super(TestExp, self).__init__()
        self.args = args
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.65
        self.test_size=(640,640)
        self.test_conf=0.001
        self.nmsthre=0.5
        self.use_cuda=use_cuda
        self.use_tide=use_tide
        self.data_dir=data_dir
        mode='test'
        self.source= datacache.test_source
        self.test_dataset = TestDataset(base_dataset=self.source,
                                        preproc=ValTransform())
        self.loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=1,
                                 use_cuda=self.use_cuda,
                                 use_shuffle=False,)
        self.num_img=len(self.source)
        self.act='silu'
        self.save_dir = self.get_index(save_dir)
        self.save_file= os.path.join(self.save_dir,'save_pred.json')
        self.save_txt= os.path.join(self.save_dir,'save_pred.txt')
        self.pic_type='.png'
        self.itr=0
        self.photo_dir=os.path.join(self.save_dir, "conf={},nms={}".format(self.test_conf,self.nmsthre))
        self.coco=coco(save_dir=save_dir,image_set=mode,year='2007',data_dir=data_dir)
        self.outcome_file=os.path.join(self.save_dir,"outcome.txt")
        self.names=[]
        self.values=[]
        os.makedirs(self.photo_dir, exist_ok=True)
    def get_index(self,save_dir):
        max = 0
        for dir in os.listdir(save_dir):
            if dir.isdigit():
                dir = int(dir)
                if (max < dir):
                    max = dir
        max = max + 1
        dir=os.path.join(save_dir,str(max))
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir
    def push_network(self,model):
        self.model = model
        if self.use_cuda:
            self.model = self.model.cuda()
        torch.save(self.model,os.path.join(self.save_dir,'save_weight.pth'))
        self.model.eval()
    def show_prediction(self,imgs,outcomes,targets,scores):
        for img_g,outcome,target,score in zip(imgs,outcomes,targets,scores):
            img_g=img_g.permute(1,2,0).cpu().numpy().astype(np.uint8)
            img_g=np.ascontiguousarray(img_g)
            pred_boxes=outcome[:,:4]
            gt_boxes=target[:,1:]
            img=img_g[...,2:3]
            img=np.repeat(img,3,axis=2)
            pos_mask=score>0.5
            img=self.draw_box(img,pred_boxes[pos_mask],(255,0,0),scores=score[pos_mask])
            img=self.draw_box(img,gt_boxes,(0,255,0))
            cv2.imwrite(os.path.join(self.photo_dir,"{}.png".format(self.itr)),img)
            self.itr=self.itr+1
    def model_predict(self, image):
        with torch.no_grad():
            bboxes, scores = self.model(image)
            o_bboxes = []
            o_scores = []
            for bbox,score in zip(bboxes,scores):
                o_bboxes.append(bbox)
                o_scores.append(score)
        return o_bboxes, o_scores
    def draw_box(self,image,boxes_g,color=(0,255,0),class_name='ship',scores=None):
        boxes = self.tranform_int(boxes_g).copy()
        for i,box in enumerate(boxes):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color)
            if scores is None:
                continue
            score=scores[i,0]
            text = '{}:{:.1f}%'.format(class_name, score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                image,
                (box[0], box[1] + 31),
                (box[0] + txt_size[0] + 1, box[1] + int(1.5 * txt_size[1])+30),
                (255,0,0),
                -1)
            cv2.putText(image, text, (box[0], box[1] + txt_size[1]+30), font, 0.4, (255,255,255))
        return image
    def save_pred(self):
        all_boxes =[[]]
        f = open(self.save_txt, 'w')
        ######################################
        res = []
        for i,(imgs,_,_,targets,names) in enumerate(tqdm(self.loader)):
            start = time.time()
            outcomes, scores = self.model_predict(imgs)
            end = time.time()
            res.append(end-start)
            self.show_prediction(imgs,outcomes,targets,scores)
            if len(scores)<=0:
                all_boxes[0].append(np.array([[0,0,0,0,0]]))
            outcomes = outcomes 
            for name,score,outcome in zip(names,scores,outcomes):
                det=np.concatenate([score,outcome* self.args.origin_size/self.args.input_size],-1)
                all_boxes[0].append(det)
                for s, o in zip(score, outcome):
                    f.write("{} {} {} {} {} {}\n".format(name,float(s), o[0], o[1], o[2], o[3]))
        self.coco._write_coco_results_file(all_boxes=all_boxes,res_file=self.save_file)
        time_sum = 0
        for i in res:
            time_sum += i
        self.names.append("FPS")
        self.values.append(1.0/(time_sum/len(res)))
        self.get_parameters()
        self.get_gflops()
        return self.save_dir
    def get_parameters(self):
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.model.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量
        self.names.append("params")
        self.values.append(Total_params / 1e6)
    def get_gflops(self):
        inputs = torch.rand([1,3,640,640]).cuda()
        flops, params = profile(self.model, inputs=(inputs))
        self.names.append("gflops")
        self.values.append(flops / 1e9 * 2)
    def tranform_int(self, boxes):
        box_list = []
        for box in boxes:
            box_list.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        return box_list
    def read_coco_outcome(self):
        stats, precision = self.coco._do_detection_eval(res_file=self.save_file, output_dir=self.save_dir)
        names=[]
        values=[]
        for stat in stats:
            names.append(stat[0])
            values.append(stat[1])
        self.get_p_r(precision)
        np.save(os.path.join(self.save_dir,"precisions"),precision)
        self.write_excel(sheet_name="coco",names=names,values=values)
    def write_excel(self,sheet_name,names,values):
        sheet=self.f.add_sheet(sheet_name,True)
        for i,(name,value) in enumerate(zip(names,values)):
            sheet.write(0,i,name)
            sheet.write(1,i,value)
    def compute_ap(self):
        import xlwt
        self.f=xlwt.Workbook('encoding =utf-8')
        self.read_coco_outcome()
        self.f.save(os.path.join(self.save_dir, 'outcome.xls'))
if __name__=="__main__":
    # print(os.listdir(r"../datasets/ISDD/VOC2007"))
    exp=TestExp(
        use_cuda=cfg.use_cuda,
        data_dir=r'./datasets/SII',
        save_dir=r'./save_outcome',
        use_tide=True)
    exp.push_network(r'./new_evaluator/save_weight.pth')
    exp.save_pred()
    exp.compute_ap()