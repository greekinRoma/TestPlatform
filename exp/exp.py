from network.Network import Network
from DataLoader.dataloader import DataLoader
from torch import nn
import os
from new_evaluator.train_eval import Evaluator
from utils.Txt_controller import txt_writer
from DataLoader.dataset.traindatasets import TrainDatasets
from DataLoader.dataset.testdataset import TestDataset
from DataLoader.dataset.traintransform import TrainTransform
from DataLoader.dataset.valtransform import ValTransform
from DataLoader.dataset.dataaugment import AugmentController
from DataLoader.dataset.data_cache import DataCache
from Models import Get_Network
class MyExp(nn.Module):
    def __init__(self,data_cache:DataCache,args,save_path):
        super(MyExp,self).__init__()
        self.data_dir =args['coco_data_dir']
        self.target_dir=args['target_dir']
        self.use_cuda=args['use_cuda']
        #------------------------dataset---------------------------#
        self.enable_mosaic=args['enable_mosaic']
        self.use_valid=args['use_valid']
        self.name=args['name']
        self.batch_size = args['batch_size']
        self.mosaic_prob = args['mosaic_prob']
        self.mixup_prob = args['mixup_prob']
        # ------------------------------------------------------------------------dataset_controller-------------------------------------------------------------#
        self.soft_finetune_beta = args['beta']
        self.trainval_source = data_cache.trainval_source
        self.test_source = data_cache.test_source
        self.mask_source= data_cache.mask_source
        #self.mask_source.get_ids()
        self.aug_controller = AugmentController(input_w=args['img_size'][0],
                                           input_h=args['img_size'][1],
                                           degrees=args['degrees'],
                                           translate=args['translate'],
                                           mosaic_scale=args['mosaic_scale'],
                                           mixup_scale=args['mixup_scale'],
                                           shear=args['shear'])
        self.train_dataset = TrainDatasets(
            masksource=self.mask_source,
            vocsource=self.trainval_source,
            targetsource=None,
            maxtarget=args['maxtarget'],
            aug_prob=args['aug_prob'],
            mixup_mosaic_prob=args['mixup_mosaic_prob'],
            mixup_prob=args['mixup_prob'],
            mosaic_prob=args['mosaic_prob'],
            flip_prob=args['flip_prob'],
            gen_prob=args['gen_prob'],
            mixup=self.aug_controller.get_mixup(),
            mosaic=self.aug_controller.get_mosaic(),
            preproc=TrainTransform()
        )
        self.test_dataset = TestDataset(base_dataset=self.test_source,
                                        preproc=ValTransform())
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.batch_size,
                                       use_shuffle=True,
                                       use_cuda=self.use_cuda)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=1,
                                      use_shuffle=False,
                                      use_cuda=self.use_cuda)
        #---------------------------path----------------------------#
        self.save_path = save_path
        self.file_name = os.path.join('training_save', args['net_name'] + '_' + "OTA")
        #---------------------------evaluator-----------------------#
        self.txt_writer=txt_writer(self.file_name,r'preform.txt')
        # ----------------------------------------------------------#
        self.num_batch=len(self.train_loader)
        self.max_epoch=args['max_epoch']
        self.input_size=640
        self.train_network = Get_Network('detr')
        self.model = self.train_network.model
        self.process = self.train_network.postprocessors
        if self.use_cuda:
            self.model=self.model.cuda()
        # --------------  evaluation --------------------- #
        self.test_evaluator = Evaluator(self.test_loader,need_change=False)
        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = args['max_epoch']
        self.print_interval = 2
        self.eval_interval = 1
        self.save_history_ckpt = True
    def get_trainloader(self):
        return self.train_loader
    def get_model(self):
        return self.model
    def reset_map(self):
        self.train_loader.resetmap()
    def get_test_evaluator(self):
        return self.test_evaluator
    def get_testloader(self):
        return self.test_loader
    def save_index(self,save_dir):
        self.txt_writer.save_file(save_dir)