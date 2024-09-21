import datetime
import random
import time
import loguru
import numpy as np
from DataLoader.dataprefetcher import DataPrefetcher
from exp.exp import MyExp
from utils import *
from utils.model_utils import *
import traceback
import torch
import os
from loguru import logger
import shutil
class Trainer:
    def __init__(self, exp:MyExp):
        self.exp = exp
        self.max_epoch = int(exp.max_epoch)
        self.network = exp.train_network
        self.save_history_ckpt = exp.save_history_ckpt
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.use_cuda=self.exp.use_cuda
        self.args = self.exp.args
        # data/dataloader related attr
        self.data_type = torch.float32
        self.input_size = exp.input_size
        self.soft_finetune_beta = exp.soft_finetune_beta
        self.best_ap = 0
        self.start_epoch=0
        #controller
        self.use_valid=exp.use_valid
        # metric record
        self.file_name =exp.file_name
        self.save_path=exp.save_path
        self.txt_writer=exp.txt_writer
        self.encoder = self.exp.encoder
        os.makedirs(self.file_name,exist_ok=True)
        setup_logger(self.file_name,distributed_rank=0,filename="train_log.txt",mode="a")
    def train(self):
        try:
            self.before_train()
            self.train_in_epoch()
        except (Exception,BaseException) as e:
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            raise
        finally:
            self.after_train()
    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()
    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()
    def train_one_iter(self):
        inps,masks,use_augs,targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        if self.encoder is not None:
            targets = self.encoder(targets,args=self.args)
        outputs = self.network.train(inputs=inps,targets=targets,iter=self.iter)
        self.meter.update(
            iter_time = 0,
            data_time = 0,
            lr = 0,
            ** outputs,)
    def test_dataloader(self):
        from VisionTools.save_img import save_outcome
        self.test_trainloader_dir = os.path.join(self.file_name, r'trainloader')
        self.test_testloader_dir = os.path.join(self.file_name, r'testloader')
        if os.path.exists(self.test_trainloader_dir):
            shutil.rmtree(self.test_testloader_dir)
        if os.path.exists(self.test_trainloader_dir):
            shutil.rmtree(self.test_trainloader_dir)
        os.makedirs(self.test_trainloader_dir, exist_ok=True)
        os.makedirs(self.test_testloader_dir, exist_ok=True)
        for imgs,_,_,targets,names in self.train_loader:
            save_outcome(names=names,
                         save_dir=self.test_trainloader_dir,
                         labels=targets,
                         imgs=imgs,
                         need_change=True)
        test_loader=self.exp.get_testloader()
        for imgs,_,_,targets,names in test_loader:
            save_outcome(names=names,
                         save_dir=self.test_testloader_dir,
                         labels=targets,
                         imgs=imgs,
                         need_change=False)
    def before_train(self):
        # data related init
        self.train_loader = self.exp.get_trainloader()
        logger.info("init prefetcher, this might take one minute or less...")
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)
        self.test_evaluator = self.exp.get_test_evaluator()
        np.random.seed(int(time.time()))
    def after_train(self):
        logger.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        self.exp.reset_map()
        prob=1.- self.soft_finetune_beta*(self.epoch/self.max_epoch)**2
        self.train_loader.reset_prob(prob)
        self.prefetcher = DataPrefetcher(self.train_loader,
                                         use_cuda=self.use_cuda)
        self.prefetcher.preload()
    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.network.model)
            self.evaluate_and_save_model()
    def before_iter(self):
        t=time.time()
        torch.manual_seed(int(t))

    def after_iter(self):
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size, eta_str))
            )
            self.meter.clear_meters()

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

#/--------------------------------------------------test evaluator-------------------------------------------------------/#
    def evaluate_test_model(self):
        stats= self.test_evaluator.eval(self.network)
        return stats
    def save_test_model(self,update_best_ckpt,ap50_95):
        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)
    def evaluate_and_save_model(self):
        stats=self.evaluate_test_model()
        ap50=stats[1][1]
        # ap50_95=stats[0][1][1]
        # logger.info("AP@50:{}".format(ap50))
        # logger.info("AP@50:95:{}".format(ap50_95))
        update_best_ckpt = ap50 > self.best_ap
        self.best_ap = max(self.best_ap, ap50)
        self.save_test_model(update_best_ckpt=update_best_ckpt,ap50_95=ap50)
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": self.network.model.state_dict(),
            "best_ap": self.best_ap,
            "curr_ap": ap,
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )
