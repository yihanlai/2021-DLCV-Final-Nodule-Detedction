# ------------------------------------------------------------------------
# LunaDataset
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from dlwpt-code (https://github.com/deep-learning-with-pytorch/dlwpt-code)
# Copyright (c). and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import hashlib
import os
from random import setstate
import shutil
import socket
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from dataloader import Luna2dDataset, TrainingLuna2dDataset, getCt
from util import enumerateWithEstimate
from logconf import logging
from model import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
METRICS_FN_LOSS_NDX = 2
METRICS_ALL_LOSS_NDX = 3

METRICS_PTP_NDX = 4
METRICS_PFN_NDX = 5
METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=24,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=50 ,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        self.args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None


        self.augmentation_dict = {}
        if self.args.augmented or self.args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.args.augmented or self.args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.args.augmented or self.args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.args.augmented or self.args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.args.augmented or self.args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()

    # define segmentation model(Unet) and augmentation method
    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=7,  # 七張切片
            n_classes=1,    # 是否為結節 
            depth=3,        # Unet的深度
            wf=4,           # Unet第一層的過濾器數量(2^wf)
            padding=True,   # 使輸出和輸入的圖片大小一樣
            batch_norm=True,
            up_mode='upconv',
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model
    
    # define optimizer(ADam)
    def initOptimizer(self):
        return Adam(self.segmentation_model.parameters())
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)

    # define train dataloader
    def initTrainDl(self):
        train_ds = TrainingLuna2dDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3, # 目標切片上下各切三片
        )

        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl
    
    # define val dataloader
    def initValDl(self):
        val_ds = Luna2dDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_trn_seg_' + self.args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_seg_' + self.args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        self.validation_cadence = 5 # 執行驗證的頻率
        for epoch_ndx in range(1, self.args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.args.epochs,
                len(train_dl),
                len(val_dl),
                self.args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()
        
        # dataloader enumerate 
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classificationThreshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            tp = (     predictionBool_g *  label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) *  label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g

    # 紀錄 model輸出結果(test 可參考)
    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12] # 從dataset中取出12組ct資料
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5 # 從該ct中取出六張間距相等的切片
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup
                # input slice and label
                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)
                
                # predicted label
                prediction_g = self.segmentation_model(input_g)[0]
                # 預測結果 包含圖片中個別像素為結節的機率
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1,:,:] /= 2000
                ct_t[:-1,:,:] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:,:,:] = ctSlice_a.reshape((512,512,1))         # 產生ct的灰階底圖
                image_a[:,:,0] += prediction_a & (1 - label_a)          # 偽陽性（紅色）
                image_a[:,:,0] += (1 - prediction_a) & label_a          # 偽陰性（橘色）
                image_a[:,:,1] += ((1 - prediction_a) & label_a) * 0.5  

                image_a[:,:,1] += prediction_a & label_a                # 真陽性（綠色）
                image_a *= 0.5
                image_a.clip(0, 1, image_a)
                
                # 輸出至tensorboard
                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

                # 產生ground truth image with label
                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    image_a[:,:,1] += label_a  # Green

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',
                    )
                # This flush prevents TB from getting confused about which data item belongs where.
                writer.flush()
    
    # 紀錄 model evaluation metric (tp, fp, fn)
    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        # log.info("E{} {}".format(
        #     epoch_ndx,
        #     type(self).__name__,
        # ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "loss: {loss/all:.4f} , "
                 + "precision: {pr/precision:.4f} , "
                 + "recall: {pr/recall:.4f} , "
                 + "f1 score: {pr/f1_score:.4f} "
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                  + "loss: {loss/all:.4f} , "
                  + "tp: {percent_all/tp:-5.1f}% , fn: {percent_all/fn:-5.1f}% , fp: {percent_all/fp:-9.1f}% "
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        # 以recall作為評分標準
        score = metrics_dict['pr/recall']
        return score
    
    # save model
    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join('models', self.args.tb_prefix, '{}_{}_{}.state'.format(
                type_str,
                self.time_str,
                self.totalTrainingSamples_count,
            ))

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model

        # 去除DataParallel包裹器
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(), 
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'models',
                self.args.tb_prefix,
                f'{type_str}_{self.time_str}_best.state')
            shutil.copyfile(file_path, best_path)
            
            ######### test save pth ################
            torch.save(state, os.path.join('models', self.args.tb_prefix, 'model_best.pth'))

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    SegmentationTrainingApp().main()