# ------------------------------------------------------------------------
# LunaDataset
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from dlwpt-code (https://github.com/deep-learning-with-pytorch/dlwpt-code)
# Copyright (c). and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import glob
import os
import sys
from typing import Tuple
import csv
from collections import namedtuple

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from util import enumerateWithEstimate
from dataloader import Luna2dDataset, LunaTestDataset, getDataList, getDataInfoDict, getCt, getTestDataList, getTestCt
from model import UNetWrapper

from logconf import logging
from util import xyz2irc, irc2xyz

log = logging.getLogger(__name__)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'series_uid, center_xyz',
)

class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=24,
            type=int,
        )

        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )
        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False,
        )

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='models/training_v1/seg_2022-01-11_17.21.13_best.state',
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        self.cli_args = parser.parse_args(sys_argv)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.seg_model = self.initModels()

    def initModels(self):
        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)

            seg_model.to(self.device)

        return seg_model

    def initSegmentationDl(self, series_uid):
        seg_ds = LunaTestDataset(series_uid)

        # seg_ds = Luna2dDataset(
        #         contextSlices_count=3,
        #         series_uid=series_uid,
        #         fullCt_bool=True,
        #     )
        # seg_dl = DataLoader(
        #     seg_ds,
        #     batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
        #     num_workers=self.cli_args.num_workers,
        #     pin_memory=self.use_cuda,
        # )
        
        return seg_ds

    def main(self):

        # val_ds = Luna2dDataset(
        #     val_stride=10,
        #     isValSet_bool=True,
        # )

        # # val_set:　some uid in validation dataset 
        # val_set = set(
        #     candidateInfo_tup
        #     for candidateInfo_tup in val_ds.series_list
        # )

        # # 　series_set: all uid in training data or specific uid
        # if self.cli_args.series_uid:
        #     series_set = set(self.cli_args.series_uid.split(','))
        # else:
        #     series_set = set(
        #         candidateInfo_tup.series_uid
        #         for candidateInfo_tup in getDataList()
        #     )
        
        # val_list = sorted(series_set & val_set)

        
        val_set = set(
            candidateInfo_tup
            for candidateInfo_tup in getTestDataList()
        )
        val_list = sorted(list(val_set))
        total_uid = len(val_list)
        
        # test_set = set(
        #     candidateInfo_tup
        #     for candidateInfo_tup in getTestDataList()
        # )
        # test_list = list(test_set)
        
        series_iter = enumerateWithEstimate(
            val_list ,
            "Series",
        )
    
        # output csv
        with open('output_candidate.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            # field = ['seriesuid','coordX','coordY','coordZ','probability']
            field = ['seriesuid','coordX','coordY','coordZ', 'candidate']
            csv_writer.writerow(field)
        
            count_true = 0  
            count_false = 0
        
            for i, data in enumerate(series_iter) :
                series_uid = data[1]
                ct = getTestCt(series_uid)

                # 預測結節
                mask_a, output_a = self.segmentCt(ct, series_uid)
                candidateInfo_list = self.groupSegmentationOutput(series_uid, ct, mask_a, output_a)
                count = len(candidateInfo_list)
                annos = []
                
                print(f"[{i+1}/{total_uid}]: found {count} nodule candidates in {series_uid}")
                for candidate in candidateInfo_list:
                    center_xyz = candidate.center_xyz
                    # prob = candidate.prob
                    # s = f"center xyz {center_xyz}, probability {prob}"
                    # print(s)  

                    candidate_bool, annos =  self.check(series_uid, center_xyz)

                    if candidate_bool: # 是假的
                        data = [series_uid, center_xyz[0], center_xyz[1], center_xyz[2], 0]
                        csv_writer.writerow(data)
                        data = []
                        count_false +=1 

                count_true += len(annos)
                for anno in  annos:
                    data = [series_uid, anno[0], anno[1], anno[2], 1]
                    csv_writer.writerow(data)
                    data = [] 
                # print("######################################")
                # exit()

        
    # 預測模型輸出
    def segmentCt(self, ct, series_uid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            seg_dl = self.initSegmentationDl(series_uid)
            
            ######## for testing ########
            input_t = seg_dl[0][0]
            input_g = input_t.to(self.device)

            for i, input_g_slice in enumerate(input_g):
                prediction_g = self.seg_model(input_g_slice.unsqueeze(0))
                output_a[i] = prediction_g.squeeze(0).squeeze(0).detach().cpu().numpy()
            
            ######## for eval ########
            # for input_t, _, _, slice_ndx_list in seg_dl:
            #     print(input_t.shape)
            #     input_g = input_t.to(self.device)
            #     prediction_g = self.seg_model(input_g)

            #     # 將結果複製到輸出array中
            #     for i, slice_ndx in enumerate(slice_ndx_list):
            #         output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            # 輸出的機率值與treshold比較 得到binary mask
            mask_a = output_a > 0.5 
            
            # erosion operation
            mask_a = morphology.binary_erosion(mask_a, iterations=1)
            # dilation operation
            # mask_a = morphology.binary_dilation(mask_a, iterations=5)
        
        return mask_a, output_a
    

    #　分組找出結節位置
    def groupSegmentationOutput(self, series_uid,  ct, clean_a, output_a):
        
        # connected-components algorithm to group the pixels
        candidateLabel_a, candidate_count = measurements.label(clean_a)

        # 求 probability
        # prob_list = []
        # labels = np.unique(candidateLabel_a)
        # for i in labels[1:]:
        #     mask = np.where(candidateLabel_a == i)
        #     prob = np.mean(output_a[mask])
        #     prob_list.append(np.round(prob, 2))

        # objects = measurements.find_objects(candidateLabel_a)
        
        # 求質心代表結節位置 
        centerIrc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001, # 輸入不能為複數所以+1001
            labels=candidateLabel_a,
            index=np.arange(1, candidate_count+1),
        )

        candidateInfo_list = []
        for i, center_irc in enumerate(centerIrc_list): # IRC to XYZ
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            
            
            # object_irc = objects[i]
            # i_length = object_irc[0].stop - object_irc[0].start
            # r_length = object_irc[1].stop - object_irc[1].start
            # c_length = object_irc[2].stop - object_irc[2].start
            # volumn = i_length*ct.vxSize_xyz.z + r_length*ct.vxSize_xyz.x + c_length*ct.vxSize_xyz.y
            # if(volumn < 100):
            #     continue

            # candidateInfo_tup = CandidateInfoTuple(series_uid, center_xyz, prob_list[i])
            candidateInfo_tup = CandidateInfoTuple(series_uid, center_xyz)
            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list 

    def check(self, series_uid=None, center_xyz=None):
        dataInfo_dict = getDataInfoDict()
        candidate_bool = True
        anno = []
        if series_uid  in dataInfo_dict.keys():
            current_data = dataInfo_dict[series_uid]
            x_, y_, z_ = center_xyz[0], center_xyz[1], center_xyz[2]

            for tup in current_data:
                r = tup[0]
                x, y, z = tup[2][0], tup[2][1], tup[2][2] 
                R = r/2
                anno.append([x, y, z])
                if (np.abs(x-x_) < R) and (np.abs(y-y_)<R) and (np.abs(z-z_)<R):
                    candidate_bool = False
                    print(f"##### anno x, y, z = {x}, {y}, {z}, pred x, y, z = {x_}, {y_}, {z_}") 
        return candidate_bool, anno


if __name__ == '__main__':
    NoduleAnalysisApp().main()