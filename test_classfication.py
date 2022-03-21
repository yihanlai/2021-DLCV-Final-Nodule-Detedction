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
from dataloader import LunaTestDataset, getDataInfoDict, getTestDataList, getTestCt, LunaTestFalsePositiveDataset
from model import UNetWrapper
from model_classification import LunaModel
from logconf import logging
from util import xyz2irc, irc2xyz

from pytorch3dunet.unet3d.model import ResidualUNetFeatureExtract
from pytorch3dunet.unet3d.losses import DiceLoss
from pytorch3dunet.unet3d.metrics import MeanIoU


log = logging.getLogger(__name__)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, diameter_mm, series_uid, center_xyz',
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

        parser.add_argument('--segmentation-path',
            help="Path to the segmentation model",
            default='2D-Unet-segmentation.pthstate',
        )

        parser.add_argument('--classification-path',
            help="Path to the classification model",
            default='3D-Unet-classifier.pth'
        )

        self.cli_args = parser.parse_args(sys_argv)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.seg_model, self.cls_model = self.initModels()

    def initModels(self):
        #### segmentation model
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

        #### classifacation model
        cls_dict = torch.load(self.cli_args.classification_path)
        cls_model = ResidualUNetFeatureExtract(in_channels=1, out_channels=2)
        cls_model.load_state_dict(cls_dict)
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)
            
        
        
        return seg_model, cls_model

    def initSegmentationDl(self, series_uid):
        seg_ds = LunaTestDataset(series_uid)

        return seg_ds

    def initClassificationDl(self, series_uid, candidateInfo_list):
        # cls_ds = LunaDataset(
        #         sortby_str='series_uid',
        #         candidateInfo_list=candidateInfo_list,
        #     )

        cls_ds = LunaTestFalsePositiveDataset(
            series_uid=series_uid,
            candidateInfo_list=candidateInfo_list,
        )

        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl

    def main(self):
        # val_set = set(
        #     candidateInfo_tup
        #     for candidateInfo_tup in getTestDataList()
        # )
        # val_list = sorted(list(val_set))
        # total_uid = len(val_list)
        
        test_set = set(
            candidateInfo_tup
            for candidateInfo_tup in getTestDataList()
        )
        test_list = list(test_set)
        total_uid = len(test_list)

        series_iter = enumerateWithEstimate(
            test_list ,
            "Series",
        )
    
        # output csv
        with open('output_classifation_testing.csv', 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            field = ['seriesuid','coordX','coordY','coordZ','probability']
            # field = ['seriesuid','coordX','coordY','coordZ', 'candidate']
            csv_writer.writerow(field)
            csvfile.flush() 

            # ####### load output_classification_candidates.csv as classifacation input
            # all_candidateInfo_list = []
            # with open('output_classifation_candidates.csv', 'r') as inputfile:
            #         for row in list(csv.reader(inputfile))[1:]:
            #             series_uid = row[0]
            #             CandidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            #             candidateInfo_tup = CandidateInfoTuple(False, False, 0.0, series_uid, CandidateCenter_xyz)
            #             all_candidateInfo_list.append(candidateInfo_tup)

            for i, data in enumerate(series_iter) :
                series_uid = data[1]
                ct = getTestCt(series_uid)

                mask_a, output_a = self.segmentCt(ct, series_uid)
                candidateInfo_list = self.groupSegmentationOutput(series_uid, ct, mask_a, output_a)
                count_candidate = len(candidateInfo_list)
                
                # candidateInfo_list = [candidate for candidate in all_candidateInfo_list if candidate.series_uid == series_uid]
                count_candidate = len(candidateInfo_list)
                print(f"[{i+1}/{total_uid}]: found {count_candidate} nodule candidates in {series_uid}")
                
                classifications_list = self.classifyCandidates(ct, series_uid,candidateInfo_list)
                count_final = 0
                # annos = []
                
                for prob, center_xyz, center_irc in classifications_list:
                    # candidate_bool, annos =  self.check(series_uid, center_xyz)
                    
                    if prob > 0.8: # predict to 1 (is nodule)
                        prob = round(prob, 2)
                        count_final += 1
                        s = f"center xyz {center_xyz}, probability {prob}"
                        print(s) 

                        data = [series_uid, center_xyz[0], center_xyz[1], center_xyz[2], prob]
                        csv_writer.writerow(data)
                        csvfile.flush() 
                        data = [] 
                        
                        # 到底是不是真的真的
                        # if candidate_bool: # 是假的
                        #     data = [series_uid, center_xyz[0], center_xyz[1], center_xyz[2], 0]
                        #     csv_writer.writerow(data)
                        #     csvfile.flush() 
                        #     data = []        

                # for anno in  annos:
                #     data = [series_uid, anno[0], anno[1], anno[2], 1]
                #     csv_writer.writerow(data)
                #     csvfile.flush() 
                #     data = []
                    
                print(f"[{i+1}/{total_uid}]: found {count_final} nodules in {series_uid}") 
                print("######################################")
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


            candidateInfo_tup = CandidateInfoTuple(False, False, 0.0, series_uid, center_xyz)
            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list 

    # 分類模型
    def classifyCandidates(self, ct, series_uid, candidateInfo_list):
            cls_dl = self.initClassificationDl(series_uid, candidateInfo_list)
            classifications_list = []
            
            for batch_ndx, batch_tup in enumerate(cls_dl):
                input_t, center_list = batch_tup
                input_g = input_t.to(self.device)
                
                with torch.no_grad():
                    probability_nodule_g = self.cls_model(input_g)

                output = torch.argmax(probability_nodule_g, axis=1)
                # print("prob: ", probability_nodule_g)
                # print("class: ", output)
            
                zip_iter = zip(center_list, probability_nodule_g[:,1].tolist())
                # zip_iter = zip(center_list, output.tolist())
                
                for center_irc, prob_nodule in zip_iter:
                    center_xyz = irc2xyz(center_irc,
                        direction_a=ct.direction_a,
                        origin_xyz=ct.origin_xyz,
                        vxSize_xyz=ct.vxSize_xyz,
                    )
                    cls_tup = (prob_nodule, center_xyz, center_irc)
                    classifications_list.append(cls_tup)

            return classifications_list

    # check if find the true nodules (for false positive reduction)
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
                    # print(f"##### anno x, y, z = {x}, {y}, {z}, pred x, y, z = {x_}, {y_}, {z_}") 
        return candidate_bool, anno

if __name__ == '__main__':
    NoduleAnalysisApp().main()