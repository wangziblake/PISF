# -*- coding: utf-8 -*-
"""
Reconstruction Code for PISF - physics-informed synthetic data learning for fast MRI - with 2D enhancement

Created on 2023/09/12

@author: Zi Wang

If you want to use this code, please cite following paper:

Zi Wang et al.,
One for Multiple: Physics-informed Synthetic Data Boosts Generalizable Deep Learning for Fast MRI Reconstruction,
arXiv:2307.13220, DOI: 10.48550/arXiv.2307.13220, 2023.

Email: Zi Wang (wangzi1023@stu.xmu.edu.cn; wangziblake@163.com) CC: Xiaobo Qu (quxiaobo@xmu.edu.cn)
Homepage: http://csrc.xmu.edu.cn

Affiliations: Computational Sensing Group (CSG), Departments of Electronic Science, Xiamen University, Xiamen 361005, China
All rights are reserved by CSG.
"""

import torch
import os
import warnings
import time
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Tools.Loaddata_1DLearning import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # only show the error information
warnings.filterwarnings("ignore")
print("pyTorch version: " + torch.__version__)  # Code for pyTorch 1.10.0


# --------------------------------------- #
# ---------------For users--------------- #
# Input datasets:
# Name of data - Number of data (See in file: DemoData_Result)
# FastMRI_PDWKnee-2/FastMRI_T2WBrain-2/GE_PDWAnkle-1/UI_T1FBrain-1
# Philips_T2FBrain_Patient-1/Siemens_CineCardiac_Patient-12

rec_data_name = 'FastMRI_PDWKnee'  # Name of test data
data_num_start = 1
data_all_num = 2  # Number of test data

SR = '25'  # Sampling rate: 13 20 25 33
sampling_pattern = '1D_Cartesian'  # 1D_Cartesian, 2D_Random
TEST_mode = '1D'  # 1D or 2D undersampling

# GPU Configs:
GPU_used = '0'  # Switch GPU
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0 use the first gpu in the gpu_list
# ------------------End------------------ #
# --------------------------------------- #


# ---------------Not for users--------------- #
# 2D k-space enhancement
KernalSize = [5, 5]
CalibTik = 0.01  # Tikhonov regularization in the calibration
SelectedPara = 'E_'  # Selected parameters for example: 'E_'

# Fixed network configs:
data_name = 'SynData'  # SynData
model_scheme = 'PISF'  # Model scheme
model_scheme_save = 'PISF_' + SelectedPara + '1DGau'
model_SR = '25'
PhaseNumber = 10

# Switch mode for 1D or 2D undersampling
if TEST_mode == '1D':
    from Models.PISF_Model_Enhance import *
elif TEST_mode == '2D':
    from Models.PISF_Model_2DinTest_Enhance import *
else:
    print('Error.')


# ---------------Start Reconstruction--------------- #
print("-----------------------------")
print("Sampling rate: %s%%" % SR)
print("Model: %s, %s" % (model_scheme_save, model_SR))
print("Train_Data: %s" % data_name)
print("-----------------------------")
print("Reconstruction starts.\n")


# ----- For parallel MRI reconstruction. -----#
# Input format: .mat
# Output format: .mat

# Loading Data
data_nus_dir = '../../DemoData_Result/TestData/{data_name}/{sampling_pattern}/SR{sampling_rate}/'\
    .format(data_name=rec_data_name, sampling_pattern=sampling_pattern, sampling_rate=SR)

data_rec_dir = '../../DemoData_Result/Result/{data_name}/{model_scheme}_{model_SR}_{data_train}/{sampling_pattern}/SR{sampling_rate}/'\
    .format(data_name=rec_data_name, model_scheme=model_scheme_save, model_SR=model_SR, data_train=data_name, sampling_pattern=sampling_pattern, sampling_rate=SR)

model_dir = ('%s_%s' % (model_scheme, data_name))

os.makedirs(data_rec_dir)

# Build model
model = PISF(PhaseNumber, ncoil=8)
model = model.to(device)

model.load_state_dict(torch.load('%s/Saved_Model.pth' % model_dir))

model.eval()  # no update BN
with torch.no_grad():
    for k in range(data_num_start, data_num_start + data_all_num):
        Testing_inputs_Norm, Testing_mask_coil, factor_norm, kernel_4d = \
            loaddata_Test_1DLearning_Enhancement(data_nus_dir, k, KernalSize, CalibTik)
        y_inputs = torch.tensor(np.transpose(Testing_inputs_Norm, (0, 3, 1, 2)), dtype=torch.complex64).to(device)
        mask = torch.tensor(np.transpose(Testing_mask_coil, (0, 3, 1, 2)), dtype=torch.complex64).to(device)
        kernel_r = torch.tensor(np.transpose(kernel_4d.real, (3, 2, 0, 1)), dtype=torch.float32).to(device)  # [ksize, ksize, ncoil, ncoil] -> [ncoil, ncoil, ksize, ksize]
        kernel_i = torch.tensor(np.transpose(kernel_4d.imag, (3, 2, 0, 1)), dtype=torch.float32).to(device)  # [ksize, ksize, ncoil, ncoil] -> [ncoil, ncoil, ksize, ksize]

        output = model(y_inputs, mask, kernel_r, kernel_i)  # include all network phases [nslice, ncoil, kx, ky]
        output_last = output[-1].cpu().data.numpy()

        k_rec = np.transpose(output_last, (0, 2, 3, 1))
        k_rec_de_norm = k_rec * factor_norm
        image_rec_de_norm = np_ifft1c_hybrid(k_rec_de_norm, 2)

        # Save as .mat
        mdict = {'PISF_image_rec': image_rec_de_norm}
        savefile_name = os.path.join(data_rec_dir, r'PISF_' + SelectedPara + str(k) + '.mat')
        sio.savemat(savefile_name, mdict)

        # ---------------Show images and PSNRs---------------#
        # If you have no fully sampled data (i.e., prospective experiment), please comment the following two lines
        # and also other lines related to it in image visualization
        Label_image = loaddata_Label_1DLearning(data_nus_dir, k)  # Only for comparison and evaluation
        Label_image_sos = np_sos(Label_image)[0]

        ZF_image_sos = np_sos(np_ifft1c_hybrid(Testing_inputs_Norm, 2))[0]
        PISF_image_sos = np_sos(image_rec_de_norm)[0]
        Label_image_sos_norm = Label_image_sos / np.amax(Label_image_sos)
        ZF_image_sos_norm = ZF_image_sos / np.amax(ZF_image_sos)
        PISF_image_sos_norm = PISF_image_sos / np.amax(PISF_image_sos)
        Mask = np.abs(np.squeeze(Testing_mask_coil[0, :, :, 0]))

        PSNR_ZF = myPSNR(Label_image_sos_norm, ZF_image_sos_norm)
        PSNR_PISF = myPSNR(Label_image_sos_norm, PISF_image_sos_norm)

        myplot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
        plt.clf()
        plt.subplot(141)
        myplot(Mask)
        plt.axis('off')
        plt.title('Mask ' + SR + '%')
        plt.subplot(142)
        myplot(Label_image_sos_norm)
        plt.axis('off')
        plt.title('Fully-sampled')
        plt.subplot(143)
        myplot(ZF_image_sos_norm)
        plt.title('Zero-filled\nPSNR=' + str(PSNR_ZF.round(2)) + ' dB')
        plt.axis('off')
        plt.subplot(144)
        myplot(PISF_image_sos_norm)
        plt.title('PISF-Rec\nPSNR=' + str(PSNR_PISF.round(2)) + ' dB')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
        plt.show()
        plt.savefig('../' + rec_data_name + '_' + sampling_pattern + '_' + SR + '_' + str(k) + '.png')
# ----- End parallel MRI reconstruction. -----#


# ---------------End Reconstruction--------------- #
print("\nReconstruction finishes.")
print("-----------------------------")
print("Sampling rate: %s%%" % SR)
print("Model: %s, %s" % (model_scheme_save, model_SR))
print("Rec_Data: %s" % rec_data_name)
print("Number of samples: %d" % data_all_num)
print("-----------------------------")
