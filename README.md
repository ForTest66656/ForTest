# LDDMM-Face: Large Deformation Diffeomorphic Metric Learning for Cross-annotation Face Alignment

This repository contains an implementation of LDDMM-Face for the TIP submission "LDDMM-Face: Large Deformation Diffeomorphic Metric Learning for Cross-annotation Face Alignment".

## Quick start
### Environment
This code is developed using on **Python 3.6** and Pytorch 1.7.1 on CentOS 7 with NVIDIA GPUs. Training and testing are performed using 1 RTX 3090 GPU with CUDA 11.0. Other platforms or GPUs are not fully tested.

### Install
1. Install Pytorch
2. Install dependencies
```shell
pip install -r requirements.txt
```

### Data
1. You need to download the annotations files which have been processed from [GoogleDrive](https://drive.google.com/drive/folders/1XhQParhbEKzOb2TKbDHlvYAhSceNAjW2?usp=sharing).

2. You need to download images (300W, WFLW) from official websites and then put them into `images` folder for each dataset.

3. You need to download annotations (HELEN) from official websites and then put them into `helen` folder.

Your `data` directory should look like this:

````
HRNet-Facial-Landmark-Detection
-- lib
-- experiments
-- tools
-- data
   |-- 300w
   |   |-- face_landmarks_300w_test.csv
   |   |-- face_landmarks_300w_train.csv
   |   |-- face_landmarks_300w_valid.csv
   |   |-- face_landmarks_300w_valid_challenge.csv
   |   |-- face_landmarks_300w_valid_common.csv
   |   |-- images
   |   |   |-- helen
   |   |   |   |-- annotation
   |   |   |   |-- train.txt
   |   |   |   |-- test.txt
   |   |   |-- afw
   |   |   |-- 300W
   |   |   |-- ibug
   |   |   |-- lfpw
   |-- cofw
   |   |-- test_annotations
   |   |-- COFW_test_color.mat
   |   |-- COFW_train_color.mat  
   |-- wflw
   |   |-- face_landmarks_wflw_test.csv
   |   |-- face_landmarks_wflw_test_blur.csv
   |   |-- face_landmarks_wflw_test_expression.csv
   |   |-- face_landmarks_wflw_test_illumination.csv
   |   |-- face_landmarks_wflw_test_largepose.csv
   |   |-- face_landmarks_wflw_test_makeup.csv
   |   |-- face_landmarks_wflw_test_occlusion.csv
   |   |-- face_landmarks_wflw_train.csv
   |   |-- images

````

4. Generate `face_landmarks_helen_test.csv` and `face_landmarks_helen_train.csv`.
```shell
python data/300w/helen/txt2csv.py
```

### Train
Please specify the configuration file in ```experiments```.

**Training script will be released once the paper is accepted.**
```shell
python tools/train.py --cfg <CONFIG-FILE>
```

### Test
Download pretrained models from [OneDrive](https://bigbigchina-my.sharepoint.com/:f:/g/personal/t4486_tvv_ink/EtHI_VGMe-5LmN_unFK2U00B6QR_3yJs-WEnWcOWeB4hMA?e=h9eHkW).
```shell
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL-FILE>
```
300W training label fraction 100%
```shell
python tools/test.py --cfg experiments/300w/face_alignment_300w_lddmm_hrnet_w18.yaml --model-file pretrained/300W_lddmm_hrnet.pth
```
300W training label fraction 50%
```shell
python tools/test.py --cfg experiments/300w/face_alignment_300w_lddmm_hrnet_w18_p50.yaml --model-file pretrained/300W_lddmm_hrnet_p50.pth
```
WFLW training label fraction 100%
```shell
python tools/test.py --cfg experiments/wflw/face_alignment_wflw_lddmm_hrnet_w18.yaml --model-file pretrained/WFLW_lddmm_hrnet.pth
```
WFLW training label fraction 50%
```shell
python tools/test.py --cfg experiments/wflw/face_alignment_wflw_lddmm_hrnet_w18_p50.yaml --model-file pretrained/WFLW_lddmm_hrnet_p50.pth
```
HELEN training label fraction 100%
```shell
python tools/test.py --cfg experiments/helen/face_alignment_helen_lddmm_hrnet_w18.yaml --model-file pretrained/HELEN_lddmm_hrnet.pth
```
HELEN training label fraction 50%
```shell
python tools/test.py --cfg experiments/helen/face_alignment_helen_lddmm_hrnet_w18_p50.yaml --model-file pretrained/HELEN_lddmm_hrnet_p50.pth
```
HELEN training label fraction 33%
```shell
python tools/test.py --cfg experiments/helen/face_alignment_helen_lddmm_hrnet_w18_p33.yaml --model-file pretrained/HELEN_lddmm_hrnet_p33.pth
```
