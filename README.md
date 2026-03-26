# PhysDrive Dataset

## 📖 Abstract

**Here is [NIPS2025] PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring, collected by The Hong Kong University of Science and Technology (Guangzhou).**  
The PhysDrive, comprises about 24 hours(**1500K frames**) of recordings from RGB camera, NIR camera, and an mmWave radar of 48 subjects. The dataset was designed to ontactless in-vehicle physiological sensing with dedicated consideration on various modality settings and driving factors, including **Three vehicle types, four illumination conditions, three road conditions, and driver motions**. PhysDrive is comprehensive with six synchronized ground truths (**ECG, BVP, Respiration, HR, RR, and SpO2**) and can be used in conjunction with the [rPPG-toolbox](https://github.com/ubicomplab/rPPG-Toolbox).  

## 🔥 Updates
**[2025/5]** **Code for benchmark and preprocessing is updated.**  
 For those who have downloaded or are preparing to download our dataset: you are recommended to star this repo in case the dataset is updated without notice.



## 🗝️ Access and Usage
**This dataset is built for academic use. Any commercial usage is banned.**  
There are two kinds of datasets for your convenience: 
Preprocessed (one-subject raw RGB and NIR data and all-subject preprocessed mmWave data ([link](https://www.kaggle.com/datasets/xiaoyang274/physdrive), no need for data share agreement); Raw (all-subject raw data, requires signing data share agreement, please contact Xiao Yang xyang856@connect.hkust-gz.edu.cn).    



## ⚙️ Experiment Procedure  
<img src='https://github.com/WJULYW/PhysDrive-Dataset/blob/main/figs/experiment_design.png' width = 80% height = 80% />

## 📊 Apparatus and Distribution
<img src='https://github.com/WJULYW/PhysDrive-Dataset/blob/main/figs/distribution.png' width = 80% height = 80% />


## 🖥️ The Dataset Structure
PhysDrive dataset (Preprocessed)
```
├── mmWave/
│ ├── AFH1/ # # The first character is composed of the letters A,B,C. A is Segment-A0, B is Segment-B, and C is Segment-C SUV; the second character is composed of the letters M, F. M stands for male, and F stands for female; the third letter is composed of the letters Z, H, W, Y. Z stands for Noon, H for Dusk & Early morning, W for Midnight, and Y for Rainy & Cloudy day.
│ │ ├── AFH1_00/
│ │ │ ├── resp.mat # Respiration signal
│ │ │ ├── mmwave.mat # Cropped mmWave radar signal (n_doppler, n_angle, n_range = 8, 16, 8)
│ │ │ └── ecg.mat # ECG signal
│ │ ├── …
│ │ └── AFH1_118/
│ └── CMZ2/
│
├── RGB and IR (one subject sample)/
│ ├── AMH1/
│ │ ├── AS/ # The first character is composed of the letters A,B,C, where "A" represents Flat&Unobstructed Road, "B" represents Flat&Congested Road, and "C" represents Bumpy & Congested Road; the second character indicates "Stationary" or "Talking".
│ │ ├── IR.mp4 # Infrared video
│ │ ├── RGB.mp4 # RGB video
│ │ ├── Recording_Physiological_Data.csv # Record all physiological data along with the corresponding timestamps.
│ │ ├── Label/
│ │ │ ├── HR.mat # filted Heart Rate
│ │ │ ├── BVP.mat # filted Blood Volume Pulse
│ │ │ ├── RESP.mat # filted Respiration signal
│ │ │ ├── ECG.mat # filted ECG signal
│ │ │ └── SPO2.mat # Blood oxygen saturation
│ │ └── STMap/
│ │ └── STMap_RGB.png # Spatial-temporal map extracted from RGB video
│ └── …
│
├── AT/
├── BS/
├── BT/
├── CS/
```


## 📄 Citation
Title: [PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring](https://arxiv.org/abs/2507.19172) 

Jiyao Wang, Xiao Yang, Qingyong Hu, Jiankai Tang, Can Liu, Dengbo He, Yuntao Wang, Yingcong Chen, Kaishun Wu, NIPS, 2025  
```
@article{wang2024efficient,
  title={Efficient mixture-of-expert for video-based driver state and physiological multi-task estimation in conditional autonomous driving},
  author={Wang, Jiyao and Yang, Xiao and Wang, Zhenyu and Wei, Ximeng and Wang, Ange and He, Dengbo and Wu, Kaishun},
  journal={arXiv preprint arXiv:2410.21086},
  year={2024}
}
```
