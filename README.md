# PhysDrive Dataset

## 📖 Abstract

**Here is PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring, collected by The Hong Kong University of Science and Technology (Guangzhou).**  
The PhysDrive, comprises about 24 hours(**1500K frames**) of recordings from RGB camera, NIR camera, and an mmWave radar of 48 subjects. The dataset was designed to ontactless in-vehicle physiological sensing with dedicated consideration on various modality settings and driving factors, including **Three vehicle types, four illumination conditions, three road conditions, and driver motions**. PhysDrive is comprehensive with six synchronized ground truths (**ECG, BVP, Respiration, HR, RR, and SpO2**) and can be used in conjunction with the [rPPG-toolbox](https://github.com/ubicomplab/rPPG-Toolbox).  

## 🔥 Updates
```
**[2024/9]** **Opensource Benchmark Open. Please make a PR with your result and code.**  
**[2024/8]** **Contact Email is updated. Please contact ```tjk24@mails.tsinghua.edu.cn``` for the application.**   
**[2024/1]** **Citation BibTex and Data Release Agreement are updated.**  
**[2023/11]** **Subset of two participants' data available for educational purposes, subject to their consent. University faculty may apply for access to this subset via email.**  
**[2023/6]** **size.csv file is updated, allowing users to check data integrity.**
```
**[2025/5]** **Code for benchmark and preprocessing is updated.**  
 For those who have downloaded or are preparing to download our dataset: you are recommended to star this repo in case the dataset is updated without notice.


```
//## 🔍 Samples
//|                           |LED-low|LED-high|Incandescent|Nature|
//|:-------------------------:|:-----:|:------:|:----------:|:----:|
//|Skin Tone 3<br />Stationary|![](gif/LED-low_S.gif)|![](gif/LED-high_S.gif)|![](gif/Incandescent_S.gif)|![](gif/Nature_S.gif)|
//|Skin Tone 4<br />Rotation  |![](gif/LED-low_R.gif)|![](gif/LED-high_R.gif)|![](gif/Incandescent_R.gif)|![](gif/Nature_R.gif)|
//|Skin Tone 5<br />Talking   |![](gif/LED-low_T.gif)|![](gif/LED-high_T.gif)|![](gif/Incandescent_T.gif)|![](gif/Nature_T.gif)|
//|Skin Tone 6<br />Walking   |![](gif/LED-low_W.gif)|![](gif/LED-high_W.gif)|![](gif/Incandescent_W.gif)|![](gif/Nature_W.gif)|
```

## 🗝️ Access and Usage
**This dataset is built for academic use. Any commercial usage is banned.**  
There are two kinds of datasets for your convenience: 
Preprocessed (one-subject raw RGB and NIR data and all-subject preprocessed mmWave data ([link](https://www.kaggle.com/datasets/xiaoyang274/physdrive), no need for data share agreement); Raw (all raw dataset(requires signing data share agreement, please contact jwanggo@connect.ust.hk).    
```
\\There are two ways for downloads： OneDrive and Baidu Netdisk for researchers from different regions.  For those researchers in China, a hard disk could also be a solution.
\\To access the dataset, you are supposed to download this [data release agreement](https://github.com/McJackTang/MMPD_rPPG_dataset/blob/main/MMPD_Release_Agreement.pdf).  
\\Please scan and dispatch the completed agreement via your institutional email to <tjk24@mails.tsinghua.edu.cn> and cc <yuntaowang@tsinghua.edu.cn>. The email should have the subject line 'MMPD Access Request -  your institution.' In the email,  outline your institution's website and publications for seeking access to the MMPD, including its intended application in your specific research project. The email should be sent by a faculty rather than a student.
```


## ⚙️ Experiment Procedure[Updated]  
<img src='https://github.com/WJULYW/PhysDrive-Dataset/blob/main/figs/experiment_design.pdf' width = 50% height = 50% />

## 📊 Distribution
<img src='https://github.com/WJULYW/PhysDrive-Dataset/blob/main/figs/experiment_design.pdf' width = 50% height = 50% />


## 🖥️ The Dataset Structure
PhysDrive dataset (Preprocessed)
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
## Benchmark for Open-source Models
We noticed that plenty of new papers are emerging that use MMPD as a test dataset. We are encouraging the community to open-source related codes of their methods and pull a request in the following. We will keep this updated and other researchers could use those indexes as benchmark results. If you only use part of the MMPD dataset, you should put the details on note.
| Year | METHODS      |Training Dataset | MAE  | RMSE  | MAPE  | PEARSON | Paper Link  | Code Link | Reproducibility| Note|
|-------|--------------|--------------|------|-------|-------|---------|-------|---------|---------|---------|
|2023|PhysNet|UBFC| 10.24 | 16.54|12.46 | 0.29  |[Paper](https://doi.org/10.48550/arXiv.2302.03840)|[Code](https://github.com/McJackTang/MMPD_rPPG_dataset/)|100+ Star|Official|
## 📝 Results from the original paper(tested on MMPD)
### Simplest scenario
In the simplest scenario, we only include the stationary, skin tone type 3, and artificial light conditions as benchmarks.
| METHODS      | MAE  | RMSE  | MAPE  | PEARSON |
|--------------|------|-------|-------|---------|
| ICA          | 8.75 | 12.35 | 12.26 | 0.21    |
| POS          | 7.69 | 11.95 | 11.45 | 0.19    |
| CHROME       | 8.81 | 13.18 | 12.95 | -0.03   |
| GREEN        | 10.57| 15.03 | 14.59 | 0.23    |
| LGI          | 7.46 | 11.92 | 10.12 | 0.12    |
| PBV          | 8.15 | 11.52 | 11.04 | 0.35    |
| TS-CAN(trained on PURE) | 1.78 |3.57 | 2.47 |0.93  |
| TS-CAN(trained on UBFC) | **1.46** | **3.13**  | **2.04**  | **0.94**   |

### Unsupervised Signal Processing Methods（Subset）

We evaluated six traditional unsupervised methods in our dataset. In the skin tone comparison, we excluded the exercise, natural light, and walking conditions to eliminate any confounding factors and concentrate on the task. Similarly, the motion comparison experiments excluded the exercise and natural light conditions, while the light comparison experiments excluded the exercise and walking conditions. This approach enabled us to exclude cofounding factors and better understand the unique challenges posed by each task.

<img src='https://github.com/McJackTang/Markdown_images/blob/main/unsupervised.jpg' width = 70% height = 70%/>

### Supervised Deep Learning Methods（Subset）
In this paper, we investigated how state-of-the-art supervised neural networks perform on MMPD and studied the influence of skin tone, motion, and light. We used the same exclusion criteria as the evaluation on unsupervised methods.

<img src='https://github.com/McJackTang/Markdown_images/blob/main/supervised.jpg' width = 70% height = 70% />

### Full Dataset Result
For the full dataset, no existing methods could accurately predict the PPG wave and heart rate.  We are looking forward to algorithms that could be applied to daily scenarios. Researchers are encouraged to report their results and communicate with us.

  | METHODS      | MAE   | RMSE  | MAPE  | PEARSON |
|--------------|-------|-------|-------|---------|
| ICA          | 18.57 | 24.28 | 20.85 | 0.00    |
| POS          | 12.34 | 17.70 | 14.43 | 0.17    |
| CHROME       | 13.63 | 18.75 | 15.96 | 0.08    |
| GREEN        | 21.73 | 27.72 | 24.44 | -0.02   |
| LGI          | 17.02 | 23.28 | 18.92 | 0.04    |
| PBV          | 17.88 | 23.53 | 20.11 | 0.09    |

| METHODS(trained on PURE) | MAE   | RMSE  | MAPE  | PEARSON |
|--------------------------|-------|-------|-------|---------|
| TS-CAN                   | 13.94 | 21.61 | 15.14 | 0.20    |
| DeepPhys                 | 16.92 | 24.61 | 18.54 | 0.05    |
| EfficientPhys            | 14.03 | 21.62 | 15.32 | 0.17    |
| PhysNet                  | 13.22 | 19.61 | 14.73 | 0.23    |

| METHODS(trained on UBFC) | MAE   | RMSE  | MAPE  | PEARSON |
|--------------------------|-------|-------|-------|---------|
| TS-CAN                   | 14.01 | 21.04 | 15.48 | 0.24    |      
| DeepPhys                 | 17.50 | 25.00 | 19.27 | 0.05    |
| EfficientPhys            | 13.78 | 22.25 | 15.15 | 0.09    |
| PhysNet                  | **10.24** | **16.54** | **12.46** | **0.29**    |

| METHODS(trained on SCAMPS) | MAE   | RMSE  | MAPE  | PEARSON |
|----------------------------|-------|-------|-------|---------|
| TS-CAN                     | 19.05 | 24.20 | 21.77 | 0.14    |      
| DeepPhys                   | 15.22 | 23.17 | 16.56 | 0.09    |
| EfficientPhys              | 20.37 | 25.04 | 23.48 | 0.11    |
| PhysNet                    | 21.03 | 25.35 | 24.68 | 0.14    |

## 📄 Citation
Title: [MMPD: Multi-Domain Mobile Video Physiology Dataset](https://doi.org/10.48550/arXiv.2302.03840)  
Jiankai Tang, Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, Xin Liu, "MMPD: Multi-Domain Mobile Video Physiology Dataset", IEEE EMBC, 2023  

@INPROCEEDINGS{10340857,
  author={Tang, Jiankai and Chen, Kequan and Wang, Yuntao and Shi, Yuanchun and Patel, Shwetak and McDuff, Daniel and Liu, Xin},
  booktitle={2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 
  title={MMPD: Multi-Domain Mobile Video Physiology Dataset}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/EMBC40787.2023.10340857}}
```
