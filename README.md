# TBNet
This repo holds codes of the paper: TB-Net: Intra- and Inter-video Correlation Learning for Continuous Sign Language Recognition.

This repo is based on VAC (ICCV 2021)[[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) [[code]](https://github.com/ycmin95/VAC_CSLR) and SMKD (ICCV 2021)[[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) [[code]](https://github.com/ycmin95/VAC_CSLR). Many thanks for their great work!

Our training and inference procedure is modified from VAC(ICCV 2021) and SMKD (ICCV 2021).
## Requirements

- Python (>3.7).
  
- Pytorch (>1.8).

- ctcdecode

- sclite

- clip

Other configurations can refer to VAC [[code]](https://github.com/ycmin95/VAC_CSLR) and DilatedSLR [[code]](https://github.com/ustc-slr/DilatedSLR).

## Data Preparation
You can obtain the recognition results of TBNet based on the provided CKPT on the PHOENIX2014, PHOENIX2014-T, and CSLDaily datasets.

### PHOENIX2014 dataset
- RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).

- Run the following code to resize original image to 256x256px for augmentation.

   ```bash
   cd ./datapreprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
- RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

- The data preprocessing is the same as the RWTH-PHOENIX-Weather 2014 dataset.

### CSL-Daily dataset

- Request the CSL-Daily Dataset from this [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.html) [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)


- The data preprocessing is the same as the RWTH-PHOENIX-Weather 2014 dataset.
   
## Inference

### PHOENIX2014 dataset

| Backbone | Dev WER  | Test WER  | Inference model               |
| -------- | ---------- | ----------- | ------------------------- |
| Baseline | 20.6%      | 21.0%       | ------------------------- | 
|  ViT16   | 18.9%      | 19.6%       | [[Baidu]](https://pan.baidu.com/s/13AZ1qqcGZkJeO8hyBFsMcA) (passwd: 3ppm)<br /> |

### PHOENIX2014-T dataset

| Backbone | Dev WER  | Test WER  | Inference model               |
| -------- | ---------- | ----------- | ------------------------- |
| Baseline | 20.1%      | 21.8%       | ------------------------- |
|  ViT16   | 18.8%      | 20.0%       | [[Baidu]](https://pan.baidu.com/s/1asPbeAXnBsARevh6IbN65g) (passwd: chdq)<br /> |

### CSL-Daily dataset

| Backbone | Dev WER  | Test WER  | Inference model               |
| -------- | ---------- | ----------- | ------------------------- |
| Baseline | 30.3%      | 30.2%       | ------------------------- |
|  ViT16   | 28.4%      | 28.2%       | [[Baidu]](https://pan.baidu.com/s/18wJkCPv40w2HVKbh9seyLw) (passwd: ezl2)<br /> |

​	To evaluate the model, run the code below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

