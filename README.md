# TB_Net
This repo holds codes of the paper: TB-Net: Intra- and Inter-video Correlation Learning for Continuous Sign Language Recognition.

This repo is based on VAC (ICCV 2021)[[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) [[code]](https://github.com/ycmin95/VAC_CSLR) and SMKD (ICCV 2021)[[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) [[code]](https://github.com/ycmin95/VAC_CSLR). Many thanks for their great work!

Our training and inference procedure is modified from VAC(ICCV 2021) and SMKD (ICCV 2021). If you are familiar with VAC, you can play with TBNet easily!
## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  ```bash
  mkdir ./software
  ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
  ```

- [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) for ctc supervision.

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of TBNet.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-T.py --process-image --multiprocessing
   ```

### CSL dataset
The results of TBNet on CSL dataset is placed in the supplementary material.

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL.py --process-image --multiprocessing
   ``` 

### CSL-Daily dataset

1. Request the CSL-Daily Dataset from this website [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL-Daily`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL-Daily.py --process-image --multiprocessing
   ``` 
   
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
|  ViT16   | 28.4%      | 28.2%       | [[Baidu]](链接：https://pan.baidu.com/s/18wJkCPv40w2HVKbh9seyLw) (passwd: ezl2)<br /> |

​	To evaluate the pretrained model, run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

