# Med-DANet（ECCV 2022）and Med-DANet V2（WACV 2024）

This repo is the official implementation for: 
[Med-DANet: Dynamic Architecture Network for Efficient Medical Volumetric Segmentation](https://arxiv.org/abs/2206.06575) and [Med-DANet V2: A Flexible Dynamic Architecture for Efficient Medical Volumetric Segmentation](https://arxiv.org/abs/2310.18656).

## Requirements
- python 3.7
- pytorch 1.6.0
- torchvision 0.7.0
- pickle
- nibabel
- fvcore
- kornia
- setproctitle
- tensorboardX
- pickle

## Data Acquisition
- The multimodal brain tumor datasets (**BraTS 2019** & **BraTS 2020**) could be acquired from [here](https://ipp.cbica.upenn.edu/).

- The liver tumor dataset **LiTS 2017** could be acquired from [here](https://competitions.codalab.org/competitions/17094#participate-get-data).

## Data Preprocess (BraTS 2019 & BraTS 2020)
After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

`python3 ./data/preprocess.py`

## Training
Run the training script on BraTS dataset. Distributed training is available for training the proposed TransBTS.

`sh train.sh`

## Testing 
If  you want to test the model which has been trained on the BraTS dataset, run the testing script as following.

`python3 test.py`

After the testing process stops, you can upload the submission file to [here](https://ipp.cbica.upenn.edu/) for the final Dice_scores.

## Citation
If you use our code or models in your work or find it is helpful, please cite the corresponding paper:

- **Med-DANet**:
```
@inproceedings{wang2022med,
  title={Med-DANet: Dynamic Architecture Network for Efficient Medical Volumetric Segmentation},
  author={Wang, Wenxuan and Chen, Chen and Wang, Jing and Zha, Sen and Zhang, Yan and Li, Jiangyun},
  booktitle={European Conference on Computer Vision},
  pages={506--522},
  year={2022},
  organization={Springer}
}
```
- **Med-DANet V2**:
```
@inproceedings{shen2024med,
  title={Med-DANet V2: A Flexible Dynamic Architecture for Efficient Medical Volumetric Segmentation},
  author={Shen, Haoran and Zhang, Yifu and Wang, Wenxuan and Chen, Chen and Liu, Jing and Song, Shanshan and Li, Jiangyun},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7871--7881},
  year={2024}
}
```
