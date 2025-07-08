# Baselines


## FIDNet
FIDNet (Fully Interpolation Decoding Network) is designed to perform semantic segmentation on LiDAR point clouds by projecting them into 2D spherical range images. This transformation allows the use of conventional 2D convolutional neural networks for processing.

### Train Details ###
```
Pretrained: True
Epochs: 30
Batch Size: 2
Learning Rate: 0.001
Loss: Cross Entropy + Tversky
```
### Publication ###
```
@article{zhao2021fidnet,
  title={FIDNet: LiDAR Point Cloud Semantic Segmentation with Fully Interpolation Decoding},
  author={Zhao, Yiming and Bai, Lin and Huang, Xinming},
  journal={arXiv preprint arXiv:2109.03787},
  year={2021}
}
```
### Acknowledgments ###
Code framework derived from [FIDNet](https://github.com/placeforyiming/IROS21-FIDNet-SemanticKITTI)

### Licence ###
The FIDNet repo does not provide a Licence File. Contantact the maintainer of the FIDNet repo if you have any questions.

## CENet
CENet: Toward Concise and Efficient LiDAR Semantic Segmentation for Autonomous Driving

### Train Details ###
```
Pretrained: True
Epochs: 10
Batch Size: 2
Learning Rate: 0001
Loss: Cross Entropy + Tversky (Plan B, Multi-Scale Loss, see CENet-Paper)
```

### Publication ###
```
@inproceedings{cheng2022cenet,
  title={Cenet: Toward Concise and Efficient Lidar Semantic Segmentation for Autonomous Driving},
  author={Cheng, Hui--Xian and Han, Xian--Feng and Xiao, Guo--Qiang},
  booktitle={2022 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={01--06},
  year={2022},
  organization={IEEE}
}
```
### Acknowledgments ###
Code framework derived from [CENet](https://github.com/huixiancheng/CENet)

### Licence ###
MIT License

## SalsaNext
SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving

### Train Details ###
```
Pretrained: True
Epochs: 30
Batch Size: 2
Learning Rate: 0001
Loss: Cross Entropy + Tversky 
```

### Publication ###
```
@misc{cortinhal2020salsanext,
    title={SalsaNext: Fast, Uncertainty-aware Semantic Segmentation of LiDAR Point Clouds for Autonomous Driving},
    author={Tiago Cortinhal and George Tzelepis and Eren Erdal Aksoy},
    year={2020},
    eprint={2003.03653},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
### Acknowledgments ###
Code framework derived from [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext)

### Licence ###
MIT License
