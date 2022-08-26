
**Title: Image reconstruction with Transformer for mask-based lensless imaging**

**Authors: Xiuxi Pan, Xiao Chen, Saori Takeyama, Masahiro Yamaguchi**

**DOI: https://doi.org/10.1364/OL.455378**

```ol-47-7-1843.pdf``` is the paper.

**News Reports related to this paper (English & Japanese)**:
[WIRED](https://wired.jp/article/mask-based-lensless-imaging/),
[Nikkei](https://www.nikkei.com/article/DGXZQOUC12CUO0S2A510C2000000/),
[Phys.org](https://phys.org/news/2022-04-lensless-imaging-advanced-machine-image.html),
[EurekAlert!](https://www.eurekalert.org/news-releases/951125),
[Tokyo Tech News](https://www.titech.ac.jp/news/2022/063968), et al.


# USAGE
## Training
```datasets/prepare_datasets.py``` prepares .npy files of dataset address;

```configs.yaml``` defines training implementations;

```train.py``` stars training.

An example of running ```train.py``` in linux: 
```
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train.py &
```

## Prediction
```predict.py``` starts prediction.

Checkpoint (```checkpoints/best.pth```) and input patterns (```result/in-wild/pattern/``` and ```result/on-screen/pattern/```) can be used to verify our results.

## Note
```GrayPSF.npy``` is PSF of our lensless camera. It is not used in this reconstruction method, but a useful file to evaluate status of the optical system.
