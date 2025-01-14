# Training Notes

## Docker file Instructions

- [CPU] To build the docker file: `docker build -f Dockerfile.cpu -t fsahack:cpu .`
- [GPU] To build the docker file: `docker build -f Dockerfile.gpu -t fsahack:gpu .`
- [CPU] To run the docker file: `docker run -it -v ~/Facial-Skincare-Analysis:/root/Facial-Skincare-Analysis fsahack:cpu`
- [GPU] To run the docker file: `docker run -it --gpus all --shm-size=4g -v ~/Facial-Skincare-Analysis:/root/Facial-Skincare-Analysis fsahack:gpu`

## Dataset

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1ysutxP3IHKhQA8brdwPlhwbduo-kYt1x
tar -xvf Classification.tar
rm Classification.tar
```


## Training instructions

1. Load dataset: `python prep_dataset.py`
2. Train model: `python train.py --epochs 20 --batch_size 16`
