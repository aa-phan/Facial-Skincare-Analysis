# Training Notes

## Docker file Instructions

- [CPU] To build the docker file: `docker build -f Dockerfile.cpu -t fsahack:cpu .`
- [GPU] To build the docker file: `docker build -f Dockerfile.gpu -t fsahack:gpu .`
- [CPU] To run the docker file: `docker run -it -v ~/Facial-Skincare-Analysis:/root/Facial-Skincare-Analysis fsahack:cpu`
- [GPU] To run the docker file: `docker run -it --gpus all -v ~/Facial-Skincare-Analysis:/root/Facial-Skincare-Analysis fsahack:gpu`

## Dataset Instructions

1. Fitzpatrick17k dataset:
    - `wget https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/refs/heads/main/fitzpatrick17k.csv`

Alternatively, run setup_dataset.sh to download all the datasets.

```bash
chmod +x setup_dataset.sh && ./setup_dataset.sh
```
