# Training Notes

## Docker file Instructions

- To build the docker file: `docker build -t skingpt4_llama2_image .`
- To run the docker file: `docker run -it --gpus all -v ~/Facial-Skincare-Analysis:/root/Facial-Skincare-Analysis skingpt4_llama2_image`

## Dataset Instructions

1. Fitzpatrick17k dataset:
    - `wget https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/refs/heads/main/fitzpatrick17k.csv`

Alternatively, run setup_dataset.sh to download all the datasets.

```bash
chmod +x setup_dataset.sh && ./setup_dataset.sh
```
