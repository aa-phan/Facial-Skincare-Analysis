# Training Notes

## Docker file Instructions

- [CPU] To build the docker file: `docker build -f Dockerfile.cpu -t fsahack:cpu .`
- [GPU] To build the docker file: `docker build -f Dockerfile.gpu -t fsahack:gpu .`
- [CPU] To run the docker file: `docker run -it -v path_to_directory:/root/Facial-Skincare-Analysis -p 5000:5000 fsahack:cpu`
- [GPU] To run the docker file: `docker run -it --gpus all --shm-size=4g -v ~/Facial-Skincare-Analysis:/root/Facial-Skincare-Analysis -p 5000:5000 fsahack:gpu`

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

## Inferance instructions

1. Download checkpoint@ [drivelink](https://drive.google.com/file/d/1wA_-I52FVhspx_bveAjUEAMfj6xTR1hI/view?usp=sharing) and save it in the `checkpoints` folder.
2. Load the server: `python endpoint.py`
3. Make a POST request to the server: `curl.exe -X POST -F "image=@image.jpg" http://localhost:5000/predict`
