# Training Notes

## Docker file Instructions

- To build the docker file: `docker build -t skingpt4_llama2_image .`
- To run the docker file: `docker run -it --gpus all -v ~/Facial-Skincare-Analysis:/workspace/data skingpt4_llama2_image`
