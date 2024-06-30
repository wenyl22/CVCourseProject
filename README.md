# CVCourseProject

## Project Description

This project is a part of the Computer Vision course at the University of Tsinghua. The goal of the project is to revive the color of ancient Chinese paintings.

## Samples

The samples mentioned in the report are stored in `./sample/` folder.

## Data
We store training data in the following structure, where A is Chinese painting and B is real world image. Test data are stored in similar structure, but in `test_A` and `test_B` folders. Dataprocessing is done in `./dataset/` folder with `canny.py` and `blur.py` files. 

    CVCourseProject/
    │
    ├── dataset/
    │   ├── datasetname/
    │       ├── train_A/
    │       ├── train_A_blur/
    │       ├── train_A_canny/
    │       ├── train_B/
    │       ├── train_B_blur/
    │       └── train_B_canny/

## Model

The main cycleGAN model is stored in `./models/` folder. SuperResolution model is stored in `./sr/` folder.

## Training

Training is done in `./train.py` file. 

## Testing

Testing is done in `./test.py` file.