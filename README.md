# DeLo
This is the official code for paper:" DeLo: Dual Decomposed Low-Rank Expert Collaboration for Continual Missing Modality Learning"

## Installing Dependencies

We tested our code on Ubuntu 22.04 with PyTorch 1.13. You can use `environment.yml` and `requirements.txt` to install dependencies.

## Run

bash scripts/food101_both_0.7.sh

## Data Preparation

Download `UPMC-Food101` and `MM-IMDb` datasets according to the [MAP](https://github.com/YiLunLee/missing_aware_prompts) repo and organize them as following:

```text
data
├── MM-IMDB-CMML
│   ├── images
│   ├── labels
│   └── MM-IMDB-CMML.json
└── UPMC-Food101-CMML
    ├── images
    ├── texts
    └── UPMC-Food101-CMML.json

