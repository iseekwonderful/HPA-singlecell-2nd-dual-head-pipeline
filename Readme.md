## I. HARDWARE:
### Train:
* Ubuntu 20.04.1 LTS (~200GB free disk space)
* CPU: 8C
* RAM: 64GB
* 1 x RTX 3090

### Predict
* Please refer to kaggle notebook [sub1](https://www.kaggle.com/steamedsheep/hpa-final-submission-2-candidate-ii) and [sub2](https://www.kaggle.com/steamedsheep/draft-of-submission-of-dakiro-model-sunzi-w0-5?scriptVersionId=62584023)

## II. SOFTWARE
* Please refer to `Dockerfile`
* Directly run container: `docker pull steamedsheep/hpa_pipeline1:v1.15`.

## III. Data Setup

1. Make fold input and results at same dir of the repo.

2. Download pregenerated cells from kaggle notebook and decompress to `input/train_cell_256`, after decompression, there will be
about 70GB `png` file in `input/train_cell_256`, since It's diffcult to use Kaggle API to download notebook output, Please download `result.zip` of the notebook below:
* [train cell #1](https://www.kaggle.com/steamedsheep/hpa-cellslicing-256-fix-i)
* [train_cell_#2](https://www.kaggle.com/steamedsheep/hpa-cellslicing-ii-fix-256)
* [Phil upload images #1](https://www.kaggle.com/steamedsheep/hpa-cellslicing-256-fix-external-i)
* [Phil upload images #2](https://www.kaggle.com/steamedsheep/hpa-cellslicing-256-fix-external-ii)
* [Phil upload images #3](https://www.kaggle.com/steamedsheep/hpa-cellslicing-256-fix-external-iii)
* [ext not in Phil but rare](https://www.kaggle.com/steamedsheep/hpa-cellslicing-256-fix-external-rare-i)

## IV. Train and predict.
1. Start docker at solution dir `sudo docker run -v $PWD:/workspace -it --gpus '"device=0"' steamedsheep/hpa_pipeline1:v1.2`
2. `/bin/bash train.sh`

### Full training
* run `train.sh` to get the result