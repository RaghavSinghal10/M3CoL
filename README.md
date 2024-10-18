# Harnessing Shared Relations via Multimodal Mixup Contrastive Learning for Multimodal Classification


## Environment
Python=3.8, Pytorch=1.8.0, CUDA=11.1
```
conda create -n mmc-m3co python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Data Preparation
[UPMC-Food-101](https://visiir.isir.upmc.fr/explore) is a multimodal food classification dataset. We adopt the most commonly used split method and remove those image-text pairs with missing images or text. The final dataset split is available [here](https://drive.google.com/drive/folders/11U1pjjQ5z6NaG9Gojo6QrSbIqEMYft7m?usp=share_link).

[N24News](https://github.com/billywzh717/n24news) is a multimodal news classification dataset. We adopt the original split method.

ROSMAP, BRCA can be downloaded from this [link](https://github.com/txWang/MOGONET/).

Add the datasets to a directory, and use this directory path for training.

## Training

The follwing commands need to run to train the models:

N24News:
train_image_text.py --dataset=n24news --multi_mixup --text_encoder=roberta_base --text_type=caption --data_dir=/your/data/directory/path

Change the text_type and text_encoder for various text types and encoders.

Food101:
train_image_text.py --dataset=Food101 --multi_mixup --text_encoder=bert_base --data_dir=/your/data/directory/path

ROSMAP:
train_medical --dataset=rosmap --data_dir=/your/data/directory/path

BRCA:
train_medical --dataset=brca --data_dir=/your/data/directory/path