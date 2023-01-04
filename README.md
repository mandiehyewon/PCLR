# PCLR (Patient Contrastive Learning of Representations)

This repository is for the finetuning of PCLR pretrained model; as the model has built using Keras, so this repository uses Keras to finetune the model. 
PCLR is a pre-training strategy that yields a neural network that extracts representations of ECGs.
The representations are designed to be used in linear models without finetuning the network.
This readme shows how to load a model trained on over three million ECGs using PCLR.

## Requirements
This code was tested using python 3.7.
It can be used using virtual env.
```bash
python3.7 -m venv env
source env/bin/activate
pip install -r requirements.txt
python -i get_representations.py  # test the setup worked
>>> test_get_representations()
```

## Usage
### Getting ECG representations
You can get ECG representations using [get_representations.py](./get_representations.py).
`get_representations.get_representations` builds `N x 320` ECG representations from `N` ECGs.

The model expects 10s 12-lead ECGs with a specific lead order and interpolated to be 4,096 samples long.
[preprocess_ecg.py](./preprocess_ecg.py) shows how to do the pre-processing.

### Building un-trained PCLR and comparison models

You can get compiled, but un-trained models with the hyperparameters selected in our training set.
`python -i build_model.py`
```python
pclr_model = PCLR_model()
clocs_model = CLOCS_model()
CAE = CAE_model()
ribeiro_r = ribeiro_r_model()
```
`build_model.py` uses code from [the google research implementation of SimCLR](https://github.com/google-research/simclr/)
and [the official implementation](https://github.com/antonior92/automatic-ecg-diagnosis) of "Automatic diagnosis of the 12-lead ECG using a deep neural network",
Ribeiro et al 2020.

## PCLR model weight
PCLR model trained with three million ECGs [PCLR.h5](./PCLR.h5) and three million ECGs without [PCLR_wo_apollo.h5](./PCLR_wo_apollo.h5).