# Building Representation and Finetuning with PCLR (Patient Contrastive Learning of Representations)

**Author:** Hyewon Jeong

**Last Edited:** Jan 20, 2022

This repository is for the finetuning of PCLR pretrained model; as the model has built using Keras, so this repository uses Keras to finetune the model. 
PCLR is a pre-training strategy that yields a neural network that extracts representations of ECGs.
The representations are designed to be used in linear models without finetuning the network.
This readme shows how to load a model trained on over three million ECGs using PCLR, which has retrieved from the [official PCLR github repository](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/PCLR)

## Requirements
This code was tested using python 3.7.
It can be used using virtual env or conda environment.

### Virtual Environment:
```bash
python3.7 -m venv env
source env/bin/activate
pip install -r requirements.txt
python -i get_representations.py  # test the setup worked
>>> test_get_representations()
```

### Conda Environment:
```bash
conda env create -f env.yml
conda activate pclr
```

## Usage
### Getting ECG representations
You can get ECG representations using [get_representations.py](./src/get_representations.py).
`get_representations.get_representations` builds `N x 320` ECG representations from `N` ECGs.

The model expects 10s 12-lead ECGs with a specific lead order and interpolated to be 4,096 samples long.
[preprocess_ecg.py](./src/preprocess_ecg.py) shows how to do the pre-processing.

### Building un-trained PCLR and comparison models

You can get compiled, but un-trained models with the hyperparameters selected in our training set.
`python -i build_model.py`
```python
pclr_model = PCLR_model()
clocs_model = CLOCS_model()
CAE = CAE_model()
ribeiro_r = ribeiro_r_model()
```
[build_model.py](./src/build_model.py) uses code from [the google research implementation of SimCLR](https://github.com/google-research/simclr/)
and [the official implementation](https://github.com/antonior92/automatic-ecg-diagnosis) of "Automatic diagnosis of the 12-lead ECG using a deep neural network",
Ribeiro et al 2020.

## PCLR model weight
PCLR model trained with three million ECGs [PCLR.h5](./PCLR.h5) and three million ECGs not including the Apollo dataset [PCLR_wo_apollo.h5](./PCLR_wo_apollo.h5).

## Finetuning PCLR model
Please refer to [finetune_pclr_regression.ipynb](./notebook/finetune_pclr_regression.ipynb) or [finetune_pclr_classification.ipynb](./notebook/finetune_pclr_classification.ipynb) for finetuning code.

## Citing this Work
This repository is used to get the PCLR representations of ECGs as part of [Deep Metric Learning for the Hemodynamics Inference with Electrocardiogram Signals](https://proceedings.mlr.press/v219/jeong23a/jeong23a.pdf). When using the code, please cite the paper and the [Gituhub](https://github.com/mandiehyewon/ssldml), as well as the original PCLR paper. bibtex:

```
@article{jeong2023deep,
  title={Deep Metric Learning for the Hemodynamics Inference with Electrocardiogram Signals},
  author={Jeong, Hyewon and Stultz, Collin M and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:2308.04650},
  year={2023}
}

@article{diamant2022patient,
  title={Patient contrastive learning: A performant, expressive, and practical approach to electrocardiogram modeling},
  author={Diamant, Nathaniel and Reinertsen, Erik and Song, Steven and Aguirre, Aaron D and Stultz, Collin M and Batra, Puneet},
  journal={PLoS computational biology},
  volume={18},
  number={2},
  pages={e1009862},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
