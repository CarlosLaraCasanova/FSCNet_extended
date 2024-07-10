# Cephalometric Landmarks Localization

This repository contains the original code for the paper:

**Cascade of convolutional models for few-shot automatic cephalometric landmarks localization**

Guillermo Gomez-Trenado, Pablo Mesejo, Oscar Cordon  
*Engineering Applications of Artificial Intelligence*, Volume 123, 2023, 106391

## Overview

This project implements a method for the automatic localization of cephalometric landmarks using a cascade of convolutional models. The approach is designed for few-shot learning scenarios, making it suitable for small datasets commonly encountered in forensic anthropology.

## Features

- Cascade of conditional convolutional networks
- High-resolution cephalometric landmark prediction
- Validated against expert annotators
- Significantly outperforms existing facial landmark localization methods

## Usage

See ```dltrain/arguments.py``` for descriptions on the expected input.

```bash
python train_forensic.py -d /path/to/images --gt /path/to/ground_truth.csv --tosave /path/to/save/models --eval_out /path/to/save/evaluation/results --cv /path/to/5-fold_indices.json --ver /path/to/3ddfa_output.npy -m /path/to/optimization_matrices.npy --model resnet18 --dataset all_augm
```

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{gomez2023cascade,
  title={Cascade of convolutional models for few-shot automatic cephalometric landmarks localization},
  author={Gomez-Trenado, Guillermo and Mesejo, Pablo and Cordon, Oscar},
  journal={Engineering Applications of Artificial Intelligence},
  volume={123},
  pages={106391},
  year={2023},
  publisher={Elsevier}
}
```