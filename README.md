# semisup-seg-efficient

Code for reproducing results from "[A baseline for semi-supervised learning of efficient semantic segmentation models](https://arxiv.org/abs/2106.07075)".

This repository also contains slides, and some figures.

## Setup

### Dependencies

The code depends on [Vidlu](https://github.com/Ivan1248/Vidlu). If Vidlu is not installed, you might have to add the root directory of the repository to the `PYTHONPATH` environment variable. E.g. in Bash, this should run without error with "path/to/vidlu" substituted for the correct path:

```sh
PYTHONPATH="${PYTHONPATH}:path/to/vidlu" python -c "import vidlu.utils; print('success')"
```

### Directory structure

For the directory structure for datasets, results and other data, see the [directory configuration section](https://github.com/Ivan1248/vidlu#directory-configuration). Note that this repository uses its own copy of `dirs.py`.

### Datasets

CIFAR-10 will be downloaded automatically to the "datasets" directory. The directory for Cityscapes, "datasets/Cityscapes", has to be set up manually.

1. From <https://www.cityscapes-dataset.com/downloads/> download:
    - leftImg8bit_trainvaltest.zip (11GB)
    - gtFine_trainvaltest.zip (241MB)
2. Unzip both files directly into "datasets/Cityscapes" so that training images and labels are in "Cityscapes/leftImg8bit/train" and "Cityscapes/gtFine/train".

### Configuration check

This runs 3 experiment configurations for checking whether everything is set up properly:

```sh
bash test_setup.sh
```

Vidlu should be visible to Python through `PYTHONPATH`. The `CUDA_VISIBLE_DEVICES` environment variable can be used to choose a GPU.

## Experiments

Each table from the paper has its own script. You might want to comment out some parts or change the number of runs per experiment. 

Experiments on half-resolution Cityscapes with different proportions of labels (Table 1):

```sh
bash experiments_label_proportions.sh
```

Consistency variant comparison (Table 2):

```sh
bash experiments_cons_variants.sh
```

Qualitative results (Figure 1 in appendix):

```sh
bash generate_images.sh
```
