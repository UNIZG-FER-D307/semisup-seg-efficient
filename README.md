# semisup-seg-efficient

Code for reproducing results from the paper "[A baseline for semi-supervised learning of efficient semantic segmentation models](https://arxiv.org/abs/2106.07075)".

There are also slides and some figures in "data".

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

1. From <https://www.cityscapes-dataset.com/downloads/> download `leftImg8bit_trainvaltest.zip` (11GiB) and `gtFine_trainvaltest.zip` (241MiB).
2. Extract everything directly into "datasets/Cityscapes". Directories with training images and labels should be at "Cityscapes/leftImg8bit/train" and "Cityscapes/gtFine/train".

### Configuration check

This runs 3 experiment configurations for checking whether everything is set up properly:

```sh
bash test_setup.sh
```

The `CUDA_VISIBLE_DEVICES` environment variable can be used to choose a GPU and, if necessary, `PYTHONPATH` should be like in the [Dependencies](#dependencies) section.  

`vidlu_ext` is a [Vidlu extension](https://github.com/Ivan1248/Vidlu#extensions). It must be in the working directory or findable via `PYTHONPATH` so that Vidlu can find and load it. Everything should work if you run scripts from the root of this repository.

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
