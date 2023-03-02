# semisup-seg-efficient

Code for reproducing results from the paper "[A baseline for semi-supervised learning of efficient semantic segmentation models](https://arxiv.org/abs/2106.07075)".

There are also slides and some figures in "data".

**To do**: I will update the code to include additional experiments and figures from "[Revisiting consistency for semi-supervised semantic segmentation
](https://arxiv.org/abs/2106.07075)".


## Setup

Download the repository with
```sh
git clone https://github.com/Ivan1248/semisup-seg-efficient
```

### Dependencies

The code depends on [Vidlu v0.1.0](https://github.com/Ivan1248/vidlu/releases/tag/v0.1.0), which can be downloaded with
```sh
git clone https://github.com/Ivan1248/vidlu.git --branch v0.1.0
```
or installed with
```sh
pip install git+https://github.com/Ivan1248/vidlu@v0.1.0
```
Note that you have to use the version tagged "v0.1.0" for this code to work. 

If you have only downloaded Vidlu, you might have to add its root directory (the one that contains `vidlu`, `scripts`, `README.md`, ...) to the `PYTHONPATH` environment variable. E.g. in Bash, this should run without error with "`PYTHONPATH="${PYTHONPATH}:path/to/vidlu-repo`" substituted for the root path of the Vidlu repository:

```sh
PYTHONPATH="${PYTHONPATH}:path/to/vidlu-repo" python -c "import vidlu.utils; print('success')"
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
cd semisup-seg-efficient
bash test_setup.sh
```

The `CUDA_VISIBLE_DEVICES` environment variable can be used to choose a GPU and, if necessary, `PYTHONPATH` should be like in the [Dependencies](#dependencies) section.  

`vidlu_ext` is a [Vidlu extension](https://github.com/Ivan1248/Vidlu#extensions). It must be in the working directory or findable via `PYTHONPATH` so that Vidlu can find and load it. Everything should work if you run scripts from the root of this repository.

The command
```
python -c "import vidlu.factories; print(vidlu.factories.extensions)"
```
should print something like
```
ExtensionDict(ext=<module 'vidlu_ext' from '/path/to/semisup-seg-efficient/vidlu_ext/__init__.py'>)
```

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

## The paper

Conference version: http://www.mva-org.jp/Proceedings/2021/papers/P2-3.pdf  
Updated version on ArXiv: https://arxiv.org/abs/2106.07075

BibTeX:
```
@inproceedings{grubisic2021mva,
  author    = {Ivan Grubi\v{s}i\'{c} and
               Marin Or\v{s}i\'{c} and
               Sini\v{s}a \v{S}egvi\'{c}},
  title     = {A baseline for semi-supervised learning of efficient semantic segmentation models},
  booktitle = {17th International Conference on Machine Vision and Applications, {MVA} 2021, Aichi, Japan, July 25-27, 2021},
  pages     = {1--5},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.07075},
  doi       = {10.23919/MVA51890.2021.9511402},
}
```
