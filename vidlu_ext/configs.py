from functools import partial

import torch
from torch import optim

from vidlu.transforms import jitter
from vidlu.modules import losses
from vidlu.optim.lr_schedulers import CosineLR
import vidlu.optim as vo
from vidlu.training import steps
import vidlu.training.extensions as te
from vidlu.training.robustness import attacks
from vidlu.training.robustness import perturbation as pert
from vidlu.configs.training import TrainerConfig, OptimizerMaker
import vidlu.modules.inputwise as vmi

# Robustness

phtps_attack_20 = partial(
    attacks.PertModelAttack,
    pert_model_f=partial(pert.PhotoTPS20, clamp=False, forward_arg_count=3),
    initializer=pert.MultiInit(
        tps=pert.NormalInit({'offsets': (0, 0.1)}),
        photometric=pert.UniformInit(
            {'add_v.addend': [-0.25, 0.25],
             'mul_s.factor': [0.25, 2.],
             'add_h.addend': [-0.1, 0.1],
             'mul_v.factor': [0.25, 2.]})),
    projection=None,
    optim_f=partial(vo.ProcessedGradientDescent, process_grad=torch.sign),
    step_size=0.01,  # 0.01 the image height/width
    step_count=0,
)

cutmix_attack_21 = partial(
    attacks.PertModelAttack,
    pert_model_f=partial(vmi.CutMix, mask_gen=jitter.BoxMaskGenerator(prop_range=0.5),
                         combination='pairs'),
    initializer=lambda *a: None,
    projection=None,
    optim_f=partial(vo.ProcessedGradientDescent, process_grad=torch.sign),
    step_size=None,
)

# Training

supervised = TrainerConfig(
    eval_step=steps.supervised_eval_step,
    train_step=steps.supervised_train_step,
)

classification = TrainerConfig(
    supervised,
    loss=losses.nll_loss_l
)

# https://github.com/orsic/swiftnet/blob/master/configs/rn18_single_scale.py
swiftnet_cityscapes = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params='backbone.backbone', lr=1e-4, weight_decay=2.5e-5)],
                               lr=4e-4, betas=(0.9, 0.99), weight_decay=1e-4),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=14,
    eval_batch_size=4,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                           scale_dist='log-uniform'),
)

deeplabv2_cityscapes = TrainerConfig(
    classification,
    optimizer_f=OptimizerMaker(optim.Adam,
                               [dict(params='backbone.aspp', lr=4e-4, weight_decay=1e-4)],
                               lr=1e-4, betas=(0.9, 0.99), weight_decay=2.5e-5),
    lr_scheduler_f=partial(CosineLR, eta_min=1e-6),
    epoch_count=250,
    batch_size=4,
    eval_batch_size=4,  # 6
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(768, 768), max_scale=2, overflow=0,
                                           scale_dist='log-uniform'),
)

swiftnet_cityscapes_halfres = TrainerConfig(
    swiftnet_cityscapes,
    batch_size=8,
    jitter=jitter.SegRandScaleCropPadHFlip(shape=(448, 448), max_scale=1.5, overflow=0,
                                           scale_dist="log-uniform"))

deeplabv2_cityscapes_halfres = TrainerConfig(
    swiftnet_cityscapes_halfres,
    optimizer_f=deeplabv2_cityscapes.optimizer_f,
)

semisup_cons_phtps20 = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(phtps_attack_20, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=steps.SemisupVATStep(),
    eval_step=steps.SemisupVATEvalStep(),
)

semisup_cons_cutmix = TrainerConfig(
    te.SemisupVAT,
    attack_f=partial(cutmix_attack_21, step_count=0, loss=losses.kl_div_ll,
                     output_to_target=lambda x: x),
    train_step=steps.SemisupVATStep(),
    eval_step=steps.SemisupVATEvalStep(),
)
