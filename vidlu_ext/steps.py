import contextlib as ctx
import typing as T
import dataclasses as dc
from functools import partial
import copy

import torch
from torch import nn

import vidlu.torch_utils as vtu
import vidlu.modules.losses as vml
from vidlu.utils.collections import NameDict
from vidlu.training.steps import optimization_step, _prepare_semisup_batch_s, update_mean_teacher

def _cons_output_to_target(out_uns, stop_grad, output_to_target):
    with torch.no_grad() if stop_grad else ctx.suppress():
        target_uns = output_to_target(out_uns)
    if stop_grad:  # for the case when output_to_target is identity
        target_uns = target_uns.detach()
    return target_uns


def _perturb(attack, model, x, attack_target, loss_mask='create', attack_eval_model=False):
    """Applies corresponding perturbations to the input, unsupervised target, and validity mask.
    Also computes the prediction in the perturbed input."""
    if loss_mask == 'create':
        loss_mask = torch.ones_like(attack_target[:, 0, ...])
    with vtu.switch_training(model, False) if attack_eval_model else \
            vtu.norm_stats_tracking_off(model) if model.training else ctx.suppress():
        pmodel = attack(model, x, attack_target, loss_mask=loss_mask)
        x_p, target_p, loss_mask_p = pmodel(x, attack_target, loss_mask)
    return NameDict(x=x_p, target=target_p, loss_mask=loss_mask_p.detach(), pmodel=pmodel)


@dc.dataclass
class SemisupCleanTargetConsStepBase:
    """Base class for VAT, mean teacher, ...

    attack_eval_model is set to False because training on adversarial examples
    of the evaluation model instance is more likely to result in overfitting to
    adversarial examples.
    """
    alpha: float = 1
    attack_eval_model: bool = False  # intentionally false to avoid overfitting
    pert_bn_stats_updating: bool = False
    uns_loss_on_all: bool = False
    entropy_loss_coef: float = 0
    loss_cons: T.Optional[T.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    block_grad_on_clean: bool = True
    block_grad_on_pert: bool = False
    eval_mode_teacher: bool = False
    pert_both: bool = False
    rev_cons: bool = False
    mem_efficient: bool = True  # TODO: replace with ~joint_batch
    eval: bool = False

    def __call__(self, trainer, batch):
        attack = trainer.attack
        model, teacher = self.get_student_and_teacher(trainer)
        loss_cons, output_to_target = self._get_cons_loss_and_output_to_target(attack)

        if teacher is not model and not self.mem_efficient:
            raise RuntimeError("Cannot run unlabeled and labeled examples in the same batch "
                               + "because the teacher does not equal the student")
        joint_batch = teacher is model and (not self.mem_efficient or self.uns_loss_on_all)

        x_l, y_l, x_u, x_all = _prepare_semisup_batch_s(batch, self.uns_loss_on_all, joint_batch)
        loss_mask = 'create'
        perturb_x_u = lambda attack_target, loss_mask: _perturb(
            attack=attack, model=model, x=x_u, attack_target=attack_target,
            attack_eval_model=self.attack_eval_model, loss_mask=loss_mask)

        model.eval() if self.eval else model.train()  # teacher always in eval mode

        with optimization_step(trainer.optimizer) if not self.eval else ctx.suppress():
            # Supervised loss
            with torch.no_grad() if self.eval else ctx.suppress():
                out_all = model(x_all) if joint_batch else None
                out_l = out_all[:len(y_l)] if joint_batch else model(x_l)
                loss_l = trainer.loss(out_l, y_l, reduction="mean")
                if not self.eval and not joint_batch:  # back-propagates supervised loss sooner
                    loss_l.backward()
                    loss_l = loss_l.detach()

            # Unsupervised loss
            detach_clean = self.eval or self.block_grad_on_clean
            detach_pert = self.eval or self.block_grad_on_pert  # non-default

            with torch.no_grad() if detach_clean else ctx.suppress():
                out_u, loss_mask, additional = self._get_teacher_out(teacher, x_u, perturb_x_u,
                                                                     loss_mask, out_all)
                pert = perturb_x_u(attack_target=output_to_target(out_u), loss_mask=loss_mask)
                del pert.pmodel  # memory saving
                if detach_clean:
                    pert.target = pert.target.detach()

            with torch.no_grad() if detach_pert else ctx.suppress():
                with ctx.suppress() if self.pert_bn_stats_updating else \
                        vtu.norm_stats_tracking_off(model):
                    pert.out = model(pert.x)
            with torch.no_grad() if self.eval else ctx.suppress():
                loss_u = loss_cons(pert.out, pert.target)[pert.loss_mask >= 1 - 1e-6].mean()
                loss = loss_l.add(loss_u, alpha=self.alpha)
                with ctx.suppress() if self.entropy_loss_coef else torch.no_grad():  # memory saving
                    loss_ent = vml.entropy_l(pert.out).mean()
                    if self.entropy_loss_coef:
                        loss.add_(loss_ent, alpha=self.entropy_loss_coef)
                if not self.eval:
                    loss.backward()

        return NameDict(x=x_l, target=y_l, out=out_l, loss_l=loss_l.item(), loss_u=loss_u.item(),
                        x_u=x_u, x_p=pert.x, y_p=pert.target, out_u=out_u, out_p=pert.out,
                        loss_ent=loss_ent.item(), **additional)

    def get_student_and_teacher(self, trainer):
        return [trainer.model] * 2

    def _get_cons_loss_and_output_to_target(self, attack):
        lc = self.loss_cons or attack.loss
        loss_cons = (lambda a, b: lc(b, a)) if self.rev_cons else lc
        output_to_target = partial(_cons_output_to_target, stop_grad=self.block_grad_on_clean,
                                   output_to_target=attack.output_to_target)
        return loss_cons, output_to_target

    def _get_teacher_out(self, teacher, x_u, perturb_x_u, loss_mask, out_all):
        with vtu.switch_training(teacher,
                                 False) if self.eval_mode_teacher else ctx.suppress():
            if self.pert_both:
                loss_mask, out_u, additional = self._run_pert_teacher_branch(
                    x_u, perturb_x_u, teacher, loss_mask)
            else:  # default
                additional = NameDict()
                out_u = teacher(x_u) if out_all is None else out_all[-len(x_u):]
        return out_u, loss_mask, additional

    def _run_pert_teacher_branch(self, x_u, perturb_x_u, teacher, loss_mask):
        additional = NameDict()
        pertt = perturb_x_u(attack_target=x_u.new_zeros(x_u[:, :1].shape),
                            loss_mask=loss_mask)
        with vtu.norm_stats_tracking_off(
                teacher) if self.pert_bn_stats_updating else ctx.suppress():
            pertt.out = teacher(pertt.x)
        additional.out_up = out_up = pertt.out
        additional.x_pt = pertt.x
        if len(out_up.squeeze().shape) > 2:  # TODO: make more general
            additional.x_pr, out_u, loss_mask = pertt.pmodel.tps.inverse(
                pertt.x, out_up, pertt.loss_mask)
        else:
            out_u = out_up
        return loss_mask, out_u, additional


@dc.dataclass
class SemisupVATStep(SemisupCleanTargetConsStepBase):
    pass


@dc.dataclass
class SemisupVATEvalStep(SemisupVATStep):
    eval: bool = True
    uns_loss_on_all: bool = True


@dc.dataclass
class MeanTeacherStep(SemisupCleanTargetConsStepBase):
    ema_decay: float = 0.99
    ema_teacher: T.Optional[T.Union[nn.Module, dict]] = None  # should this be here?

    def __call__(self, trainer, batch):
        result = super().__call__(trainer, batch)
        update_mean_teacher(self.ema_teacher, trainer.model, self.ema_decay)
        return result

    def get_student_and_teacher(self, trainer):
        model = trainer.model
        if self.ema_teacher is None or isinstance(self.ema_teacher, dict):
            state_dict, self.ema_teacher = self.ema_teacher, copy.deepcopy(model)
            self.ema_teacher.eval()
            if state_dict is not None:
                self.ema_teacher.load_state_dict(state_dict)
        elif next(self.ema_teacher.parameters()).device != next(model.parameters()).device:
            self.ema_teacher.to(model.device)
        return trainer.model, self.ema_teacher

    def state_dict(self):
        t = self.ema_teacher
        return t if isinstance(t, dict) else dict() if isinstance(t, dict) else t.state_dict()

    def load_state_dict(self, state_dict):
        if self.ema_teacher is None or isinstance(self.ema_teacher, dict):
            self.ema_teacher = dict(state_dict) if len(state_dict) > 0 else None
        else:
            self.ema_teacher.load_state_dict(state_dict)
