import dis
import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import fastai
from fastai.vision import *
from fastai.callbacks import *
import torch.nn.functional as F
import plot
from adversarial import run_wgan
from adversarial import run_waae

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, **targets):
    labels,_=targets
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


class LossCallback(Callback):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def on_epoch_begin(self, **kwargs):
        self.loss_reconstruction, self.total = 0, 1e-8

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.loss_reconstruction / self.total)

    def __call__(self, item):
        self.loss_reconstruction += item
        self.total += 1

class LossModel(LossCallback):
    def __init__(self, name="model loss"):
        super().__init__(name)

class LossReconstruction(LossCallback):
    def __init__(self, name="reconstruction loss"):
        super().__init__(name)

class LossGenerator(LossCallback):
    def __init__(self, name="generator loss"):
        super().__init__(name)

class LossDiscriminator(LossCallback):
    def __init__(self, name="discriminator loss"):
        super().__init__(name)


class GC(LearnerCallback):
    def __init__(self,learn, params, clip):
        super(GC, self).__init__(learn)
        self.params=params
        self.clip=clip

    def on_backward_end(self, **kwargs):
        nn.utils.clip_grad_norm_(self.params, self.clip)


class Precision(Callback):

    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        logits, _ = last_output
        v, a = last_target

        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*a.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * a)

        self.correct += scores.sum()
        self.total += a.size(0)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.correct / self.total)

class TotalLoss(nn.Module):
    def __init__(self, reconstruction, adv_mode, g, d, d_lr, gamma_r, gamma_a, loss_m, loss_r, loss_g, loss_d, penality,
                 critic,
                 adv):
        super().__init__()
        self.gamma_r = gamma_r
        self.gamma_a = gamma_a
        self.reconstruction = reconstruction
        self.adv = adv

        self.loss_m = loss_m
        self.loss_r = loss_r
        self.loss_g = loss_g
        self.loss_d = loss_d

        if reconstruction:
            self.G = g
            self.D = d
            self.adv_mode = adv_mode
            if adv_mode == "wgan":
                self.run_adv = run_wgan
            else:
                self.run_adv = run_waae

            self.optimizer_D = torch.optim.Adamax(self.D.parameters(), lr=d_lr, betas=(0.5, 0.999))
            self.critic = critic
            self.penality = penality

    def forward(self, input, v, a):
        logits, h = input

        loss_rec = 0
        g_loss = 0
        if self.reconstruction and self.D.training:
            #adv loss
            if self.adv:
                g_loss, d_loss, g_rec, v = self.run_adv(self.G, self.D, self.optimizer_D, self.penality, self.critic, h,
                                                        v)
                self.loss_g(g_loss.data)
                if d_loss:
                    self.loss_d(d_loss.data)
            else:
                v = v.permute(0, 2, 1).view(h.size(0), -1, 6, 6)
                g_rec = self.G(h)

            #rec loss
            if self.adv_mode == "waae":
                loss_rec = F.mse_loss(g_rec, v)
                self.loss_r(loss_rec.data)

        #model loss
        loss_model = instance_bce_with_logits(logits, a)
        self.loss_m(loss_model.data)
        return loss_model + self.gamma_r * loss_rec + self.gamma_a * g_loss

class SaveModel(LearnerCallback):
    def __init__(self, learn, output):
        super(SaveModel, self).__init__(learn)
        self.best_accuracy_score = 0.0
        self.output = output
        self.model = learn

    def on_epoch_end(self, last_metrics, **kwargs):
        #last_metrics = val_loss, metric1, metric2,...
        accuracy_score = last_metrics[1]
        if accuracy_score > self.best_accuracy_score:
            old = os.path.join(self.output, 'model_%.4f.pth' % float(self.best_accuracy_score))
            if os.path.exists(old):
                os.remove(old)
            ret = self.model.save('model_%.4f' % float(accuracy_score), return_path=True)
            self.best_accuracy_score = accuracy_score

    def on_train_end(self,**kwargs):
        for p in ["plot_lr","plot_losses","plot_metrics"]:
            f = getattr(plot,p)
            fig, ax = f(self.learn.recorder)
            out = os.path.join(self.output,p)
            pickle.dump(ax, open(out, 'wb+'))
            fig.savefig(out)

        with open("all_scores.txt", "a+") as f:
            f.write('%s-%.4f \n'%(self.output,self.best_accuracy_score))


def train(model, train_loader, eval_loader, num_epochs, batch_size, output, reconstruction, lr,
          gamma_r,
          gamma_a,
          size,
          early_stop,
          dropout_hid,
          adv_mode,
          penality,
          critic,
          adv,
          ckpt,
          logger):

    logger.write("batch_size %s, "
                 "reconstruction %s, "
                 "gamma_r %s, "
                 "gamma_a %s, "
                 "size %s, "
                 "dropout_hid %s, "
                 "adv_mode %s, "
                 "penality %s, "
                 "critic %s, "
                 "ad %s, "
                 % (str(batch_size), reconstruction, str(gamma_r), str(gamma_a), str(size), str(dropout_hid),
                    adv_mode,
                    str(penality),
                    str(critic),
                    str(adv),
                    ))


    generator, discriminator = None, None,
    #define main model
    model_group = [model.w_emb,
                model.q_emb,
                model.v_att,
                model.q_net,
                model.v_net,
                model.classifier]
    generator_group = []
    if reconstruction:
        #define adversarial, include generator on main model
        discriminator = model.D
        generator = model.G
        generator_group = [model.G]

    #Loss
    lmod = LossModel()
    lrec = LossReconstruction()
    ladvg = LossGenerator()
    ladvd = LossDiscriminator()
    total_loss = TotalLoss(reconstruction, adv_mode, generator, discriminator, lr/10, gamma_r, gamma_a,
                           lmod,
                           lrec,
                           ladvg,
                           ladvd,
                           penality,
                           critic,
                           adv,
                           )

    #Learner
    learn = Learner(DataBunch(train_dl=train_loader,
                              valid_dl=eval_loader),
                    model=model,
                    wd=0.0,
                    metrics=[Precision(), lmod, lrec, ladvg, ladvd],
                    loss_func=total_loss,
                    model_dir=output,
                    opt_func=torch.optim.Adamax,
                    layer_groups=model_group+generator_group)


    if ckpt is not None:
        print("Loading", str(ckpt))
        learn.load(ckpt)


    #Callback
    p_gc = []
    for m in model_group:
        p_gc += list(m.parameters())
    gp = GC(learn, p_gc, 0.25)
    sm = SaveModel(learn, output=output)
    es = EarlyStoppingCallback(learn=learn, monitor='precision', min_delta=0.0001, patience=early_stop)


    print("LR is",lr)

    #Go
    learn.fit(num_epochs, lr, callbacks=[gp,sm,es])


def evaluate(model, eval_loader, output, ckpt):
    lmod = LossModel()
    lrec = LossReconstruction()
    ladvg = LossGenerator()
    ladvd = LossDiscriminator()
    total_loss = TotalLoss(True, False, model.G, model.D, 10/10, 0, 0.,
                           lmod,
                           lrec,
                           ladvg,
                           ladvd,
                           0.,
                           0.,
                           0.,
                           )

    learn = Learner(DataBunch(train_dl=eval_loader,
                              valid_dl=eval_loader),
                    model=model,
                    wd=0.0,
                    metrics=[Precision(), lmod, lrec],
                    loss_func=total_loss,
                    model_dir=output,
                    )
    learn.load(ckpt)

    print(learn.validate()[1])

