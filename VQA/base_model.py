import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from adversarial import Generator, Discriminator
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid, v_dim, reconstruction, size,
                 dropout_hid, gamma_r, adv_mode, logger):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.reconstruction = reconstruction
        self.num_hid = num_hid
        self.v_dim = v_dim
        self.size = size
        self.dropout_hid = dropout_hid
        self.gamma_r = gamma_r
        self.adv_mode = adv_mode
        self.d = nn.Dropout(self.dropout_hid)

        if self.reconstruction:
            if adv_mode == "wgan":
                self.G = Generator(noise_dim=0)
                self.D = Discriminator(num_hid=0)

            else:
                self.G = Generator(noise_dim=0)
                self.D = nn.Sequential(
                    nn.Linear(self.num_hid, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                    )

            logger.write('G parameters %.2f M' % (sum(p.numel() for p in self.G.parameters() if p.requires_grad) / 1e6))
            logger.write('D parameters %.2f M' % (sum(p.numel() for p in self.D.parameters() if p.requires_grad) / 1e6))

    def forward(self, _, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        q = q.cuda()
        v = v.cuda()
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)

        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = self.d(q_repr * v_repr)
        logits = self.classifier(joint_repr)
        return logits, joint_repr

def build_baseline0_newatt(dataset, num_hid, reconstruction, size=64, dropout_hid=0.0, gamma_r=0.0, adv_mode="wgan", logger=None):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid, dataset.v_dim,
                     reconstruction,
                     size,
                     dropout_hid,
                     gamma_r,
                     adv_mode,
                     logger)
