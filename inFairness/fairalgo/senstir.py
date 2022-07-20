
from tkinter import Y
import torch
from torch import nn

from inFairness.auditor import SenSTIRAuditor
from inFairness.utils import datautils

from datainterfaces import FairModelResponse


class SenSTIR(nn.Module):
  def __init__(self, network, distance_q, distance_y, loss_fn, rho, eps, auditor_nsteps, auditor_lr):
    self.network = network
    self.distance_q = distance_q
    self.distance_y = distance_y
    self.loss_fn = loss_fn
    self.rho = rho
    self.eps = eps
    self.auditor_nsteps = auditor_nsteps
    self.auditor_lr = auditor_lr
    self.auditor = self.__init_auditor__()

  def __init_auditor__(self):
    auditor = SenSTIRAuditor(
      self.distance_q,
      self.distance_y,
      self.auditor_nsteps,
      self.auditor_lr,
    )
  
  def forward_train(self, Q, rels):
    device = datautils.get_device(X)

    min_lambda = torch.tensor(1e-5, device=device)

    if self.lamb is None:
      self.lamb = torch.tensor(1e-5, device = device)
    if type(self.eps) is float:
      self.eps = torch.tensor(self.eps, device=device)
    
    Q_worst = self.auditor.generate_worst_case_examples(
      self.network, Q, self.lamb
    )

    mean_dist_q = self.distance_q(Q,Q_worst).mean()
    lr_factor = torch.maximum(mean_dist_q, self.eps) / torch.minimum(mean_dist_q, self.eps)
    self.lamb = torch.maximum(min_lambda, self.lamb + lr_factor * (mean_dist_q - self.eps))

    rels_pred = self.network(Q)
    rels_pred_worst = self.network(Q_worst)

    fair_loss = torch.mean(
      self.loss_fn(rels_pred, rels) + self.rho * self.distance_y(rels_pred, rels_pred_worst)
    )

    response = FairModelResponse(loss=fair_loss, y_pred=rels_pred)
    return response