import torch
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer, required
import itertools as it
import torch.nn.functional as F
from torch.autograd import Variable

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP) + torch.sum(FN)) + 1e-6)

    return SE, torch.sum(FN)


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN) + torch.sum(FP)) + 1e-6)

    return SP, torch.sum(FP)


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP) + torch.sum(FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte() + GT.byte()) == 2)
    Union = torch.sum((SR.byte() + GT.byte()) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte() + GT.byte()) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5

        num = target.size(0)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, beta=0.3):
        batch_size = target.size(0)
        loss = 0.0
        alpha = 1.0 - beta

        for i in range(batch_size):
            prob = input[i]
            ref = target[i]

            tp = (ref * prob).sum()
            fp = ((1 - ref) * prob).sum()
            fn = (ref * (1 - prob)).sum()
            tversky = tp / (tp + alpha * fp + beta * fn)
            loss = loss + (1 - tversky)
        return loss / batch_size

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


class GHMC(nn.Module):
    """

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """

    def __init__(self, bins=10, momentum=0, loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight



    def forward(self, pred, target, label_weight=None, *args, **kwargs):
        """Calculate the GHM loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label

        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(
            target, label_weight, pred.size(-1))

        label_weight = torch.ones((target.size(0),target.size(1))).cuda()

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred).cuda()

        # gradient length
        g = torch.abs(pred.detach() - target).cuda()
        # valid label center
        valid = label_weight > 0
        # valid label number
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # Divide the corresponding gradient value into the corresponding bin, 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            # samples exist in the bin
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment calculate num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                                      + (1 - mmt) * num_in_bin
                    # weights/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot

        return loss

class ssloss(nn.Module):
    def __init__(self):
        super(ssloss, self).__init__()

    def forward(self, input, target):
        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()
        tn = ((1 - input) * (1 - target)).sum()
        seloss = fn / (tp + fn)
        sploss = fp / (tn + fp)
        loss = seloss + sploss

        return loss

class mix(nn.Module):
    def __init__(self):
        super(mix, self).__init__()
        self.loss1 = GHMC()
        self.loss2 = ssloss()

    def forward(self,input, target, alpha=0.7):
        loss = alpha * self.loss1(input,target) + (1-alpha) * self.loss2(input, target)
        return loss

class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):

        logpt = - F.binary_cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss



class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])

                p.data.copy_(p_data_fp32)

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0

        self.slow_weights = [
            [p.cuda().clone().detach() for p in group['params']]
            for group in self.param_groups]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False
        self.state = optimizer.state

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()

        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(p.data - q.data, alpha=self.alpha)
                p.data.copy_(q.data)
        return loss


