import torch
import torch.nn as nn

from configs import get_trainmae_config


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def Color_print(line):
    print(bcolors.OKGREEN + line + bcolors.ENDC)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)

    return res


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        args = get_trainmae_config()
        self.args = args.parse_args()
        self.smooth = self.args.label_smooth

    def forward(self, x, y):
        x = torch.log_softmax(x, dim=1)
        y = y * (1 - self.smooth) + self.smooth / x.size(0)
        loss = -torch.sum(y.mul(x))

        return loss

