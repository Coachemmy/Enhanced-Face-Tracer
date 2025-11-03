import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def lpips_loss(x, y, loss_fn):
    out = loss_fn(x, y).mean()
    return out

def cosine_loss(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return 1 - cos(x, y)

def l2_loss(x, y):
    return nn.MSELoss()(x, y)

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res

class AAMSoftmax(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, n_class, s, margin=0.5):
        super(AAMSoftmax, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 512), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.easy_margin = False
        self.k = 5
        self.ce = nn.CrossEntropyLoss()


    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mode: str, k: int):
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m # cos(A+B) = cosAcosB-sinAsinB
        phi = torch.where((cosine - self.theta) > 0, phi, cosine - self.sinmm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        if mode == 'normal':
            loss = self.ce(output, labels)
            prec1 = accuracy(output.detach(), labels.detach(), topk=(1,k))[1]

            return loss, prec1
        elif mode == 'cross':
            _, max_label = output.topk(1, 1)
            return max_label[0][0]
        else:
            raise Exception('Unsupported mode!')

class Softmax(torch.nn.Module):
    def __init__(self, n_class):
        super(Softmax, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 512), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mode: str, k: int):
        output = F.linear(F.normalize(logits), F.normalize(self.weight))
        if mode == 'normal':
            loss = self.ce(output, labels)
            prec1 = accuracy(output.detach(), labels.detach(), topk=(1,k))[1]

            return loss, prec1
        elif mode == 'cross':
            _, max_label = output.topk(1, 1)
            return max_label[0][0]
        else:
            raise Exception('Unsupported mode!')
