import torch
from torch.autograd import Variable
import torchvision.models as models
import numpy as np

import params
import helper

opt = params.opt

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def attack():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = opt.batch_size
    eps = opt.epsilon
    images, labels = helper.get_image_with_label()

    # to cuda
    images = Variable(FloatTensor(images.to('cuda')), requires_grad=True)
    labels = Variable(LongTensor(labels.to('cuda')), requires_grad=False)

    # initial net for computing perturbation
    net = models.__dict__[opt.arch](pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # label mapping
    class_labels = helper.get_imagenet_labels()

    # generate perturbation
    criterion = torch.nn.CrossEntropyLoss()
    adv_labels, init_pert = helper.non_targeted_attack(net, images, labels, criterion, eps=eps)
    adv_examples, opt_pert = helper.optimize_weight_sgd(net, images, labels, adv_labels,
                                                        init_pert, seed=seed, eps=eps)

    # prediction
    adv_pred, adv_conf = helper.predict_with_confidence(net, adv_examples)
    adv_pred = adv_pred.detach().cpu().numpy()
    adv_label = class_labels[adv_pred[0]]

    # LO norm
    opt_pert_norm = helper.lp_norm(opt_pert, batch_size).reshape(batch_size)

    # visualize result
    label = class_labels[labels.detach().cpu().numpy()[0]]
    helper.plot(images, label, opt_pert, opt_pert_norm, adv_examples, adv_label)


if __name__ == '__main__':
    attack()
