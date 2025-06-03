
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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

from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = [15, 5]

def get_image_with_label():
    """
    Returns normalized image with ground-truth label

    :return:
    """

    img_path = "./../input/candle.jpeg"
    label = 470

    img_orig = Image.open(img_path)

    images = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])(img_orig)

    images = images.reshape(1, 3, 224, 224)
    labels = torch.tensor([label])

    return images, labels


def grad_cam():
    # feature layer
    model = models.vgg16(pretrained=True)
    target_layers = [model.features[-1]]

    # clean image
    img, label = get_image_with_label()

    # mapping label
    class_labels = helper.get_imagenet_labels()
    viz_label = class_labels[label.detach().cpu().numpy()[0]]

    # plot clean image
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    plt.title("Clean Image \n \"{}\"".format(viz_label))
    plt.axis('off')

    # grad_cam
    input_tensor = img
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    guide = GuidedBackpropReLUModel(model=model, use_cuda=True)

    # use the predicted label as the decision of interest
    targets = [ClassifierOutputTarget(label)]

    # guided back-propagation
    grayscale_guide = guide(input_tensor)

    # grad-cam
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # guided grad-cam
    x = np.multiply(grayscale_cam.reshape(224, 224, 1), grayscale_guide)

    # plot guided grad-cam
    plt.subplot(1, 3, 3)
    plt.imshow(x, cmap='gray')
    plt.title("Guided Grad-CAM \n \"{}\"".format(viz_label))
    plt.axis('off')

    # grad-cam
    rgb_img = img[0, :].permute(1, 2, 0).detach().cpu().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # plot grad_cam
    plt.subplot(1, 3, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM \n \"{}\"".format(viz_label))
    plt.axis('off')
    plt.show()


def attack():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set default hyper-parameters
    opt.opt_epochs = 100
    eps = 10 / 255
    tau = 0.20
    images, labels = get_image_with_label()

    # to cuda
    images = Variable(FloatTensor(images.to('cuda')), requires_grad=True)
    labels = Variable(LongTensor(labels.to('cuda')), requires_grad=False)

    # initial net for computing perturbation
    net = models.__dict__["vgg16"](pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # mapping label
    class_labels = helper.get_imagenet_labels()

    # predict
    pred, conf = helper.predict_with_confidence(net, images)
    pred = pred.detach().cpu().numpy()
    pred_label = class_labels[pred[0]]
    print("ori prediction: {}, {}".format(pred[0], pred_label))

    criterion = torch.nn.CrossEntropyLoss()

    # attack
    adv_labels, init_pert = helper.non_targeted_attack(net, images, labels, criterion, eps=eps)
    adv_examples, opt_pert = helper.optimize_weight_sgd(net, images, labels, adv_labels, init_pert, seed=seed, eps=eps, tau=tau)

    # prediction
    adv_pred, adv_conf = helper.predict_with_confidence(net, adv_examples)
    adv_pred = adv_pred.detach().cpu().numpy()
    adv_label = class_labels[adv_pred[0]]
    print("adv prediction: {}, {}".format(adv_pred[0], adv_label))

    return adv_examples, adv_pred[0]


def grad_cam_noise():
    # feature layer
    model = models.vgg16(pretrained=True)
    target_layers = [model.features[-1]]

    # generate AE
    img, label = attack()

    # mapping label
    class_labels = helper.get_imagenet_labels()
    viz_label = class_labels[label]

    # plot adversarial example
    plt.subplot(1, 3, 1)
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    plt.title("Adversarial Example \n \"{}\"".format(viz_label))
    plt.axis('off')

    # grad_cam
    input_tensor = img.clone().detach().float().requires_grad_(True)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    guide = GuidedBackpropReLUModel(model=model, use_cuda=True)

    # use the predicted label as the decision of interest
    targets = [ClassifierOutputTarget(label)]

    # guided back-propagation
    grayscale_guide = guide(input_tensor)

    # grad-cam
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # guided grad-cam
    x = np.multiply(grayscale_cam.reshape(224, 224, 1), grayscale_guide)

    # plot guided grad-cam
    plt.subplot(1, 3, 3)
    plt.imshow(x, cmap='gray')
    plt.title("Guided Grad-CAM \n \"{}\"".format(viz_label))
    plt.axis('off')

    # grad-cam vis
    rgb_img = img[0, :].permute(1, 2, 0).detach().cpu().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # plot grad-cam
    plt.subplot(1, 3, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM \n \"{}\"".format(viz_label))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    grad_cam()
    grad_cam_noise()
