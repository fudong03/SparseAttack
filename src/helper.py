import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import params
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
opt = params.opt


def get_imagenet_labels(txt_path='./../input/imagenet_classes.txt'):
    """
    Returns all imagenet labels

    :return:
    """

    with open(txt_path) as f:
        classes = [line.strip() for line in f.readlines()]

    return classes


def get_image_with_label():
    """
    Returns normalized image with ground-truth label

    :return:
    """

    img_path = "./../input/giant_panda.jpeg"
    img_orig = Image.open(img_path)

    images = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])(img_orig)

    images = images.reshape(1, 3, 224, 224)

    # giant panda
    label = 388
    labels = torch.tensor([label])

    return images, labels


def non_targeted_attack(net, clean_images, orig_labels, criterion, iter_num=opt.iter_num, eps=6 / 255):
    """
    Generate targeted adversarial examples and perturbation through I-FGSM.

    :param eps:
    :param iter_num:
    :param net:
    :param clean_images:
    :param orig_labels:
    :param criterion:
    :return:
    """

    clean_images = Variable(FloatTensor(clean_images.to('cuda')), requires_grad=True)
    orig_labels = Variable(LongTensor(orig_labels.to('cuda')), requires_grad=False)

    adv_examples = clean_images
    for i in range(iter_num):
        outputs = net(adv_examples)
        ae_loss = criterion(outputs, orig_labels)
        net.zero_grad()
        ae_loss.backward()

        data_grad = clean_images.grad.data
        adv_examples = fgsm(clean_images, data_grad, eps=eps)

    adv_labels = predict(net, adv_examples)
    initial_pert = clean_images - adv_examples

    return adv_labels, initial_pert


def fgsm(images, data_grad, eps=6 / 255, targeted=False, val_min=opt.val_min, val_max=opt.val_max):
    """
    Generating perturbation through FGSM

    :param images:
    :param eps:
    :param data_grad:
    :param targeted: bool, whether the FGSM is targeted or non-targeted
    :param val_min:
    :param val_max:

    :return:
    """

    if targeted:
        eps = -1 * eps

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images + eps * sign_data_grad

    # Adding clipping to maintain [-1,1] range
    perturbed_images = torch.clamp(perturbed_images, val_min, val_max)

    return perturbed_images


def predict(net, image):
    """
    Predict labels through classifier f

    :param net:
    :param image:
    :return:
    """

    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1)

    return predicted


def predict_with_confidence(net, image):
    """
    Predict labels through classifier f

    :param net:
    :param image:
    :return:
    """

    outputs = net(image)
    softmax_val, predicted = torch.max(outputs.data, 1)

    probabilities = torch.nn.functional.softmax(outputs.data, dim=1)[0] * 100

    return predicted, probabilities[predicted.detach().cpu().numpy()[0]]


def tuning_opt_weight(net, labels, opt_W, cur_W, opt_adv, cur_adv, norm=0):
    """
    Tuning optimal weight

    :param net:
        the architecture that produces gradients
    :param labels:
        ground truth labels
    :param opt_W:
        last optimal weight
    :param cur_W:
        current weight
    :param cur_adv:
        current adversarial examples
    :param opt_adv:
        optimal adversarial examples
    :param norm:
        distance matrix

    :return:
    """
    batch_size = opt_W.shape[0]
    channel_num = opt_W.shape[1]
    width = opt_W.shape[2]
    height = opt_W.shape[3]

    # reshape for torch.where filter
    cur_W = cur_W.reshape(batch_size, channel_num * width * height)
    opt_W = opt_W.reshape(batch_size, channel_num * width * height)

    cur_w_norm = torch.norm(cur_W, p=norm, dim=1, keepdim=True)
    opt_w_norm = torch.norm(opt_W, p=norm, dim=1, keepdim=True)

    # labels that the classifier outputs
    outputs = net(cur_adv)
    _, predicted = torch.max(outputs.data, 1)

    # check whether current weight can fool the classifier
    cond1 = predicted != labels
    cond1 = cond1.reshape(batch_size, 1)

    # the norm of current weight less than the norm of last optimal weight
    cond2 = torch.gt(opt_w_norm, cur_w_norm)

    opt_W = torch.where(torch.logical_and(cond1, cond2), cur_W, opt_W)
    opt_adv = torch.where(torch.logical_and(cond1, cond2), cur_adv, opt_adv)

    # reshape to original shape
    opt_W = opt_W.reshape(batch_size, channel_num, width, height)

    return opt_W, opt_adv


def plot_adv(images, label, conf):
    """
    plot adversarial examples and corresponding classification result

    :param images: resulting adversarial examples
    :param label: incorrect prediction
    :param conf: prediction confidence
    :return:
    """

    plt.figure(figsize=(6, 5.5), dpi=80)

    if torch.cuda.is_available():
        images = images.detach().cpu()

    batch_size = images.shape[0]

    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.tight_layout()

        formatter = "{0:.1f}"
        confidence = formatter.format(conf)

        image = images[i]
        image = image.permute(1, 2, 0)
        plt.imshow((image * 255).numpy().astype(np.uint8))

        title = "{}, {}% confidence".format(label, confidence)
        plt.title(title, fontsize=20, fontfamily='Times New Roman')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def plot_pert(pert_list, pert_norms):
    """
    plot perturbation and perturbed pixels

    :param pert_list: resulting adversarial examples
    :param pert_norms:
    :return:
    """

    plt.figure(figsize=(6, 5.5), dpi=80)

    if torch.cuda.is_available():
        pert_list = pert_list.detach().cpu()
        pert_norms = pert_norms.detach().cpu().numpy()

    batch_size = pert_list.shape[0]

    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.tight_layout()

        pert = pert_list[i]
        pert = pert.permute(1, 2, 0)
        plt.imshow((pert * 255).numpy().astype(np.uint8))

        # calculate percentage
        formatter = "{0:.3f}"
        total_pixel = pert.shape[0] * pert.shape[1] * pert.shape[2]
        norm = int(pert_norms[i])
        percentage = formatter.format(norm / total_pixel * 100)

        title = "# perturbed pixels: {} ({}%)".format(norm, percentage)
        plt.title(title, fontsize=18, fontfamily='Times New Roman')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def plot(images, label, pert_list, pert_norms, adv_images, adv_label):
    """
    Plot natural images, noise, and corresponding adversarial examples.

    :param images:
        natural images
    :param label:
        true label
    :param pert_list:
        perturbations
    :param pert_norms:
        l0 norm of perturbations
    :param adv_images:
        adversarial examples
    :param adv_label:
        adversarial label

    """

    # detach
    if torch.cuda.is_available():
        images = images.detach().cpu()
        pert_list = pert_list.detach().cpu()
        pert_norms = pert_norms.detach().cpu().numpy()
        adv_images = adv_images.detach().cpu()

    image = images[0]
    image = image.permute(1, 2, 0)

    pert = pert_list[0]
    pert = pert.permute(1, 2, 0)
    pert_norm = int(pert_norms[0])

    adv_image = adv_images[0]
    adv_image = adv_image.permute(1, 2, 0)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    # calculate percentage
    formatter = "{0:.3f}"
    total_pixel = pert.shape[0] * pert.shape[1] * pert.shape[2]
    percentage = formatter.format(pert_norm / total_pixel * 100)
    pert_title = '{} pixels ({}%)'.format(pert_norm, percentage)

    axes[0].set_title(label)
    axes[1].set_title(pert_title)
    axes[2].set_title(adv_label)

    axes[0].imshow((image * 255).numpy().astype(np.uint8))
    axes[1].imshow((pert * 255).numpy().astype(np.uint8))
    axes[2].imshow((adv_image * 255).numpy().astype(np.uint8))

    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def optimize_weight_sgd(net, images, labels, target_labels, init_pert, eps=6 / 255, seed=1, tau=opt.tau):
    """
    Optimize weight through gradient descent

    :param seed:
    :param eps:
    :param net:
    :param images:
    :param labels:
    :param target_labels:
    :param init_pert:
    :return:
    """
    torch.manual_seed(seed)

    Omega = get_omega(images, eps)

    batch_size = images.shape[0]
    img_size = images.shape[2]
    criterion = torch.nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if cuda else "cpu"

    # initialize weight
    opt_adv = images - init_pert
    opt_W = torch.ones((batch_size, images.shape[1], img_size, img_size), device=device)
    W = torch.randn((batch_size, images.shape[1], img_size, img_size), requires_grad=True, device=device)

    for _ in tqdm(range(opt.opt_epochs)):
        noise_prime = torch.mul(W, init_pert)
        cur_adv = images - noise_prime

        # update optimal weight
        opt_W, opt_adv = tuning_opt_weight(net, labels, opt_W, W, opt_adv, cur_adv, norm=0)

        outputs = net(cur_adv)
        c_loss = criterion(outputs, target_labels)
        n_loss = torch.norm(W, p=0)
        loss = opt.beta * (opt.lamb * c_loss + n_loss)

        # calculate gradients
        grads = torch.autograd.grad(loss, [W])
        W_grad = grads[0]
        W = W - opt.learning_rate * W_grad

        W = activate_weight(W, Omega, eps, tau=tau)

    opt_pert = images - opt_adv

    return opt_adv, opt_pert


def activate_weight(W, Omega, eps, tau=opt.tau, relu=torch.nn.ReLU(), tanh=torch.nn.Tanh()):
    """
        The relu activation function serves to l1-approximation l0 attack
        and the tanh function serves to meet the box constraint.

        :param eps:
        :param tanh:
        :param Omega:
        :param tau:
        :param relu:
        :param W:
    """

    W = relu(W - tau / eps)

    W = tanh(W)
    W = torch.mul(Omega, W)
    return W


def get_omega(X, eps):
    """
    Computing the hyper-parameter Omega

    :param X:
    :return:
    """

    constrain1 = torch.mul(1 / eps, X)
    constrain2 = torch.mul(1 / eps, 1 - X)
    Omega = torch.min(constrain1, constrain2)
    return Omega


def lp_norm(X, batch_size, p=0):
    """ return L_p norm """

    view = X.view(batch_size, -1)
    norm = torch.norm(view, p=p, dim=1, keepdim=True)
    return norm
