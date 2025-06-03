import argparse

parser = argparse.ArgumentParser()

# hyper-parameters for I-FGSM
parser.add_argument("-it", "--iter_num", type=int, default=10, help='the number of iterations for iterative FGSM.')
parser.add_argument("-min", "--val_min", type=int, default=-1, help='the minimum val of pixel.')
parser.add_argument("-max", "--val_max", type=int, default=1, help='the maximum val of pixel.')
parser.add_argument("-eps", "--epsilon", type=int, default=6/255, help='L infinity constraint')

# hyper-parameters for WI-FGSM
parser.add_argument("-lamb", "--lamb", type=int, default=1e3, help='balance the two losses.')
parser.add_argument("-bs", "--batch_size", type=int, default=1, help='batch size')
parser.add_argument('--black_box', dest='attack_type', action='store_true')
parser.add_argument('--white_box', dest='attack_type', action='store_false')

# white box
parser.set_defaults(attack_type=False)
parser.add_argument("-oe", "--opt_epochs", type=int, default=200, help='number of epochs for optimizing weight.')
parser.add_argument("-lr", "--learning_rate", type=int, default=1e-2, help='learning rate for optimizing weight.')
parser.add_argument("-beta", "--beta", type=int, default=1e2, help='accelerate the hyper-parameter tuning speed')
parser.add_argument("-tau", "--tau", type=float, default=0.45, help='hyper-parameter to adjust the relu')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16')

opt = parser.parse_args()
