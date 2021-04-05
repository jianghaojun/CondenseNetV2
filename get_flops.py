import os
import argparse
import warnings

warnings.filterwarnings("ignore")
from model import *
from utils import measure_model

parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
parser.add_argument('--model', default='condensenetv2.cdnv2_a', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('--train_url', type=str, metavar='PATH', default=None,
                    help='path to save result and checkpoint (default: results/savedir)')

def main():
    args = parser.parse_args()

    assert args.dataset == 'imagenet'
    args.num_classes = 1000
    args.IMAGE_SIZE = 224

    ### Create Model
    model = eval(args.model)(args)
    n_flops, n_params = measure_model(model, args.IMAGE_SIZE, args.IMAGE_SIZE)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    if args.train_url:
        log_file = os.path.join(args.train_url + 'measure_model.txt')
        with open(log_file, "w") as f:
            f.write(str(n_flops / 1e6))
            f.write(str(n_params / 1e6))

        f.close()
    return


if __name__ == '__main__':
    main()
