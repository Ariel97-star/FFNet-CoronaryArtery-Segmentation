import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','FFNet', 'AttU_Net', 'ResU_Net', 'SegNet']:
        print('ERROR!! model_type should be selected in exist models')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train')
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid')
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=1,
                             num_workers=config.num_workers,
                             mode='test')

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=320)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save', type=int, default=1, help='0/1')
    parser.add_argument('--optimizer_type', type=str, default='RAdam', help='Adam/RAdam')
    parser.add_argument('--criterion_type', type=str, default='Tversky',help='BCE/Focal/mix/GHMC/Tversky/Dice')
    parser.add_argument('--model_type', type=str, default='FFNet',
                        help='U_Net/FFNet/AttU_Net/ResU_Net/SegNet')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--bestmodel_path', type=str,
                        default="./bestmodel.pkl")
    parser.add_argument('--train_path', type=str, default="./dataset/train/")
    parser.add_argument('--valid_path', type=str, default="./dataset/valid/")
    parser.add_argument('--test_path', type=str, default="./dataset/valid/")
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--cuda_idx', type=list, default=[0,1])

    config = parser.parse_args()
    main(config)
