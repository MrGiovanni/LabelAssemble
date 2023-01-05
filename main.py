import argparse
from logger import Logger
import os
import datetime 
import os.path as osp
from train import train
from test import test
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='6'
def get_arguments():
    parser = argparse.ArgumentParser(description="Assemble Label")
    parser.add_argument("--datasetType", type=str, default='assemble', help='The dataset you want to use.')  
    parser.add_argument("--device", type=str, default='cuda', help='cpu or cuda')
    parser.add_argument("--numClass", type=int, default=2, help='the total classes')
    parser.add_argument("--mode", type=str, default='train', help='train/val/test')
    parser.add_argument("--epochs", type=int, default=64, help='the epochs you want to train the model')
    parser.add_argument("--testInterval", type=int, default=1, help='test after testInterval epochs')
    parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
    parser.add_argument("--numWorkers", type=int, default=32, help='num of wokers')
    parser.add_argument("--batchSize", type=int, default=16, help='batch size')
    parser.add_argument("--saveDir", type=str, help='the model parameters saves in saveDir')
    parser.add_argument("--resumePath", type=str, help='pretrained model path')
    parser.add_argument("--workDir", type=str, default='', help='work directory')
    parser.add_argument("--loss", type=str, default='fully', help='fully/semi')
    return parser


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    if args.workDir != '':
        work_dir = args.workDir
    else:
        work_dir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    work_dir = osp.join('.work_dir', work_dir)
    os.makedirs(work_dir, exist_ok=True)
    logger = Logger(osp.join(work_dir, 'logging.txt'))
    logger.info('created work directory in %s' % work_dir)
    args.saveDir = osp.join(work_dir, args.saveDir)
    if args.mode == 'train':
        logger.info('Training starts.')
        train(args)
    else:
        logger.info('Test starts.')
        model = torch.load(args.resumePath)
        test(model, args)
    
