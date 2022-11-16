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
    parser.add_argument("--datasetType", type=str, default='assemble')
    parser.add_argument("--covidxTrainImagePath", type=str)
    parser.add_argument("--covidxTestImagePath", type=str)
    parser.add_argument("--chestImagePath", type=str)
    parser.add_argument('--covidxTrainFilePath', type=str)
    parser.add_argument('--covidxTestFilePath', type=str)
    parser.add_argument("--chestFilePath", type=str)
    parser.add_argument("--extraNumClass", type=int, default=1)
    parser.add_argument("--numClass", type=int, default=2)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--testInterval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--covidxNum", type=int, default=1)
    parser.add_argument("--chestNum", type=int, default=1)
    parser.add_argument("--numWorkers", type=int, default=32)
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--saveDir", type=str)
    parser.add_argument("--isTrain", action="store_true")
    parser.add_argument("--resumePath", type=str)
    parser.add_argument("--workDir", type=str, default='')
    parser.add_argument("--loss", type=str, default='fully')
    # temperature 
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
    if args.isTrain:
        logger.info('Training starts.')
        train(args)
    else:
        logger.info('Test starts.')
        model = torch.load(args.resumePath)
        test(model, args)
    
