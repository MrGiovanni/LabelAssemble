from config import *
import datasets
import copy


def build_dataset_helper(args, aug, start_id, source):
    if args.datasetType == 'covidx':
        return datasets.COVIDX(COVIDXConfig, mode=args.mode, source=source, augment=aug, start_id=start_id, num_classes=args.numClass), COVIDXConfig['class_num']
    elif args.datasetType == 'chestxray14':
        return datasets.ChestXRay14(ChestXray14Config, args.mode, source=source, augment=aug, start_id=start_id, num_classes=args.numClass), ChestXray14Config['class_num']
    else:
        raise NotImplementedError
        


def build_dataset(args, weak_aug=None, strong_aug=None):
    if args.datasetType == 'assemble':
        args_temp = copy.deepcopy(args)
        datasets_assembling = []
        start_id = 0
        source = 0
        for assemble_dataset in assemble_datasets:
            args_temp.datasetType = assemble_dataset
            train_set, class_used = build_dataset_helper(args_temp, weak_aug, start_id, source)  
            datasets_assembling.append(train_set)
            source += 1
            start_id += class_used
        train_set = datasets.Assemble(datasets_assembling, augments=[weak_aug, strong_aug])
    else:
        train_set, _ = build_dataset_helper(args.datasetType, weak_aug, 0, 0)  
    return train_set