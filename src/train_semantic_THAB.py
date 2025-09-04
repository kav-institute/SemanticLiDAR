import glob
import argparse
import torch
from torch.utils.data import DataLoader
from dataset.dataloader_semantic_KITTI import SemanticKitti
from models.semanticFCN import SemanticNetworkWithFPN

import torch.optim as optim

import os

from dataset.definitions import color_map, class_names

from models.tester import Tester
from models.trainer import Trainer
import json


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    config = {}
    config["BACKBONE"] = args.model_type
    config["USE_REFLECTIVITY"] = True
    config["USE_ATTENTION"] = args.attention
    config["USE_NORMALS"] = args.normals
    config["USE_MULTI_SCALE"] = args.multi_scale_meta
    config["NUM_CLASSES"] = 20
    config["CLASS_NAMES"] = class_names
    config["CLASS_COLORS"] = color_map
    config["NUM_EPOCHS"] = args.num_epochs
    config["BATCH_SIZE"] = args.batch_size
    config["LOSS_FUNCTION"] = "Tversky"
    
    data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [f"{i:02}" for i in range(11) if i != 8] for bin_path in glob.glob(f"/home/appuser/data/SemanticKitti/dataset/sequences/{folder}/velodyne/*.bin")]
    data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in glob.glob(f"/home/appuser/data/SemanticKitti/dataset/sequences/08/velodyne/*.bin")]

    #depth_dataset_train = SemanticKitti(data_path_train, rotate=args.rotate, flip=args.flip, projection=(64,512),resize=False)
    depth_dataset_train = SemanticKitti(data_path_train, rotate=True, flip=True, projection=(64,512),resize=False)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False, projection=(64,512),resize=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=8)
    
    test_mask = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    if config["USE_REFLECTIVITY"]:
        input_channels = 2
    else:
        input_channels = 1

    # Semantic Segmentation Network
    if args.normals:
        model = SemanticNetworkWithFPN(backbone=args.model_type, input_channels=input_channels, meta_channel_dim=6, num_classes=config["NUM_CLASSES"], attention=args.attention, multi_scale_meta=args.multi_scale_meta)
    else:
        model = SemanticNetworkWithFPN(backbone=args.model_type, input_channels=input_channels, meta_channel_dim=3, num_classes=config["NUM_CLASSES"], attention=args.attention, multi_scale_meta=args.multi_scale_meta)

    num_params = count_parameters(model)
    config["NUM_PARAMS"] = num_params
    print("num_params", count_parameters(model))
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1)

    # Save Path
    save_path ='/home/appuser/data/train_semantic_kitti/{}_{}{}{}/'.format(args.model_type, "a" if args.attention else "", "n" if args.normals else "", "m" if args.multi_scale_meta else "")
    
    os.makedirs(save_path, exist_ok=True)

    config["SAVE_PATH"] = save_path

    save_path = os.path.dirname(config["SAVE_PATH"])
    # write config file
    with open(os.path.join(save_path, "config.json"), 'w') as fp:
        json.dump(config, fp)

    # train model
    test_mask = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    trainer = Trainer(model, optimizer, save_path, config, scheduler = scheduler, visualize = True, test_mask=test_mask)
    trainer(dataloader_train, dataloader_test, args.num_epochs, test_every_nth_epoch=args.test_every_nth_epoch)

    # test final model
    tester = Tester(model, save_path=os.path.join(save_path, "model_final.pth"), config=config, load=False, test_mask=test_mask)
    tester(dataloader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticKitti')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        help='Type of the model to be used (default: resnet34)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--test_every_nth_epoch', type=int, default=5,
                        help='Test every nth epoch (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--attention', action='store_true',
                        help='Whether to use attention (default: False)')
    parser.add_argument('--normals', action='store_true',
                        help='Whether to normals as input (default: False)')
    parser.add_argument('--multi_scale_meta', action='store_true',
                        help='Whether to to inject meta data at multiple scales (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)
