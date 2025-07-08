import glob
import torch
from torch.utils.data import DataLoader

import torch.optim as optim


import os
import argparse
import json

import sys
import os.path as osp
from SalsaNext import SalsaNextUncertainty
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))
from dataset.dataloader_semantic_THAB import SemanticKitti
from dataset.definitions import color_map, class_names, custom_colormap


from trainer import Trainer
from tester import Tester

def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main(args):
    config = {}
    config["NORMALS"] = args.normals
    config["TEST_SCENE"] = args.test_id
    config["NUM_CLASSES"] = 20
    config["CLASS_NAMES"] = class_names
    config["CLASS_COLORS"] = color_map
    config["NUM_EPOCHS"] = args.num_epochs
    config["BATCH_SIZE"] = args.batch_size
    config["TEST_MASK"] = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    num_folder = count_folders("/home/appuser/data/SemanticTHAB/sequences/")

    if args.test_id == -1:
        data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder)]  for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
        data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i == 2] for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
    else:
        data_path_train = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i != args.test_id] for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
        data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for folder in [str(i).zfill(4) for i in range(num_folder) if i == args.test_id] for bin_path in glob.glob(f"/home/appuser/data/SemanticTHAB/sequences/{folder}/velodyne/*.bin")]
    

    if args.test_id == 6:
        config["TEST_MASK"] = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    else:
        config["TEST_MASK"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    depth_dataset_train = SemanticKitti(data_path_train, rotate=False, flip=True)
    dataloader_train = DataLoader(depth_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Network definition
    model = SalsaNextUncertainty(nclasses=config["NUM_CLASSES"])
    num_params = count_parameters(model)
    config["NUM_PARAMS"] = num_params
    print("num_params", count_parameters(model))
        
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1)
        
    # Save Path
    if args.test_id != -1:
        save_path_p1 ='/home/appuser/data/train_semantic_THAB_v2/test_split_{}/'.format(str(args.test_id).zfill(4))
    else:
        save_path_p1 ='/home/appuser/data/train_semantic_THAB_v2/test_split_{}/'.format("final")
    

    save_path_p2 ='SalsaNext/'
    save_path = os.path.join(save_path_p1,save_path_p2)
    os.makedirs(save_path, exist_ok=True)

    config["SAVE_PATH"] = save_path

    save_path = os.path.dirname(config["SAVE_PATH"])
    # write config file
    with open(os.path.join(save_path, "config.json"), 'w') as fp:
        json.dump(config, fp)

    # train model
    trainer = Trainer(model, optimizer, save_path, config, scheduler = scheduler, visualize = True)
    trainer(dataloader_train, dataloader_test, args.num_epochs)

    # test final model
    tester = Tester(model, save_path=os.path.join(save_path, "model_final.pth"), config=config, load=False, test_mask=config["TEST_MASK"])
    tester(dataloader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train script for SemanticTHAB')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model (default: 0.001)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs for training (default: 50)')
    parser.add_argument('--test_every_nth_epoch', type=int, default=1,
                        help='Test every nth epoch (default: 1)')
    parser.add_argument('--test_id', type=int, default=6,
                        help='Test ID of the test sequence for the leave one out CV. -1 for training on all')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--rotate', action='store_true',
                        help='Whether to apply rotation augmentation (default: False)')
    parser.add_argument('--normals', action='store_true',
                        help='Whether to normals as input (default: False)')
    parser.add_argument('--flip', action='store_true',
                        help='Whether to apply flip augmentation (default: False)')
    parser.add_argument('--visualization', action='store_true',
                        help='Toggle visualization during training (default: False)')
    args = parser.parse_args()

    main(args)
