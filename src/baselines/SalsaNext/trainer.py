
import torch


import time
import numpy as np
import cv2
import os
import open3d as o3d
import tqdm

from torch.utils.tensorboard import SummaryWriter

import sys
import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
from models.losses import TverskyLoss, SemanticSegmentationLoss
from models.evaluator import SemanticSegmentationEvaluator


def visualize_semantic_segmentation_cv2(mask, class_colors):
    """
    Visualize semantic segmentation mask using class colors with cv2.

    Parameters:
    - mask: 2D NumPy array containing class IDs for each pixel.
    - class_colors: Dictionary mapping class IDs to BGR colors.

    Returns:
    - visualization: Colored semantic segmentation image in BGR format.
    """
    h, w = mask.shape
    visualization = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        visualization[mask == class_id] = color

    return visualization


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, model, optimizer, save_path, config, scheduler= None, visualize = False):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        time.sleep(3)

        self.visualize = visualize

        # config
        self.config = config
        self.num_classes = self.config["NUM_CLASSES"]
        self.class_names = self.config["CLASS_NAMES"]
        self.class_colors = self.config["CLASS_COLORS"]

        # TensorBoard
        self.save_path = save_path
        self.writer = SummaryWriter(save_path)

        # Timer
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        # Loss
        self.criterion_dice = TverskyLoss()
        self.criterion_semantic = SemanticSegmentationLoss()
        
        # Evaluator
        self.evaluator = SemanticSegmentationEvaluator(self.num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_one_epoch(self, dataloder, epoch):
        self.model.train()
        total_loss = 0.0
        # train one epoch
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic) in enumerate(tqdm.tqdm(dataloder, desc=f"Epoch {epoch + 1}")):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(self.device), reflectivity.to(self.device), xyz.to(self.device), normals.to(self.device), semantic.to(self.device)
    
            # run forward path
            start_time = time.time()
            self.start.record()

            outputs_semantic = self.model(torch.cat([range_img, reflectivity, xyz],axis=1))

            self.end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # get losses
            loss_semantic = self.criterion_semantic(outputs_semantic, semantic, num_classes=self.num_classes)
            loss_dice = self.criterion_dice(outputs_semantic, semantic, num_classes=self.num_classes, alpha=0.9, beta=0.1)

            loss = loss_dice+loss_semantic 
            #loss = loss_dice+loss_semantic
            
            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)
            
            if self.visualize:
                # visualize first sample in batch
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)
                cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):

                    #time.sleep(10)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(prev_sem_pred[...,::-1].reshape(-1,3))/255.0)
        
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd])
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            total_loss += loss.item()
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                step = epoch * len(dataloder) + batch_idx
                self.writer.add_scalar('Loss', loss.item(), step)
                self.writer.add_scalar('Semantic_Loss', loss_semantic.item(), step)
                self.writer.add_scalar('Dice_Loss', loss_dice.item(), step)
        
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloder)
        self.writer.add_scalar('Loss_EPOCH', avg_loss, epoch)
        print(f"Train Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss}")

    def test_one_epoch(self, dataloder, epoch):
        inference_times = []
        self.model.eval()
        self.evaluator.reset()
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic)  in enumerate(dataloder):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(self.device), reflectivity.to(self.device), xyz.to(self.device), normals.to(self.device), semantic.to(self.device)
            start_time = time.time()

            # run forward path
            start_time = time.time()
            self.start.record()
            outputs_semantic = self.model(torch.cat([range_img, reflectivity, xyz],axis=1))
            self.end.record()
            curr_time = (time.time()-start_time)*1000
            
            # Waits for everything to finish running
            torch.cuda.synchronize()

            # log inference times
            inference_times.append(self.start.elapsed_time(self.end))
            
            outputs_semantic_argmax = torch.argmax(outputs_semantic,dim=1)

            self.evaluator.update(outputs_semantic_argmax, semantic)
        mIoU, result_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        for cls in range(self.num_classes):
            self.writer.add_scalar('IoU_{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            

        self.writer.add_scalar('mIoU_Test', mIoU*100, epoch)
        self.writer.add_scalar('Inference Time', np.median(inference_times), epoch)
        print(f"Test Epoch {epoch + 1}/{self.num_epochs}, mIoU: {mIoU}")
        return mIoU
    
    def __call__(self, dataloder_train, dataloder_test, num_epochs=50, test_every_nth_epoch=1, save_every_nth_epoch=-1):
        self.num_epochs = num_epochs
        for epoch in range(num_epochs):
            # train one epoch
            self.train_one_epoch(dataloder_train, epoch)
            # test
            if epoch > 0 and epoch % test_every_nth_epoch == 0:
                mIoU = self.test_one_epoch(dataloder_test, epoch)
                # update scheduler based on rmse
                if not isinstance(self.scheduler, type(None)):
                    self.scheduler.step(mIoU)
            # save
            if save_every_nth_epoch >= 1 and epoch % save_every_nth_epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_{}.pth".format(str(epoch).zfill(6))))

            
        # run final test
        self.test_one_epoch(dataloder_test, epoch)
        # save last epoch
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pth"))

