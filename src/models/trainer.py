
import torch

from models.losses import TverskyLoss, SemanticSegmentationLoss, LovaszSoftmax, DirichletSegmentationLoss

import time
import numpy as np
import cv2
import os
import open3d as o3d
import tqdm

from torch.utils.tensorboard import SummaryWriter
from models.evaluator import SemanticSegmentationEvaluator
from models.evaluator_unc import SemanticSegmentationUncEvaluator
from torch.special import digamma

def add_colorbar_with_ticks(image, max_depth=80, colormap=cv2.COLORMAP_TURBO, width=60, num_ticks=5, font_scale=0.3, thickness=1, color=(125, 125, 125)):
    """
    Adds a vertical colorbar with ticks and labels to the right of the image.
    
    Args:
        image: Colored depth image (H, W, 3) as BGR numpy array
        max_depth: Maximum depth value (e.g., 80 meters)
        colormap: OpenCV colormap to use
        width: Width of the colorbar in pixels
        num_ticks: Number of tick labels (e.g., 5 → 0, 20, 40, 60, 80)
        font_scale: Font size for labels
        thickness: Line and text thickness

    Returns:
        Concatenated image with labeled colorbar
    """
    height = image.shape[0]

    # Generate gradient for colorbar (top = max_depth, bottom = 0)
    gradient = np.linspace(max_depth, 0, height).astype(np.float32).reshape(-1, 1)
    gradient_norm = np.clip((gradient / max_depth) * 255.0, 0, 255).astype(np.uint8)
    colorbar = cv2.applyColorMap(cv2.resize(gradient_norm, (width, height)), colormap)

    # Add ticks and labels
    bar_with_ticks = colorbar.copy()
    for i in range(1,num_ticks):
        y = int(i * height / num_ticks)
        value = max_depth - (i * max_depth / num_ticks)
        label = f"{int(value)} m"

        # Draw tick mark
        cv2.line(bar_with_ticks, (0, y), (10, y), color=color, thickness=thickness)

        # Put label (right-justified)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = 15  # fixed padding from tick
        text_y = y + int(text_size[1] / 2)
        cv2.putText(bar_with_ticks, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Concatenate to the right of the image
    return np.concatenate((image, bar_with_ticks), axis=1)


def get_aleatoric_uncertainty(alpha, eps=1e-10, n_classes=20):
    """
    Computes aleatoric uncertainty from Dirichlet parameters.
    From paper: Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)
    https://arxiv.org/abs/1910.04819 
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], aleatoric uncertainty of expected class probabilities
    """
    proba_in = (alpha / alpha.sum(1, keepdim=True)).clamp_(1e-8, 1-1e-8)
    entropy = - torch.sum((proba_in * proba_in.log()), dim=1)
    normalized_entropy = entropy / np.log(n_classes)
    return normalized_entropy

def get_epistemic_uncertainty(alpha, eps=1e-10, n_classes = 20):
    """
    Computes epistemic uncertainty from Dirichlet parameters.
    From paper: Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)
    https://arxiv.org/abs/1910.04819 
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], epistemic uncertainty of expected class probabilities
    """
    eu = get_predictive_entropy(alpha) - get_aleatoric_uncertainty(alpha)   # Epistemic uncertainty = Total entropy - aleatoric uncertainty
    return eu

def get_epistemic_uncertainty_v2(alpha, eps=1e-10, n_classes = 20):
    """
    Computes epistemic uncertainty from Dirichlet parameters.
    From paper: Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)
    https://arxiv.org/abs/1910.04819 
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], epistemic uncertainty of expected class probabilities
    """
    
    return (n_classes/ alpha.sum(1))#/ np.log(n_classes)

def get_predictive_entropy(alpha, eps=1e-10, n_classes = 20):
    """
    Computes predictive entropy H(E[p]) from Dirichlet parameters.
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], entropy of expected class probabilities
    """
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True) + eps               # Total concentration alpha_0
    Exp_prob = alpha / alpha_0                                          # Expected class probabilities
    entropy = -torch.sum(Exp_prob * torch.log(Exp_prob + eps), dim=1)   # Entropy across classes
    return entropy/ np.log(n_classes)


def predictive_entropy(alpha, eps=1e-10):
    """
    Computes predictive entropy H(E[p]) from Dirichlet parameters.
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], entropy of expected class probabilities
    """
    S = torch.sum(alpha, dim=1, keepdim=True)       # Total concentration α₀
    p = alpha / (S + eps)                           # Expected class probabilities
    entropy = -torch.sum(p * torch.log(p + eps), dim=1)  # Entropy across classes
    return entropy

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

def add_horizontal_uncertainty_colorbar(image, max_uncertainty = np.log(20), colormap=cv2.COLORMAP_TURBO, height=20,
                                        num_ticks=5, font_scale=0.7, thickness=1, color=(225, 225, 225)):
    """
    Adds a horizontal colorbar for uncertainty values (0 to log(num_classes)) with ticks and labels below the image.

    Args:
        image: Colored uncertainty image (H, W, 3) as BGR numpy array
        num_classes: Number of classes to compute max uncertainty = log(num_classes)
        colormap: OpenCV colormap to use
        height: Height of the colorbar in pixels
        num_ticks: Number of tick labels (e.g., 5 → 0, 0.5, 1, 1.5, 2)
        font_scale: Font size for labels
        thickness: Line and text thickness

    Returns:
        Concatenated image with labeled horizontal colorbar below
    """
    
    width = image.shape[1]

    # Generate horizontal gradient (left = 0, right = max_uncertainty)
    gradient = np.linspace(0, max_uncertainty, width).astype(np.float32).reshape(1, -1)
    gradient_norm = np.clip((gradient / max_uncertainty) * 255.0, 0, 255).astype(np.uint8)
    gradient_resized = cv2.resize(gradient_norm, (width, height), interpolation=cv2.INTER_LINEAR)
    colorbar = cv2.applyColorMap(gradient_resized, colormap)

    bar_with_ticks = colorbar.copy()

    # Draw ticks and labels along width (x-axis)
    text_labels = ["Certain", "Confident", "Ambiguous", "Doubtful", "Uncertain"]
    for i in range(5):
        x = int(i * (width - 1) / (num_ticks - 1))
        value = i * max_uncertainty / (num_ticks - 1)
        label = text_labels[i]

        # Draw vertical tick mark
        #cv2.line(bar_with_ticks, (x, 0), (x, 20), color=color, thickness=thickness)

        # Get text size for horizontal centering
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if i <= 2:
            text_x = x #+ text_size[0]#// 2  # center label horizontally on tick
        elif i == 2:
            text_x = x #- text_size[0] // 2  # center label horizontally on tick
        else:
            text_x = x - text_size[0] #// 2  # center label horizontally on tick
        text_y = text_size[1]  # below tick mark

        # Put label text
        cv2.putText(bar_with_ticks, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Concatenate colorbar below the image
    return np.concatenate((image, bar_with_ticks), axis=0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Trainer:
    def __init__(self, model, optimizer, save_path, config, scheduler= None, visualize = False, test_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        time.sleep(3)

        self.visualize = visualize

        # config
        self.config = config
        self.normals = self.config["USE_NORMALS"]==True
        self.use_reflectivity = self.config["USE_REFLECTIVITY"]==True
        self.num_classes = self.config["NUM_CLASSES"]
        self.class_names = self.config["CLASS_NAMES"]
        self.class_colors = self.config["CLASS_COLORS"]

        self.loss_function = self.config["LOSS_FUNCTION"]
        # TensorBoard
        self.save_path = save_path
        self.writer = SummaryWriter(save_path)

        # Timer
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        # Loss
        if  self.loss_function == "Tversky":
            self.criterion_dice = TverskyLoss()
            self.criterion_semantic = SemanticSegmentationLoss()
        elif self.loss_function == "CE":
            self.criterion_semantic = SemanticSegmentationLoss()
        elif  self.loss_function == "Lovasz":
            self.criterion_lovasz = LovaszSoftmax()
        elif self.loss_function == "Dirichlet":
            self.criterion_unc = DirichletSegmentationLoss()
        else:
            raise NotImplementedError

        # Evaluator
        if self.loss_function == "Dirichlet":
            self.evaluator = SemanticSegmentationUncEvaluator(self.num_classes, test_mask=test_mask)
        else:
            self.evaluator = SemanticSegmentationEvaluator(self.num_classes, test_mask=test_mask)

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
            if self.use_reflectivity:
                input_img = torch.cat([range_img, reflectivity],axis=1)
            else:
                input_img = range_img
            if self.normals:
                outputs_semantic = self.model(input_img, torch.cat([xyz, normals],axis=1))
            else:
                outputs_semantic = self.model(input_img, xyz)
            self.end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            # get losses
            if  self.loss_function == "Tversky":
                loss_semantic = self.criterion_semantic(outputs_semantic, semantic, num_classes=self.num_classes)
                loss_dice = self.criterion_dice(outputs_semantic, semantic, num_classes=self.num_classes, alpha=0.9, beta=0.1)
                loss = loss_dice+loss_semantic
            elif  self.loss_function == "CE":
                loss = self.criterion_semantic(outputs_semantic, semantic, num_classes=self.num_classes)
            elif  self.loss_function == "Lovasz":
                loss = self.criterion_lovasz(outputs_semantic, semantic)
            elif self.loss_function == "Dirichlet":
                loss = self.criterion_unc(outputs_semantic, semantic)
            else:
                raise NotImplementedError
            
            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)
            semseg_img_v2 = semseg_img

            if self.visualize:
                if self.loss_function == "Dirichlet":
                    predictive = get_predictive_entropy(torch.nn.functional.softplus(outputs_semantic)+1)
                    semseg_img_v2 = torch.where(predictive>0.75, 21, semseg_img)
                    aleatoric = get_aleatoric_uncertainty(torch.nn.functional.softplus(outputs_semantic)+1)
                
                    epistemic = get_epistemic_uncertainty(torch.nn.functional.softplus(outputs_semantic)+1)
                    #predictive = epistemic + aleatoric
                    #epistemic = predictive-aleatoric
                    epistemic = (epistemic).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    aleatoric = (aleatoric).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    predictive = (predictive).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    #predictive = np.minimum(predictive, 1)
                    pred_unc_img = cv2.applyColorMap(np.uint8(255*predictive), cv2.COLORMAP_TURBO)
                    #epis_unc_img = cv2.applyColorMap(np.uint8(255*np.maximum((epistemic/np.log(self.num_classes)),0.0)), cv2.COLORMAP_TURBO)
                    epis_unc_img = cv2.applyColorMap(np.uint8(255*epistemic), cv2.COLORMAP_TURBO)
                    alea_unc_img = cv2.applyColorMap(np.uint8(255*aleatoric), cv2.COLORMAP_TURBO)
                    #print("Unc",np.min(epistemic), np.min(aleatoric), np.max(predictive))
                    cv2.imshow("pred_unc_img", add_horizontal_uncertainty_colorbar(pred_unc_img))
                    #cv2.imshow("epis_unc_img", add_horizontal_uncertainty_colorbar(epis_unc_img))
                    #cv2.imshow("alea_unc_img", add_horizontal_uncertainty_colorbar(alea_unc_img))
                # visualize first sample in batch
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                error_img = np.uint8(np.where(semantics_pred[...,None]!=semantics_gt[...,None], (0,0,255), (0,0,0)))
                semantics_pred_v2 = (semseg_img_v2).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred_v2, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)

                
                cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt,error_img)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):

                    #time.sleep(10)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    
                    #prev_sem_pred[...,0] = c
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
                #self.writer.add_scalar('Semantic_Loss', loss_semantic.item(), step)
                #self.writer.add_scalar('Dice_Loss', loss_dice.item(), step)
        
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(dataloder)
        self.writer.add_scalar('Loss_EPOCH', avg_loss, epoch)
        print(f"Train Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {avg_loss}")

    def test_one_epoch(self, dataloder, epoch):
        inference_times = []
        self.model.eval()
        self.evaluator.reset()
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic)  in enumerate(tqdm.tqdm(dataloder, desc=f"Testing Epoch {epoch + 1}")):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(self.device), reflectivity.to(self.device), xyz.to(self.device), normals.to(self.device), semantic.to(self.device)
            start_time = time.time()

            # run forward path
            start_time = time.time()
            self.start.record()
            if self.use_reflectivity:
                input_img = torch.cat([range_img, reflectivity],axis=1)
            else:
                input_img = range_img
            if self.normals:
                outputs_semantic = self.model(input_img, torch.cat([xyz, normals],axis=1))
            else:
                outputs_semantic = self.model(input_img, xyz)
            self.end.record()
            curr_time = (time.time()-start_time)*1000
            
            # Waits for everything to finish running
            torch.cuda.synchronize()

            # log inference times
            inference_times.append(self.start.elapsed_time(self.end))
            
            outputs_semantic_argmax = torch.argmax(outputs_semantic,dim=1)

            # get the most likely class
            semseg_img = torch.argmax(outputs_semantic,dim=1)
            

            if self.visualize:
                if self.loss_function == "Dirichlet":
                    predictive = get_predictive_entropy(torch.nn.functional.softplus(outputs_semantic)+1)
                    
                    #epistemic = predictive-aleatoric
                    #epistemic = (epistemic).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    #aleatoric = (aleatoric).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    predictive = (predictive).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                    predictive = np.minimum(predictive, 1)
                    pred_unc_img = cv2.applyColorMap(np.uint8(255*predictive), cv2.COLORMAP_TURBO)
                    #epis_unc_img = cv2.applyColorMap(np.uint8(255*np.maximum((epistemic/np.log(self.num_classes)),0.0)), cv2.COLORMAP_TURBO)
                    #epis_unc_img = cv2.applyColorMap(np.uint8(255*epistemic), cv2.COLORMAP_TURBO)
                    #alea_unc_img = cv2.applyColorMap(np.uint8(255*aleatoric), cv2.COLORMAP_TURBO)
                    #print("Unc",np.min(epistemic), np.min(aleatoric), np.max(predictive))
                    cv2.imshow("pred_unc_img", add_horizontal_uncertainty_colorbar(pred_unc_img))
                    #cv2.imshow("epis_unc_img", add_horizontal_uncertainty_colorbar(epis_unc_img))
                    #cv2.imshow("alea_unc_img", add_horizontal_uncertainty_colorbar(alea_unc_img))
                # visualize first sample in batch
                semantics_pred = (semseg_img).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                semantics_gt = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
                error_img = np.uint8(np.where(semantics_pred[...,None]!=semantics_gt[...,None], (0,0,255), (0,0,0)))
                xyz_img = (xyz).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
                reflectivity_img = (reflectivity).permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
                reflectivity_img = np.uint8(255*np.concatenate(3*[reflectivity_img],axis=-1))
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)

                cv2.imshow("inf", np.vstack((reflectivity_img,np.uint8(255*normal_img),prev_sem_pred,prev_sem_gt,error_img)))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):

                    #time.sleep(10)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_img.reshape(-1,3))
                    pcd.colors = o3d.utility.Vector3dVector(np.float32(prev_sem_pred[...,::-1].reshape(-1,3))/255.0)
        
                    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                    o3d.visualization.draw_geometries([mesh, pcd])

            # get the most likely class
            if self.loss_function == "Dirichlet":
                self.evaluator.update(outputs_semantic, semantic)
            else:
                self.evaluator.update(outputs_semantic_argmax, semantic)
        mIoU, result_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        for cls in range(self.num_classes):
            self.writer.add_scalar('IoU_{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            if self.loss_function == "Dirichlet":
                self.writer.add_scalar('entropy_{}'.format(self.class_names[cls]), result_dict["entropy_{}".format(self.class_names[cls])], epoch)
        if self.loss_function == "Dirichlet":
            self.writer.add_scalar('bountry_entropy', result_dict["bountry_entropy"], epoch)
        self.writer.add_scalar('mIoU_Test', mIoU*100, epoch)
        self.writer.add_scalar('Inference Time', np.median(inference_times), epoch)
        print(f"Test Epoch {epoch + 1}/{self.num_epochs}, mIoU: {mIoU}, Acc: {result_dict['Acc']}")
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
        #self.test_one_epoch(dataloder_test, epoch)
        # save last epoch
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pth"))

