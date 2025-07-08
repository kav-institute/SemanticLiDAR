
import torch
import time
import os
import json
import numpy as np
from tqdm import tqdm
import cv2

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
        visualization[mask == int(class_id)] = color

    return visualization

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Tester:
    def __init__(self, model, save_path, config, load=False, visualize=True, test_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        self.visualize = visualize
        self.model = model
        if load:
            self.model.load_state_dict(torch.load(save_path))

        self.config = config
        #self.normals = self.config["USE_NORMALS"]
        self.num_classes = self.config["NUM_CLASSES"]
        self.class_names = self.config["CLASS_NAMES"]
        self.class_colors = self.config["CLASS_COLORS"]

        self.save_path = os.path.dirname(save_path)
        time.sleep(3)

        # Timer
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        
        # Evaluator
        self.evaluator = SemanticSegmentationEvaluator(num_classes=self.num_classes, test_mask=test_mask)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    
    def __call__(self, dataloader_test):
        self.model.eval()
        self.evaluator.reset()
        for batch_idx, (range_img, reflectivity, xyz, normals, semantic)  in enumerate(dataloader_test):
            range_img, reflectivity, xyz, normals, semantic = range_img.to(self.device), reflectivity.to(self.device), xyz.to(self.device), normals.to(self.device), semantic.to(self.device)
            start_time = time.time()

            # run forward path
            # run forward path
            start_time = time.time()
            self.start.record()

            outputs_semantic = self.model(torch.cat([range_img, reflectivity, xyz],axis=1))

            self.end.record()
            curr_time = (time.time()-start_time)*1000
    
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
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
                cv2.waitKey(1)

            self.evaluator.update(semseg_img, semantic)
        mIoU, result_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        with open(os.path.join(self.save_path, "results.json"), 'w') as fp:
            json.dump(result_dict, fp, cls=MyEncoder)
 

    
    

