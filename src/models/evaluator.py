import torch
import numpy as np

class SemanticSegmentationEvaluator:

    def __init__(self, num_classes, test_mask=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        self.use_th = False
        if all(x == 1 for x in test_mask):
            self.use_th = True
        
        self.num_classes = num_classes
        self.test_mask = test_mask
        self.reset()

    def reset(self):
        self.accuracy = 0
        self.divisor = 0
        self.intersection_per_class = torch.zeros(self.num_classes)
        self.union_per_class = torch.zeros(self.num_classes)
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def update(self, outputs, targets):
        """Update metrics with a new batch of data."""
        self.accuracy += torch.mean(torch.where(outputs == targets, 1.0, 0.0)).item()
        self.divisor += 1
        intersection, union = self.compute_scores(outputs, targets)
        self.intersection_per_class += intersection
        self.union_per_class += union
        self.update_confusion_matrix(outputs, targets)

    def compute_scores(self, outputs, targets):
        intersection_per_class = torch.zeros(self.num_classes)
        union_per_class = torch.zeros(self.num_classes)
        

        for cls in range(self.num_classes):
            pred_cls = (outputs == cls).float()
            target_cls = (targets == cls).float()

            intersection_per_class[cls] = self.test_mask[cls] * (pred_cls * target_cls).sum()
            union_per_class[cls] = self.test_mask[cls] * ((pred_cls + target_cls).sum() - intersection_per_class[cls])

        return intersection_per_class, union_per_class

    def update_confusion_matrix(self, outputs, targets):
        """Update confusion matrix with current batch."""
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += torch.sum((outputs == j) & (targets == i)).item()

    def compute_final_metrics(self, class_names, reduce="mean", ignore_th=0.1):
        """Compute final metrics after processing all batches."""
        return_dict = {}
        iou_per_class = torch.zeros(self.num_classes)
        for cls in range(self.num_classes):
            if self.union_per_class[cls] == 0:
                iou_per_class[cls] = float('nan')
            else:
                iou_per_class[cls] = float(self.intersection_per_class[cls]) / float(self.union_per_class[cls])
            if self.test_mask[cls] == 0:
                iou_per_class[cls] = float('nan')

            try:
                return_dict[class_names[cls]] = iou_per_class[cls].item()
            except:
                return_dict[class_names[str(cls)]] = iou_per_class[cls].item()

        if self.use_th:
            if reduce == "mean":
                mIoU = np.nanmean(np.where(iou_per_class.numpy() < ignore_th, np.NaN, iou_per_class.numpy()))
            elif reduce == "median":
                mIoU = np.nanmedian(np.where(iou_per_class.numpy() < ignore_th, np.NaN, iou_per_class.numpy()))
            else:
                raise NotImplementedError
        else:
            if reduce == "mean":
                mIoU = np.nanmean(iou_per_class.numpy())
            elif reduce == "median":
                mIoU = np.nanmedian(iou_per_class.numpy())
            else:
                raise NotImplementedError

        return_dict["mIoU"] = mIoU
        return_dict["Acc"] = self.accuracy/self.divisor
        return mIoU, return_dict

    def get_confusion_matrix(self):
        """Return confusion matrix."""
        return self.confusion_matrix
