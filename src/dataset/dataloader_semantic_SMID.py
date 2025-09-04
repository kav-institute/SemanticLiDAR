from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import open3d as o3d
try:
    from dataset.utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
    from dataset.definitions import id_map
except:
    from utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
    from definitions import id_map, custom_colormap
import cv2

id_map = {
    0: 0,
    1: 0,
    101: 1,
    102: 2,
    103: 3,
    104: 3,
    105: 1,
    106: 4,
    107: 5,
    108: 0,
    109: 6,
    110: 7,
    111: 8,
    112: 9,
    116: 10,
    117: 11,
    118: 0,
    119: 12,
    120: 13,
    121: 14,
    122: 0,
    200: 0,
    201: 0,
    202: 0,
    203: 0,
    204: 3,
}

class SemanticKitti(Dataset):
    def __init__(self, data_path, rotate=False, flip=False, resolution=(2048,128), projection=(64,2048), resize=True):
        self.data_path = data_path
        self.rotate = rotate
        self.flip = flip
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # Add more transformations if needed
        ])
        self.resolution = resolution
        self.projection = projection
        self.resize = resize

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        frame_path, label_path = self.data_path[idx]
        # the (x, y, z, intensity) are stored in binary
        xyzi = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4) 

        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        
        # get semantic labels
        sem_label = label & 0xFFFF

        #get instance labels
        inst_label = label >> 16
        

        sem_label_map = np.array([id_map[l] for l in sem_label])

        xyzil = np.concatenate([xyzi, sem_label_map[...,np.newaxis]],axis=-1)


        # Augmentations
        
        if self.rotate:
            random_angle = np.random.randint(-180,180)
            xyzil[...,0:3] = rotate_z(xyzil[...,0:3].reshape(-1,3), random_angle)
        xyzi_img, _, _ , _ = spherical_projection(xyzil,self.projection[0],self.projection[1], theta_range=[-0.26, 0.78])
        if self.resize:
            xyzi_img = cv2.resize(xyzi_img, (2048,128), interpolation=cv2.INTER_NEAREST)
        

        if self.flip:
            do_flip = np.random.choice([True, False])
            if do_flip:
                xyzi_img = xyzi_img[:,::-1,:]
                xyzi_img[...,1] = -xyzi_img[...,1]

        label_img = xyzi_img[...,4:5]
        reflectivity_img = xyzi_img[...,3]/xyzi_img[...,3].max()
        xyzi_img = xyzi_img[...,0:3]


        range_img = np.linalg.norm(xyzi_img,axis=-1)
        
        normals = build_normal_xyz(xyzi_img[...,0:3])
        
        label_img = label_img

        semantics =  label_img

        reflectivity_img =  torch.as_tensor(reflectivity_img[...,None].transpose(2, 0, 1).astype("float32"))
        range_img =  torch.as_tensor(range_img[...,None].transpose(2, 0, 1).astype("float32"))
        xyz =  torch.as_tensor(xyzi_img[...,0:3].transpose(2, 0, 1).astype("float32"))

        normals =  torch.as_tensor(normals.transpose(2, 0, 1).astype("float32"))

        semantics =  torch.as_tensor(semantics.transpose(2, 0, 1).astype("long"))
        
        return range_img, reflectivity_img, xyz, normals, semantics

def main():
    import glob
    import open3d as o3d
    import cv2 

    data_path_test = [(bin_path, bin_path.replace("hybrid", "labels").replace("bin", "label")) for bin_path in sorted(glob.glob(f"/home/appuser/data/SMID_v1_3/sequences/02/hybrid/*.bin"))]

    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False, projection=(40,512),resize=False)
    dataloader_test = DataLoader(depth_dataset_test, batch_size=1, shuffle=False)

    for batch_idx, (range_img, reflectivity, xyz, normals, semantic)  in enumerate(dataloader_test):
        semantics = (semantic[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
        reflectivity = (reflectivity[:,0,:,:]).permute(0, 1, 2)[0,...].cpu().detach().numpy()
        normal_img = (normals.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()+1)/2
        prev_sem_pred = cv2.applyColorMap(np.uint8(semantics), custom_colormap)
        xyz = xyz.permute(0, 2, 3, 1)[0,...].cpu().detach().numpy()
        cv2.imshow("semseg", prev_sem_pred[...,::-1])
        cv2.imshow("normal_img", normal_img)
        cv2.imshow("reflectivity", cv2.applyColorMap(np.uint8(255*reflectivity),cv2.COLORMAP_JET))

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            

            #time.sleep(10)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1,3))
            pcd.colors = o3d.utility.Vector3dVector(np.float32(prev_sem_pred[...,::-1].reshape(-1,3))/255.0)

            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([mesh, pcd])

if __name__ == "__main__":
    main()
