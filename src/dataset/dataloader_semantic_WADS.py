from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
try:
    from dataset.utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
    from dataset.definitions import id_map
except:
    from utils import rotate_equirectangular_image, rotate_z, build_normal_xyz, spherical_projection
    from definitions import id_map, custom_colormap
import cv2

id_map = {
  0 : 0,     # "unlabeled"
  1 : 0,    # "outlier" mapped to "unlabeled" --------------------------mapped
  9: 0,
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 19,     # "lane-marking" to "traffic-sign" ------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  110: 0,    # "falling-snow" to "unlabeled" ----------------------------mapped
  111: 0,    # "accumulated-snow" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}


color_map = {
  0 : [0, 0, 0],
  1 : [245, 150, 100],
  2 : [245, 230, 100],
  3 : [150, 60, 30],
  4 : [180, 30, 80],
  5 : [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [125,125,125],#[255, 0, 255],
  10: [255, 150, 255],
  11: [75, 0, 75],
  12: [75, 0, 175],
  13: [0, 200, 255],
  14: [50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [150, 240, 255],
  19: [250, 10, 250],
  20: [255, 170, 170],
  21: [255, 0, 0]
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
        xyzi_img, _, _ , _ = spherical_projection(xyzil,self.projection[0],self.projection[1],theta_range=[-np.pi/2, np.pi/2])
        xyzi_img = xyzi_img[~np.all(np.linalg.norm(xyzi_img,axis=-1) == 0, axis=1)]
        if self.resize:
            xyzi_img = cv2.resize(xyzi_img, (1024,64), interpolation=cv2.INTER_NEAREST)
        

        if self.flip:
            do_flip = np.random.choice([True, False])
            if do_flip:
                xyzi_img = xyzi_img[:,::-1,:]
                xyzi_img[...,1] = -xyzi_img[...,1]

        label_img = xyzi_img[...,4:5]
        reflectivity_img = xyzi_img[...,3]
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
    import cv2 
    import open3d as o3d
    data_path_test = [(bin_path, bin_path.replace("velodyne", "labels").replace("bin", "label")) for bin_path in sorted(glob.glob(f"/home/appuser/data/WADS/11/velodyne/*.bin"))]

    depth_dataset_test = SemanticKitti(data_path_test, rotate=False, flip=False, projection=(1024,1024),resize=True)
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
