import numpy as np
import os
from scene.cameras import PseudoCamera
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
import torch 
import cv2
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from scene.gaussian_model import GaussianModel 
from scene.cameras import Camera
from scene import Scene
import torch
import warnings 
from augment import _get_2d_3d_corr, _predict_pixelwise_depth
from utils.pose_utils import create_scene_bounds, generate_random_poses_llff
warnings.filterwarnings('ignore')

class DepthSegmentationPruning: 
    def __init__(self, input_path): 
        corr_dict = _get_2d_3d_corr(f'{input_path}/sparse/0')
        self.depth_segments = {}
        for imagefile in corr_dict.keys():
            pixels = corr_dict[imagefile]['pixel_2d']
            sfm_3d = corr_dict[imagefile]['coordinate_3D']
            R, t = corr_dict[imagefile]['R'], corr_dict[imagefile]['t']
            camera_3d = R@sfm_3d.T + t 
            camera_depths = camera_3d[2, :]
            pred_depth = _predict_pixelwise_depth(f'{input_path}/images/{imagefile}', pixels, camera_depths)
            label_map, _ = self._depth_segmentation(1/pred_depth)
            self.depth_segments[imagefile] = label_map
            print(f"For image {imagefile}, total depth layers: {np.unique(label_map)}")
            
    def _depth_segmentation(self, depth_map: np.ndarray,
                            img: np.ndarray =None,
                            n_bins=15, min_region_area=0.5, morph_ksize=5):
        """
            Args:
                depth_map: HxW float depth map (can be negative/unbounded)
                img: optional original image for overlay
                n_bins: number of depth clusters
                min_region_area: minimum pixels to keep a region
                morph_ksize: kernel size for morphological cleaning
            Returns:
                label_map: HxW int map of segmented regions (0 = background)
                overlay: optional visual    ization on img
            """
        H, W = depth_map.shape
        min_region_area = min_region_area/100*H*W 
        valid_mask = np.isfinite(depth_map)
        depth_flat = depth_map[valid_mask].reshape(-1,1)

        # --- Step 1: KMeans clustering on valid depth pixels ---
        km = KMeans(n_clusters=n_bins, random_state=0)
        labels_flat = km.fit_predict(depth_flat)

        # assign back to full image
        depth_labels = np.zeros_like(depth_map, dtype=np.int32)
        depth_labels[valid_mask] = labels_flat + 1  # 0 reserved for background

        # --- Step 2: Morphological cleaning and connected components ---
        label_map = np.zeros_like(depth_labels, dtype=np.int32)
        next_label = 1

        kernel = np.ones((morph_ksize, morph_ksize), np.uint8)

        for b in range(1, n_bins+1):
            mask_bin = (depth_labels == b).astype(np.uint8)
            if np.sum(mask_bin) == 0:
                continue
            # Morphological closing
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
            # Connected components
            num_labels, comps = cv2.connectedComponents(mask_bin)
            for i in range(1, num_labels):
                comp_mask = (comps == i)
                if comp_mask.sum() < min_region_area:
                    continue
                label_map[comp_mask] = next_label
                next_label += 1

        # --- Step 3: Fill leftover pixels using nearest labeled pixel ---
        unassigned = (label_map == 0) & valid_mask
        if np.any(unassigned):
            distance, indices = ndi.distance_transform_edt(unassigned, return_distances=True, return_indices=True)
            nearest_labels = label_map[indices[0], indices[1]]
            label_map[unassigned] = nearest_labels[unassigned]

        # --- Step 4: Optional overlay visualization ---
        overlay = None
        if img is not None:
            overlay = img.copy()
            unique_labels = np.unique(label_map)
            colors = plt.cm.get_cmap('tab20', len(unique_labels))
            for i, lbl in enumerate(unique_labels):
                if lbl == 0:
                    continue
                mask = (label_map == lbl)
                color = (np.array(colors(i)[:3])*255).astype(np.uint8)
                overlay[mask] = 0.5*overlay[mask] + 0.5*color

        return label_map, overlay
    
    def _find_gaussian_group_mask(self, gaussians: GaussianModel, pipe, background,
                             mask: np.ndarray, view_cam: Camera): 
        
        mask_out = torch.tensor(mask, device=gaussians._xyz.device)
        res = render(view_cam, gaussians, pipe, background, separate_sh=False)
        rendered = res['render']
        pseudo_gt = rendered.detach().clone()
        gaussians.optimizer.zero_grad()
        gaussians.exposure_optimizer.zero_grad()
        loss = torch.mean(l1_loss(rendered, pseudo_gt*mask_out))
        loss.backward() 
        grad_norm = torch.linalg.norm(res['viewspace_points'].grad, dim=-1)
        grad_mask_out = grad_norm > 0.0

        mask_in = torch.tensor(~mask, device=gaussians._xyz.device)
        res = render(view_cam, gaussians, pipe, background, separate_sh=False)
        rendered = res['render']
        pseudo_gt = rendered.detach().clone()
        gaussians.optimizer.zero_grad()
        gaussians.exposure_optimizer.zero_grad()
        loss = torch.mean(l1_loss(rendered, pseudo_gt*mask_in))
        loss.backward() 
        grad_norm = torch.linalg.norm(res['viewspace_points'].grad, dim=-1)
        grad_mask_in = grad_norm > 0.0

        grad_mask =  grad_mask_in & (~grad_mask_out)
        return grad_mask

    def prune(self, gaussians: GaussianModel, pipe, background, view_cam: Camera): 
        imagefile = view_cam.image_name 
        prune_mask = None 
        for layer_num in np.unique(self.depth_segments[imagefile]): 
            mask = self.depth_segments[imagefile] == layer_num 

            # find gaussians
            gaussian_gr_mask = self._find_gaussian_group_mask(gaussians, pipe, background, mask, view_cam)
            world_xyz = gaussians._xyz[gaussian_gr_mask]

            # world to cam transform - find depth and depth prune mask
            cam_xyz = view_cam.R@world_xyz.T.detach().cpu().numpy() + view_cam.T.reshape((-1,1))
            mean_depth = np.mean(cam_xyz[2, :])
            std_depth = np.std(cam_xyz[2, :])
            depth_prune_mask = (cam_xyz[2, :] - mean_depth)/std_depth < -1.8
            outlier_mask = torch.tensor(depth_prune_mask)

            # merge gaussian mask and depth mask
            copy_mask = gaussian_gr_mask.clone() 
            copy_mask[:] = False 
            idx = torch.nonzero(gaussian_gr_mask, as_tuple=True)[0]
            selected = idx[outlier_mask.to(idx.device)]
            if selected.numel() == 0:
                print(f"No Depth outliers detected in layer {layer_num}")
                continue  # or skip the pruning for this iteration

            copy_mask[idx[outlier_mask]] = True
            if prune_mask is None: 
                prune_mask = copy_mask.clone()
            else: 
                prune_mask = prune_mask | copy_mask.clone()
        # prune gaussians 
        print(f"For cam: {view_cam.image_name} | layer {layer_num} | before: {gaussians._xyz.shape[0]}")
        with torch.no_grad(): 
            gaussians.tmp_radii = render(view_cam, gaussians, pipe, background, separate_sh=False)['radii']
            gaussians.prune_points(prune_mask)
            gaussians.tmp_radii = None 
            # gaussians.xyz_gradient_accum[copy_mask] = 0.0 # do not densify around those gaussians
            # print(f"For cam: {view_cam.image_name} | layer {layer_num} | total {torch.sum(copy_mask)} gaussians do not densify")

        print(f"For cam: {view_cam.image_name} | layer {layer_num} | after: {gaussians._xyz.shape[0]}")

    def densify(self, gaussians: GaussianModel,
                 view_cam: Camera, pipe, background,
                 scene: Scene, 
                 gt_image: torch.Tensor
        ): 
        # total_added = 0
        print(f"before densification size: {gaussians._xyz.shape}")
        # Render
        pkg = render(view_cam, gaussians, pipe, background, separate_sh=False)
        rendered = pkg['render']
        # Zero gradients BEFORE backward
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        # Compute loss
        loss = torch.mean(l1_loss(rendered, gt_image))
        loss.backward() 
        
        # Extract what you need immediately
        grad_norm = torch.linalg.norm(pkg['viewspace_points'].grad.detach(), dim=-1)
        del loss, rendered, pkg
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


        with torch.no_grad():
            pkg = render(view_cam, gaussians, pipe, background, separate_sh=False)
            radii = pkg['radii'].detach().clone()  # Clone to break reference
            gaussians.tmp_radii = radii


            gaussians.densify_and_clone(grad_norm, grad_threshold=10**(-4), 
                                       scene_extent=scene.cameras_extent) 
            gaussians.densify_and_split(grad_norm, grad_threshold=10**(-4), 
                                       scene_extent=scene.cameras_extent)
            gaussians.tmp_radii = None 
        # del grad_norm, radii, mask
        # torch.cuda.empty_cache()
    
        print(f"After densification {gaussians._xyz.shape}")


# class DepthGrowthLoss: 
#     def __init__(self, view_cams: list[Camera], input_path: str, alpha: float = 0.8, thres: float = 0.05):
#         self.alpha = alpha 
#         self.thres = thres

#         # generate pseudo cameras
#         npy_path = os.path.join(input_path, 'poses_bounds.npy')
#         bounds = np.load(npy_path)
#         pseudo_poses = generate_random_poses_llff(view_cams, bounds, n_poses=2000)
#         pseudo_cams = []
#         for pose in pseudo_poses:
#             pseudo_cams.append(PseudoCamera(
#                     R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view_cams[0].FoVx, FoVy=view_cams[0].FoVy,
#                     width=view_cams[0].image_width, height=view_cams[0].image_height
#         ))
#         self.all_cams = pseudo_cams + view_cams.copy()
#         self.depth_avg = []
#         for id,cam in enumerate(self.all_cams): 
#             self.depth_avg.append(torch.zeros(size=(cam.image_height, cam.image_width), device=view_cams[0].original_image.device))
    
#     def update_depth(self, gaussians, pipe, bg): 
#         with torch.no_grad(): 
#             for id, cam in enumerate(self.all_cams):
#                 pkg = render(cam, gaussians, pipe, bg, use_trained_exp=False, separate_sh=False)
#                 depth_t = pkg['depth'][0]
#                 self.depth_avg[id] = self.alpha*depth_t.detach() + (1-self.alpha)*self.depth_avg[id] 
    
#     def calculate_loss(self, gaussians, pipe, bg): 
#         rand_idx = np.random.randint(low=0, high=len(self.all_cams))
#         rand_cam = self.all_cams[rand_idx]
#         depth_rendered = render(rand_cam, gaussians, pipe, bg, use_trained_exp=False, separate_sh=False)['depth'][0]
#         mask = self.depth_avg[rand_idx] != 0.0
#         loss = torch.max(
#             torch.mean(
#                 torch.abs((self.depth_avg[rand_idx][mask] - depth_rendered[mask])/self.depth_avg[rand_idx][mask])
#             ), torch.tensor(self.thres, device=depth_rendered.device)
#         )
#         return loss 



