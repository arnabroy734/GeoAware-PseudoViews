from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import torch 
from pathlib import Path
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import numpy as np
from scene.gaussian_model import GaussianModel 
from scene.cameras import Camera
from scene import Scene
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim
from sklearn.neighbors import LocalOutlierFactor
from augment import PseudoViewGeneratorTraining
from torchmetrics.functional.regression import pearson_corrcoef
from scipy.ndimage import label



# class MaskGen: 
#     """This class is responsible for mask generation using pretrained SAM model"""
#     def __init__(self, checkpoint: str = "sam_ckpt/sam.pth"):
#         self.model_path = checkpoint 
#         self.masks = {} # key is imagefile name and value is list of masks

#     def _create_mask_generator(self, gpu_id: int = 0):
#         sam = sam_model_registry["vit_l"](self.model_path)
#         self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#         sam.to(self.device)
#         self.mask_generator = SamAutomaticMaskGenerator(
#             model=sam,
#             points_per_side=128,              # ↓ fewer proposals
#             pred_iou_thresh=0.95,           # ↑ stricter confidence
#             stability_score_thresh=0.88,    # ↑ more stable only
#             # min_mask_region_area=2000,      # ↑ filters tiny noisy segments
#             crop_n_layers=0,                # disable multi-scale crops
#             crop_overlap_ratio=0.0,         # disable overlapping crops
#         )

#     def _preprocess_masks(self, masks: list[np.ndarray]) -> list[np.ndarray]:
#         """
#         Logic:
#         - Reject mask entirely if original area < 1% of image area.
#         - Else split into connected components.
#         - Reject component if comp_area < 1% of original_mask_area.
#         - Sort remaining masks by area (ascending).
#         """
    
#         if len(masks) == 0:
#             return []

#         H, W = masks[0].shape
#         img_area = H * W
#         img_threshold = 0.005 * img_area  # 0.5% of image

#         final_masks = []

#         for m in masks:
#             orig_area = np.count_nonzero(m)

#             # Step 1: Reject entire mask if too small compared to image
#             if orig_area < img_threshold:
#                 continue

#             # Step 2: Split into connected components
#             labeled, num = label(m)

#             # Component threshold relative to original mask
#             comp_min_area = 0.01 * orig_area

#             for cid in range(1, num + 1):
#                 comp_mask = (labeled == cid)
#                 comp_area = np.count_nonzero(comp_mask)

#                 # Step 3: Reject components <1% of original mask
#                 if comp_area >= comp_min_area:
#                     final_masks.append(comp_mask)

#         # Step 4: Sort by area ascending
#         final_masks.sort(key=lambda x: np.count_nonzero(x))

#         return final_masks

#     def generate_masks(self, input_path: str): 
#         """There should be images folder inside input_path, it will generate masks for all the images"""
#         self._create_mask_generator()
#         input_path = Path(input_path).expanduser()  
#         input_path = input_path.resolve()
#         input_path = input_path/'images'
#         if not input_path.exists(): 
#             raise FileNotFoundError("Maks generation error!: images folder not present in input_path")
#         for imagefile in tqdm(input_path.iterdir(), desc='Generating masks'): 
#             img = cv2.imread(imagefile)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             masks = self.mask_generator.generate(img)
#             self.masks[imagefile.name] = [m['segmentation'] for m in masks]
#             self.masks[imagefile.name] = self._preprocess_masks(self.masks[imagefile.name])
#             break # TO BE DELETED LATER
#         del self.mask_generator 
#         torch.cuda.empty_cache()

# def show_all_masks(imagepath, masks, alpha=0.5, savepath: str = None):
#     imagepath = cv2.imread(imagepath)
#     imagepath = cv2.cvtColor(imagepath, cv2.COLOR_BGR2RGB)
#     overlay = imagepath.copy().astype(float) / 255.0 

#     for i,mask in enumerate(masks):
#         color = np.random.rand(3)  # random RGB in [0,1]
#         overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
#     plt.figure(figsize=(8,8))
#     plt.axis('off')
#     plt.imshow(overlay)
#     if savepath is not None: 
#         plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=300)

# def semantic_pruning(gaussians: GaussianModel,
#                      view_cam: Camera, pipe, background,
#                      masks: list[torch.Tensor]
#             ): 
#     total_pruned = 0
#     for key in masks.keys():
#         if key == view_cam.image_name: 
#             imagefile = key 
#             break  
#     print(f"before pruning size: {gaussians._xyz.shape}")
#     for segment_mask in masks[imagefile]:
#         inv_mask = torch.tensor(~segment_mask, device=gaussians._xyz.device)
#         rendered = render(view_cam, gaussians, pipe, background, separate_sh=False)['render']
#         pseudo_gt = rendered.detach().clone()
#         dummy_l = [
#             {'params': [gaussians._xyz], 'lr': 0.0, "name": "xyz"},
#         ]
#         optim = torch.optim.Adam(dummy_l, lr=0.0, eps=1e-15) 
#         gaussians.optimizer.zero_grad()
#         gaussians.exposure_optimizer.zero_grad()
#         optim.zero_grad()
#         loss = torch.mean(l1_loss(rendered, pseudo_gt*inv_mask))
#         loss.backward() 
#         # get xyz grad norm
#         for group in dummy_l:
#             for p in group["params"]:
#                 if p.grad is not None:
#                     if group['name'] == 'xyz':
#                         xyz_grad = torch.linalg.norm(p.grad, dim=1)
        
#         # find object related masks 
#         three_d_mask = xyz_grad > 0.0
#         X = gaussians._xyz[three_d_mask].detach().cpu().numpy()
#         if len(X) <= 10:
#             continue
#         lof = LocalOutlierFactor(n_neighbors=min(10, len(X) - 1))
#         # lof = LocalOutlierFactor(n_neighbors=1)
#         y = lof.fit_predict(X)
        
#         # detect outliers
#         outlier_mask = (y == -1)
#         # outlier_mask = detect_density_outliers(gaussians._xyz[three_d_mask].detach().cpu().numpy())
#         outlier_mask = torch.tensor(outlier_mask)
        
#         # merge outlier mask with overall mask
#         copy_mask = three_d_mask.clone() 
#         copy_mask[:] = False 
#         idx = torch.nonzero(three_d_mask, as_tuple=True)[0]

#         # TEMP CODE
#         # print("gaussians._xyz.shape:", gaussians._xyz.shape)
#         # print("three_d_mask.sum():", three_d_mask.sum().item())
#         # print("idx.shape:", idx.shape)
#         # print("outlier_mask.shape:", outlier_mask.shape)
#         # print("outlier_mask.sum():", outlier_mask.sum().item())
#         # show min/max of the selected indices (after converting to cpu numpy)
#         selected = idx[outlier_mask.to(idx.device)]
#         if selected.numel() == 0:
#             # print("No outliers detected in this iteration, skipping pruning.")
#             continue  # or skip the pruning for this iteration
#         # print("selected.min(), selected.max():", selected.min().item(), selected.max().item())

#         copy_mask[idx[outlier_mask]] = True

#         # prune gaussians
#         gaussians.prune_points(copy_mask, ignore_temp_radii=True)
#         del loss, rendered, pseudo_gt
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

#         # densify just after pruning 
#         total_pruned += torch.sum(copy_mask)


        
#     print(f"After pruning size: {gaussians._xyz.shape}")
#     print(f"TOTAL PRUNED {total_pruned}")

# def aggressive_densification(gaussians: GaussianModel,
#                 view_cam: Camera, pipe, background,
#                 scene: Scene, 
#                 gt_image: torch.Tensor
#             ): 
#     total_added = 0
#     print(f"before densification size: {gaussians._xyz.shape}")
#     pkg = render(view_cam, gaussians, pipe, background, separate_sh=False)
#     rendered = pkg['render']
#     dummy_l = [
#         {'params': [gaussians._xyz], 'lr': 0.0, "name": "xyz"},
#     ]
#     optim = torch.optim.Adam(dummy_l, lr=0.0, eps=1e-15) 
#     gaussians.optimizer.zero_grad()
#     gaussians.exposure_optimizer.zero_grad()
#     optim.zero_grad()
#     loss = torch.mean(l1_loss(rendered, gt_image))

#     loss.backward() 
#     # get xyz grad norm
#     for group in dummy_l:
#         for p in group["params"]:
#             if p.grad is not None:
#                 if group['name'] == 'xyz':
#                     xyz_grad = p.grad
        
#     # print(torch.sum(xyz_grad > 0.0))
#     # total_added += torch.sum(xyz_grad > 0.00001)
#     total_added += torch.sum(xyz_grad > 10**(-6))

#     gaussians.tmp_radii = pkg['radii']
#     gaussians.densify_and_clone(xyz_grad, grad_threshold=10**(-6), scene_extent=scene.cameras_extent) 
#     gaussians.tmp_radii = None 

#     del loss, rendered
#     torch.cuda.empty_cache()
#     torch.cuda.synchronize()

#     # densify just after pruning         
#     print(f"After densification {gaussians._xyz.shape}")
#     print(f"TOTAL ADDED: {total_added}")

# def semantic_densification(gaussians: GaussianModel,
#                      view_cam: Camera, pipe, background,
#                      masks: list[torch.Tensor],
#                      scene: Scene, 
#                      gt_image: torch.Tensor
#             ): 
#     total_added = 0
#     for key in masks.keys():
#         if key == view_cam.image_name: 
#             imagefile = key 
#             break  
#     print(f"before densification size: {gaussians._xyz.shape}")
#     for segment_mask in masks[imagefile]:
#         # mask = torch.tensor(~segment_mask, device=gaussians._xyz.device)
#         mask = torch.tensor(segment_mask, device=gaussians._xyz.device)
#         pkg = render(view_cam, gaussians, pipe, background, separate_sh=False)
#         rendered = pkg['render']
#         pseudo_gt = rendered.detach().clone()
#         dummy_l = [
#             {'params': [gaussians._xyz], 'lr': 0.0, "name": "xyz"},
#         ]
#         optim = torch.optim.Adam(dummy_l, lr=0.0, eps=1e-15) 
#         gaussians.optimizer.zero_grad()
#         gaussians.exposure_optimizer.zero_grad()
#         optim.zero_grad()
#         # loss = torch.mean(l1_loss(rendered, pseudo_gt*mask))
#         loss = torch.mean(l1_loss(rendered*mask, gt_image*mask))
#         # loss = torch.mean(l1_loss(rendered, gt_image))

#         loss.backward() 
#         # get xyz grad norm
#         for group in dummy_l:
#             for p in group["params"]:
#                 if p.grad is not None:
#                     if group['name'] == 'xyz':
#                         xyz_grad = p.grad
        
#         # print(torch.sum(xyz_grad > 0.0))
#         # total_added += torch.sum(xyz_grad > 0.00001)
#         total_added += torch.sum(xyz_grad > 10**(-5))

#         gaussians.tmp_radii = pkg['radii']
#         gaussians.densify_and_clone(xyz_grad, grad_threshold=10**(-5), scene_extent=scene.cameras_extent) 
#         gaussians.tmp_radii = None 

#         del loss, rendered
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

#         # densify just after pruning 
        
#         # break
        
#     print(f"After densification {gaussians._xyz.shape}")
#     print(total_added)

# class DepthBasedPruning:
#     def __init__(self, input_path: str):
#         gen  = PseudoViewGeneratorTraining(input_path) 
#         self.depths = gen.depths 
#     def prune(self,
#               gaussians: GaussianModel,
#               view_cam: Camera, pipe, background
#         ):
#         gt_depth = torch.tensor(1/self.depths[view_cam.image_name], device=gaussians._xyz.device)
#         pkg = render(view_cam, gaussians, pipe, background, separate_sh=False)
#         ren_depth = pkg['depth'][0]
#         loss = 1 - pearson_corrcoef(gt_depth.flatten(), ren_depth.flatten()) 
#         loss.backward()
#         grad_norm = torch.linalg.norm(pkg['viewspace_points'].grad, dim=-1)
#         mask = grad_norm > 10**(-4)
#         print(f"DEPTH PRUNING {torch.sum(mask)} GAUSSIANS")
#         gaussians.tmp_radii = pkg['radii']
#         gaussians.prune_points(mask)
#         gaussians.tmp_radii = None

#         pkg['viewspace_points'].grad = None 
#         gaussians.optimizer.zero_grad()
#         gaussians.exposure_optimizer.zero_grad()

#         del loss, pkg
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()

def _compute_gradient_magnitude(tensor):
    """
    Convert (3, H, W) to grayscale, then compute |grad_x| + |grad_y|
    
    Args:
        tensor: torch.Tensor of shape (3, H, W)
    
    Returns:
        gradient_mag: torch.Tensor of shape (H, W)
    """
    
    # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
    # Or simple average: tensor.mean(dim=0)
    gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    # gray shape: (H, W)
    
    # X gradient: |gray[:, i+1] - gray[:, i]|
    grad_x = torch.zeros_like(gray)
    grad_x[:, :-1] = torch.abs(gray[:, 1:] - gray[:, :-1])
    
    # Y gradient: |gray[i+1, :] - gray[i, :]|
    grad_y = torch.zeros_like(gray)
    grad_y[:-1, :] = torch.abs(gray[1:, :] - gray[:-1, :])
    
    # Total magnitude
    gradient_mag = grad_x + grad_y
    
    return gradient_mag

def _find_gaussian_group_mask(gaussians: GaussianModel, pipe, background,
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
        grad_mask_out = grad_norm > 10**(-5)

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

def find_hi_freq_gaussians(gaussians: GaussianModel, pipe, background, view_cams): 
    """Find mask for gaussians responsible for high freq region of images"""
    hi_freq_masks = [] 
    for view_cam in view_cams: 
        grad_mag = _compute_gradient_magnitude(view_cam.original_image)
        hi_freq_mask = grad_mag > 0.05
        hi_freq_masks.append(hi_freq_mask)

    final_mask = torch.zeros((gaussians._xyz.shape[0], ), dtype=torch.bool, device=gaussians._xyz.device)
    for i,view_cam in enumerate(view_cams):
        gaussian_mask = _find_gaussian_group_mask(gaussians, pipe, background,
                             hi_freq_masks[i], view_cam)
        final_mask = final_mask | gaussian_mask

    return final_mask
    

            

            
        