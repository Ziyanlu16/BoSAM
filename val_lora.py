import os
join = os.path.join
import numpy as np
from glob import glob
import torch
import torch.nn as nn

from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
from collections import OrderedDict, defaultdict
import json
import pickle
from utils.click_method import get_next_click3D_torch_ritm, get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL_Val
import numpy as np
import random 
import math
import torch.nn as nn
import datetime
import logging
import matplotlib.pyplot as plt
import os
join = os.path.join 
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchio as tio
from torch.utils.data.distributed import DistributedSampler
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.modeling import sam3D
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss, GeneralizedDiceFocalLoss
from contextlib import nullcontext
from utils.click_method import get_next_click3D_torch_2
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader
from utils.data_paths import img_datas
from torch.utils.data.dataloader import default_collate

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default=['/mnt/dataset/trainingdata'])
parser.add_argument('-vp', '--vis_path', type=str, default='visual/')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='/path/to/checkpoint')
parser.add_argument('--save_name', type=str, default='uniondice/')

parser.add_argument('--lora_ckpt', type=str, default='/path/to/lora_checkpoint', help='path to LoRA checkpoint')
parser.add_argument('--lora_rank', type=int, default=4, help='rank of LoRA')
parser.add_argument('--skip_existing_pred', action='store_true', default=False)

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-nc', '--num_clicks', type=int, default=20)
parser.add_argument('-pm', '--point_method', type=str, default='default')
parser.add_argument('-dt', '--data_type', type=str, default='Ts')

parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()

click_methods = {
    'default': get_next_click3D_torch_ritm,
    'ritm': get_next_click3D_torch_ritm,
    'random': get_next_click3D_torch_2,
}

def compute_iou(pred_mask, gt_semantic_seg):
    mask_threshold = 0.5
    pred_mask = (pred_mask > mask_threshold)

    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou


from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import Parameter

class _LoRA_qkv(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        qkv[:, :, :, :, :self.dim] += new_q
        qkv[:, :, :, :, -self.dim:] += new_v
        return qkv

class LoRA_Sam3D(nn.Module):
    def __init__(self, sam_model: sam3D, r: int, lora_layer=None):
        super(LoRA_Sam3D, self).__init__()
        
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False).to('cuda')
            w_b_linear_q = nn.Linear(r, self.dim, bias=False).to('cuda')
            w_a_linear_v = nn.Linear(self.dim, r, bias=False).to('cuda')
            w_b_linear_v = nn.Linear(r, self.dim, bias=False).to('cuda')
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        # Add LoRA to mask decoder
        self.mask_decoder_w_As = nn.ModuleList()
        self.mask_decoder_w_Bs = nn.ModuleList()
        
        for module in sam_model.mask_decoder.modules():
            if isinstance(module, nn.Linear):
                w_a = nn.Linear(module.in_features, r, bias=False)
                w_b = nn.Linear(r, module.out_features, bias=False)
                self.mask_decoder_w_As.append(w_a)
                self.mask_decoder_w_Bs.append(w_b)
                
                # Wrap the original linear layer
                new_module = _LoRA_qkv(module, w_a, w_b, w_a, w_b)
                
                # Replace the original module with the wrapped one
                for name, child in sam_model.mask_decoder.named_children():
                    if child is module:
                        setattr(sam_model.mask_decoder, name, new_module)
                        break

        for param in self.w_As.parameters():
            param.requires_grad = False
        for param in self.w_Bs.parameters():
            param.requires_grad = False
        for param in self.mask_decoder_w_As.parameters():
            param.requires_grad = False
        for param in self.mask_decoder_w_Bs.parameters():
            param.requires_grad = False

        self.reset_parameters()
        self.sam = sam_model

    @property
    def image_encoder(self):
        return self.sam.image_encoder

    @property
    def prompt_encoder(self):
        return self.sam.prompt_encoder

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)

    def reset_parameters(self) -> None:
        for w_A in self.w_As + self.mask_decoder_w_As:
            nn.init.trunc_normal_(w_A.weight, std=0.02)
        for w_B in self.w_Bs + self.mask_decoder_w_Bs:
            nn.init.trunc_normal_(w_B.weight, std=0.02)

    def save_lora_parameters(self, filename: str) -> None:
        lora_state_dict = {
            'w_As': [w_A.state_dict() for w_A in self.w_As],
            'w_Bs': [w_B.state_dict() for w_B in self.w_Bs],
            'mask_decoder_w_As': [w_A.state_dict() for w_A in self.mask_decoder_w_As],
            'mask_decoder_w_Bs': [w_B.state_dict() for w_B in self.mask_decoder_w_Bs]
        }
        torch.save(lora_state_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        lora_state_dict = torch.load(filename, map_location=self.sam.device)
        
        for i, w_A in enumerate(self.w_As):
            w_A.load_state_dict(lora_state_dict['w_As'][i])
        for i, w_B in enumerate(self.w_Bs):
            w_B.load_state_dict(lora_state_dict['w_Bs'][i])
        for i, w_A in enumerate(self.mask_decoder_w_As):
            w_A.load_state_dict(lora_state_dict['mask_decoder_w_As'][i])
        for i, w_B in enumerate(self.mask_decoder_w_Bs):
            w_B.load_state_dict(lora_state_dict['mask_decoder_w_Bs'][i])

def compute_dice(mask_pred, mask_gt):
    mask_threshold = 0.5

    mask_pred = (mask_pred > mask_threshold)
    mask_gt = (mask_gt > 0)
    
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN

    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )

    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)

    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad

def sam_decoder_inference(target_size, points_coords, points_labels, model, image_embeddings, mask_inputs=None, multimask = False):
    with torch.no_grad():

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points_coords.to(model.device), points_labels.to(model.device)),
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
    
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

def repixel_value(arr, is_seg=False):
    if not is_seg:
        min_val = arr.min()
        max_val = arr.max()
        new_arr = (arr - min_val) / (max_val - min_val + 1e-10) * 255.
    return new_arr

def random_point_sampling(mask, get_point = 1):
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels


def finetune_model_predict2D(img3D, gt3D, sam_model_tune, target_size=256, click_method='random', device='cuda', num_clicks=1, prev_masks=None):
    pred_list = []
    iou_list = []
    dice_list = []

    slice_mask_list = defaultdict(list)

    img3D = torch.repeat_interleave(img3D, repeats=3, dim=1) # 1 channel -> 3 channel (align to RGB)
    
    click_points = []
    click_labels = []
    for slice_idx in tqdm(range(img3D.size(-1)), desc="transverse slices", leave=False):
        img2D, gt2D = repixel_value(img3D[..., slice_idx]), gt3D[..., slice_idx]

        if (gt2D==0).all():
            empty_result = torch.zeros(list(gt3D.size()[:-1])+[1]).to(device)
            for iter in range(num_clicks):
                slice_mask_list[iter].append(empty_result)
            continue

        img2D = F.interpolate(img2D, (target_size, target_size), mode="bilinear", align_corners=False)
        gt2D = F.interpolate(gt2D.float(), (target_size, target_size), mode="nearest").int()
        
        img2D, gt2D = img2D.to(device), gt2D.to(device)
        img2D = (img2D - img2D.mean()) / img2D.std()

        with torch.no_grad():
            image_embeddings = sam_model_tune.image_encoder(img2D.float())

        points_co, points_la = torch.zeros(1,0,2).to(device), torch.zeros(1,0).to(device)
        low_res_masks = None
        gt_semantic_seg = gt2D[0, 0].to(device)
        true_masks = (gt_semantic_seg > 0)
        for iter in range(num_clicks):
            if(low_res_masks==None):
                pred_masks = torch.zeros_like(true_masks).to(device)
            else:
                pred_masks = (prev_masks[0, 0] > 0.0).to(device) 
            fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
            fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)
            mask_to_sample = torch.logical_or(fn_masks, fp_masks)
            new_points_co, _ = random_point_sampling(mask_to_sample.cpu(), get_point=1)
            new_points_la = torch.Tensor([1]).to(torch.int64) if(true_masks[new_points_co[0,1].int(), new_points_co[0,0].int()]) else torch.Tensor([0]).to(torch.int64)
            new_points_co, new_points_la = new_points_co[None].to(device), new_points_la[None].to(device)
            points_co = torch.cat([points_co, new_points_co],dim=1)
            points_la = torch.cat([points_la, new_points_la],dim=1)
            prev_masks, low_res_masks, iou_predictions = sam_decoder_inference(
                target_size, points_co, points_la, sam_model_tune, image_embeddings, 
                mask_inputs = low_res_masks, multimask = True)
            click_points.append(new_points_co)
            click_labels.append(new_points_la)
            
            slice_mask, _ = postprocess_masks(low_res_masks, target_size, (gt3D.size(2), gt3D.size(3)))
            slice_mask_list[iter].append(slice_mask[..., None]) # append (B, C, H, W, 1)
        
    for iter in range(num_clicks):
        medsam_seg = torch.cat(slice_mask_list[iter], dim=-1).cpu().numpy().squeeze()
        medsam_seg = medsam_seg > sam_model_tune.mask_threshold
        medsam_seg = medsam_seg.astype(np.uint8)
    
        pred_list.append(medsam_seg) 
        iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
        dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))

    return pred_list, click_points, click_labels, iou_list, dice_list


def finetune_model_predict3D(img3D, gt3D, sam_model_tune, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)

    click_points = []
    click_labels = []

    pred_list = []

    iou_list = []
    dice_list = []
    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

    with torch.no_grad():

        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

    for num_click in range(num_clicks):

        with torch.no_grad():
            if(num_click>1):
                click_method = "random"
            batch_points, batch_labels = click_methods[click_method](prev_masks.to(device), gt3D.to(device))

            points_co = torch.cat(batch_points, dim=0).to(device)  
            points_la = torch.cat(batch_labels, dim=0).to(device)  

            click_points.append(points_co)
            click_labels.append(points_la)

            points_input = points_co
            labels_input = points_la

            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=[points_input, labels_input],
                boxes=None,
                masks=low_res_masks.to(device),
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                multimask_output=False,
                )
            prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

            medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
            # convert prob to mask
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            pred_list.append(medsam_seg)

            iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
            dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))

    return pred_list, click_points, click_labels, iou_list, dice_list


def save_nifti(tensor, filename):
    array = tensor.cpu().numpy()
    
    image = sitk.GetImageFromArray(array)
    
    sitk.WriteImage(image, filename)
    print(f"Saved NIfTI file: {filename}")

if __name__ == "__main__":    
    all_dataset_paths = args.test_data_path
    print(all_dataset_paths)
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size,args.crop_size,args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    if(args.dim==3):
        sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
        sam_model_tune = LoRA_Sam3D(sam_model, args.lora_rank).to(device)
        if checkpoint_path is not None:
            model_dict = torch.load(checkpoint_path, map_location=device)
            state_dict = model_dict['model_state_dict']
            sam_model_tune.load_state_dict(state_dict, strict=True)
        
        if args.lora_ckpt is not None:
            sam_model_tune.load_lora_parameters(args.lora_ckpt)
            
    elif(args.dim==2):
        args.sam_checkpoint = args.checkpoint_path
        sam_model_tune = sam_model_registry[args.model_type](args).to(device)

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

    all_iou_list = []
    all_dice_list = []  

    out_dice = dict()
    out_dice_all = OrderedDict()

    for batch_data in tqdm(test_dataloader):
        image3D, gt3D, img_name = batch_data
        sz = image3D.size()
        if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):
            print("[ERROR] wrong size", sz, "for", img_name)
        modality = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))
        vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, modality, dataset)
        os.makedirs(vis_root, exist_ok=True)

        image_save_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_image3D.nii.gz"))
        gt_save_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_gt3D.nii.gz"))
        save_nifti(image3D.squeeze(), image_save_path)
        save_nifti(gt3D.squeeze(), gt_save_path)

        pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"))
        if(args.skip_existing_pred and os.path.exists(pred_path)):
            iou_list, dice_list = [], []
            for iter in range(args.num_clicks):
                curr_pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{iter}.nii.gz"))
                medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
                dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
        else:
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            if(args.dim==3):
                seg_mask_list, points, labels, iou_list, dice_list = finetune_model_predict3D(
                    image3D, gt3D, sam_model_tune, device=device, 
                    click_method=args.point_method, num_clicks=args.num_clicks, 
                    prev_masks=None)
            elif(args.dim==2):
                seg_mask_list, points, labels, iou_list, dice_list = finetune_model_predict2D(
                    image3D, gt3D, sam_model_tune, device=device, target_size=args.image_size,
                    click_method=args.point_method, num_clicks=args.num_clicks, 
                    prev_masks=None)
            os.makedirs(vis_root, exist_ok=True)
            points = [p.cpu().numpy() for p in points]
            labels = [l.cpu().numpy() for l in labels]
            pt_info = dict(points=points, labels=labels)

            print("save to", os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pred.nii.gz")))
            pt_path=os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl"))
            pickle.dump(pt_info, open(pt_path, "wb"))
            for idx, pred3D in enumerate(seg_mask_list):
                out = sitk.GetImageFromArray(pred3D)
                sitk.WriteImage(out, os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{idx}.nii.gz")))

        per_iou = max(iou_list)
        all_iou_list.append(per_iou)
        all_dice_list.append(max(dice_list))
        print(dice_list)
        out_dice[img_name] = max(dice_list)
        cur_dice_dict = OrderedDict()
        for i, dice in enumerate(dice_list):
            cur_dice_dict[f'{i}'] = dice
        out_dice_all[img_name[0]] = cur_dice_dict

    print('Mean IoU : ', sum(all_iou_list)/len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list)/len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if(args.split_num>1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
        f.writelines('}')

    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")
