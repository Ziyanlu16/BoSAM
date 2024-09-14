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

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train_turbo_lora2')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='/mnt/risk2/SAM-Med3D/work_dir/union_train_turbo_best_continue4/sam_model_dice_best.pth')
parser.add_argument('--lora_ckpt', type=str, default='/mnt/risk2/SAM-Med3D/work_dir/union_train_turbo_lora/lora_params_dice_best.pth', help='path to LoRA checkpoint')
parser.add_argument('--freeze_sam', action='store_false', default=True, help='whether to freeze SAM3D weights')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')
# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2])
parser.add_argument('--multi_gpu', action='store_true', default=True)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='coswarm')
parser.add_argument('--pcc', action='store_true', default=False)

# parser.add_argument('--step_size', type=list, default=[30, 50, 70])
# parser.add_argument('--gamma', type=float, default=0.5)

parser.add_argument('--num_epochs', type=int, default=800)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--accumulation_steps', type=int, default=30)
parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12362)

args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

from safetensors import safe_open
from safetensors.torch import save_file


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
        
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    sam_model = LoRA_Sam3D(sam_model, 4).to(device)

    if args.multi_gpu:
        sam_model = DDP(sam_model, device_ids=[args.rank], output_device=args.rank)
    return sam_model


def get_dataloaders(args):
    if args.pcc==True:
        print("Random Crop")
        train_dataset = Dataset_Union_ALL(
            paths=img_datas, 
            transform=tio.Compose([
                tio.ToCanonical(),
                tio.RandomFlip(axes=(0, 1, 2)),
            ]), 
            pcc=True,  # 启用 pcc
            threshold=1000
        )
    else:
        train_dataset = Dataset_Union_ALL(
            paths=img_datas, 
            transform=tio.Compose([
                tio.ToCanonical(),
                tio.CropOrPad(mask_name='label', target_shape=(args.img_size, args.img_size, args.img_size)),
                tio.RandomFlip(axes=(0, 1, 2)),
            ]),
            threshold=1000,
            pcc=False  # 不启用 pcc
        )

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_dataloader

def get_valdataloaders(args):
    
    val_dataset = Dataset_Union_ALL(
        paths=[
                '/mnt/dataset/val_data/verse20',
                '/mnt/dataset/val_data/verse19',
                '/mnt/dataset/val_data/MSD_T10',
                '/mnt/dataset/val_data/LIVER',
                '/mnt/dataset/val_data/KITS19',
                '/mnt/dataset/val_data/COVID',
                '/mnt/dataset/val_data/NH',
                '/mnt/dataset/val_data/CLINIC_METAL',
                '/mnt/dataset/val_data/CLINIC',
                '/mnt/dataset/val_data/SPIDER',
                '/mnt/dataset/val_data/pelvic',
                '/mnt/dataset/val_data/VERSE',
                '/mnt/dataset/val_data/COLON'
                ],
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.RandomFlip(axes=(0, 1, 2)),
        ]),
        threshold=1000,
        pcc=True  # 启用 pcc
    )

    if args.multi_gpu:
        val_sampler = DistributedSampler(val_dataset)
        shuffle = False
    else:
        val_sampler = None
        shuffle = True

    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return val_dataloader

def iter_params(modules):
        for module in modules:
            yield from module.parameters()

class BaseTrainer:
    def __init__(self, model, dataloaders,val_dataloaders, args):

        self.model = model
        self.dataloaders = dataloaders
        self.val_dataloaders = val_dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_val_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if(args.resume):
            self.init_checkpoint(join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.init_checkpoint(self.args.checkpoint)

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        
    def set_loss_fn(self):
        # self.seg_loss = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.seg_loss = GeneralizedDiceFocalLoss(sigmoid=True, reduction='mean')
    

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def set_optimizer(self):
        if self.args.multi_gpu:

            sam_model = self.model.module
        else:
            sam_model = self.model

        # 设置参数组
        param_groups = []

        # LoRA 参数
        lora_params = []
        for name, param in sam_model.named_parameters():
            if 'w_As' in name or 'w_Bs' in name:
                lora_params.append(param)
        param_groups.append({'params': lora_params, 'lr': self.args.lr})

        # LoRA 参数以外的 image_encoder 参数
        image_encoder_params = []
        for name, param in sam_model.sam.image_encoder.named_parameters():
            if 'w_As' not in name and 'w_Bs' not in name:
                image_encoder_params.append(param)
        param_groups.append({'params': image_encoder_params, 'lr': self.args.lr * 0.1})

        # prompt_encoder 参数
        prompt_encoder_params = list(sam_model.sam.prompt_encoder.parameters())
        param_groups.append({'params': prompt_encoder_params, 'lr': self.args.lr * 0.01})

        # mask_decoder 参数
        mask_decoder_params = list(sam_model.sam.mask_decoder.parameters())
        param_groups.append({'params': mask_decoder_params, 'lr': self.args.lr * 0.01})

        self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)


    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)

        if last_ckpt:
            if self.args.multi_gpu:
                self.model.module.sam.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                self.model.sam.load_state_dict(last_ckpt['model_state_dict'], strict=False)

            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']

            # 加载LoRA权重
            if self.args.lora_ckpt is not None:
                lora_ckp_path = self.args.lora_ckpt
            else:
                lora_ckp_path = ckp_path.replace('sam_model', 'lora_params')
            
            if os.path.exists(lora_ckp_path):
                lora_ckpt = torch.load(lora_ckp_path, map_location=self.args.device)
                if self.args.multi_gpu:
                    for w_A, w_A_state_dict in zip(self.model.module.w_As, lora_ckpt['w_As']):
                        w_A.load_state_dict(w_A_state_dict)
                    for w_B, w_B_state_dict in zip(self.model.module.w_Bs, lora_ckpt['w_Bs']):
                        w_B.load_state_dict(w_B_state_dict)
                else:
                    for w_A, w_A_state_dict in zip(self.model.w_As, lora_ckpt['w_As']):
                        w_A.load_state_dict(w_A_state_dict)
                    for w_B, w_B_state_dict in zip(self.model.w_Bs, lora_ckpt['w_Bs']):
                        w_B.load_state_dict(w_B_state_dict)
                print(f"Loaded LoRA checkpoint from {lora_ckp_path}")
            else:
                print(f"No LoRA checkpoint found at {lora_ckp_path}, using random initialization for LoRA weights")

            print(f"Loaded SAM3D checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
    # 保存完整模型的checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "dices": self.dices,
            "best_val_dice" : self.best_val_dice,
            "best_loss": self.best_loss,
            "best_dice": self.best_dice,
            "args": self.args,
            "used_datas": img_datas,
        }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

        # 保存LoRA参数
        lora_state_dict = {
            'w_As': [w_A.state_dict() for w_A in self.model.module.w_As],
            'w_Bs': [w_B.state_dict() for w_B in self.model.module.w_Bs]
        }
        torch.save(lora_state_dict, join(MODEL_SAVE_PATH, f"lora_params_{describe}.pth"))
    
    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(), size=(args.img_size//4,args.img_size//4,args.img_size//4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss
    

    def get_dice_score(self, prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
    
        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list)/len(dice_list)).item() 


    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        step_count = 0
        epoch_dice_list = []

        epoch_dice = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1
        
        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.dataloaders)
        else:
            tbar = self.dataloaders

        self.optimizer.zero_grad()
        step_loss = 0
        for step, (image3D, gt3D) in enumerate(tbar):

            my_context = self.model.no_sync if self.args.rank != -1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1)) # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)
                
                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    pred_list = []

                    prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)                

                epoch_loss += loss.item()

                cur_loss = loss.item()

                epoch_dice_list.append(self.get_dice_score(prev_masks, gt3D))

                loss /= self.args.accumulation_steps
                
                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = sum(epoch_dice_list) / len(epoch_dice_list)
                # step_count += 1
                # epoch_dice += print_dice
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(
                                epoch,
                                sam_model.state_dict(),
                                describe=f'{epoch}_step_dice:{print_dice}_best'
                            )
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss
            
        epoch_loss /= step
        epoch_dice = sum(epoch_dice_list) / len(epoch_dice_list)

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def validate_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_dice_list = []

        self.model.eval()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.val_dataloaders)
        else:
            tbar = self.val_dataloaders

        with torch.no_grad():
            for step, (image3D, gt3D) in enumerate(tbar):
                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).type(torch.long)
                with amp.autocast():
                    image_embedding = sam_model.image_encoder(image3D)

                    self.click_points = []
                    self.click_labels = []

                    prev_masks, loss = self.interaction(sam_model, image_embedding, gt3D, num_clicks=11)

                epoch_loss += loss.item()

                epoch_dice_list.append(self.get_dice_score(prev_masks, gt3D))

        epoch_loss /= len(tbar)
        epoch_dice = sum(epoch_dice_list) / len(epoch_dice_list)

        return epoch_loss, epoch_dice

    def eval_epoch(self, epoch, num_clicks):
        return 0
    
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
                
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)
            val_loss, val_dice = self.validate_epoch(epoch, num_clicks)

            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()
            # if self.args.multi_gpu:
            #     dist.barrier()
        
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                print(f'EPOCH: {epoch}, Train_Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Train_Dice: {epoch_dice}')
                print(f'Epoch: {epoch}, Val_Dice: {val_dice}, Val_Loss: {val_loss}')
                logger.info(f'Epoch\t {epoch}\t : train_loss: {epoch_loss:.5f}, train_dice: {epoch_dice:.5f}, val_loss: {val_loss:.5f}, val_dice: {val_dice:.5f}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                # save latest checkpoint
                self.save_checkpoint(
                    epoch, 
                    state_dict, 
                    describe='latest'
                )

                # save train loss best checkpoint
                if epoch_loss < self.best_loss: 
                    self.best_loss = epoch_loss
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='loss_best'
                    )
                
                # save train dice best checkpoint
                if epoch_dice > self.best_dice: 
                    self.best_dice = epoch_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='dice_best'
                    )
                
                if val_dice > self.best_val_dice: 
                    self.best_val_dice = val_dice
                    self.save_checkpoint(
                        epoch,
                        state_dict,
                        describe='val_dice_best'
                    )

                self.plot_result(self.losses, 'Dice + Cross Entropy Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.args.multi_gpu:
                dist.barrier()

        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)


def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(
            main_worker,
            nprocs=args.world_size,
            args=(args, )
        )
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        dataloaders = get_dataloaders(args)
        val_dataloaders = get_valdataloaders(args)

        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders,val_dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_dataloaders(args)
    val_dataloaders = get_valdataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders,val_dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
