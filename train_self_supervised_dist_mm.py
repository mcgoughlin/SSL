import matplotlib
matplotlib.use('Agg')
#import UnetsegLSTM
#matplotlib.use('pdf')
import torch 
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.backends.cudnn.enabled = False
import matplotlib.pyplot as plt
import os
import json
import time
import torch
#import matplotlib.pyplot as plt
#Below all for SimMIM

import argparse
from utils.data_utils import Sampler
from datetime import timedelta

#Below for ibot
from train_model import train_utils
import torch.nn as nn
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.distributed as dist
import nibabel as nib
import numpy as np
from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai import transforms, data
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled
)

from models.Trans import CONFIGS as CONFIGS_TM
import models.Trans as Trans
import torch.nn.functional as F
import torch.multiprocessing as mp

fig = plt.figure()
ax = fig.add_subplot(211)


class Cls_Patch_Loss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

               
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))
       

    def forward(self, student_output, teacher_output,student_mask, epoch,chunk_num):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
       

        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(chunk_num)

        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(chunk_num)

        student_mask=student_mask.chunk(chunk_num)
        #print(self.teacher_temp_schedule)
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]

        
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(chunk_num)

        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(chunk_num)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
    
        num,channel_size,ft_size=student_patch.size()

        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    
                    mask = student_mask[q]
                    
                    mask=mask.flatten(-3,-1).cuda()
                    
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)

                    #print (loss1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2

        
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center( teacher_cls,teacher_patch)                  
        
        return total_loss2,total_loss1

    @torch.no_grad()
    def update_center(self, teacher_cls,teacher_patch):
        """
        Update center used for teacher output.
        """

        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls)) 
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)
        #self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) )
        #patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
        
    parser.add_argument('--cls_w', type=float, default=0.1, required=False, help="weight for global class matching")
    parser.add_argument('--patch_w', type=float, default=0.1, required=False,help="weight for patch loss")
    parser.add_argument('--rec_w', type=float, default=1, required=False, help="weight for reconstruction loss")
    parser.add_argument('--sv_str', type=str, default='tep_sv', required=False, help="batch size for single GPU")
    parser.add_argument('--ibot_head_share', type=int, default=0, required=False, help="whether to used the shared weight")
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--workers', default=8, type=int, help='number of workers')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for single GPU')
    parser.add_argument('--max_epochs', default=700, type=int, help='number of workers')
    parser.add_argument('--distributed', action='store_true', help='start distributed training')
    parser.add_argument('--noamp', action='store_true', help='use amp')
    parser.add_argument('--logdir', default='tep_sv', type=str, help='logdir name')
    parser.add_argument('--ngpus_per_node', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--base_lr', default=0.00002, type=float, help='base learning rate')

    args = parser.parse_args()
    
    return args


class MaskGenerator:
    def __init__(self, input_size=96, mask_patch_size=8, model_patch_size=2, mask_ratio=0.6):
        self.input_size = input_size  # input image
        self.mask_patch_size = mask_patch_size 
        self.model_patch_size = model_patch_size # image patch size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size # 12 # 6    #  6 for ssim
        self.scale = self.mask_patch_size // self.model_patch_size # 4 # 8 # # 8 for siim
        
        self.token_count = self.rand_size ** 3  # 12*12*12 # # 6*6*6  # 6*6 for simim
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio)) # 27*0.6  # 36*0.6
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count] # 27~ 0.6
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1 # efficient mask
        
        mask = mask.reshape((self.rand_size, self.rand_size, self.rand_size)) # 3*3*3 
        token_mask=mask
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        
        return token_mask, mask


class MIMTransform():

    def __init__(self, crop_num=5):
    #def __init__(self):
        self.transform_img = Compose(
        [
        LoadImaged(keys=["image"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(
            2.0, 2.0, 2.0), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 128)),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(128, 128, 128), random_size=False, num_samples=1),

        RandSpatialCropSamplesd(keys=["image"], roi_size=(96, 96, 96), random_size=False, num_samples=crop_num),

        ]
        )
 
        
        self.mask_generator = MaskGenerator(
            input_size=96,
            mask_patch_size=16,#
            model_patch_size=2,#,
            mask_ratio=0.7,
        )
    
    def __call__(self, img):
        print(img)
        img = self.transform_img(img)
        mask = self.mask_generator()
        mask2 = self.mask_generator()
        #mask3 = self.mask_generator()
        #mask4 = self.mask_generator()
        #mask5 = self.mask_generator()
        
        return img, mask,mask2#,mask3,mask4,mask5


def main():
    args = parse_option()

    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu,args):
    
    #TODO Defining file paths & output directory path

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank,
                                timeout=timedelta(60))
    torch.cuda.set_device(args.gpu)
    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    cls_w=args.cls_w
    patch_w=args.patch_w
    rec_w=args.rec_w
    sv_str=args.sv_str
    ibot_head_share=args.ibot_head_share
    
    json_Path = os.path.normpath('mm_ssl_train.json')
    logdir_path = os.path.normpath(sv_str+'/mm_'+str(cls_w)+'_'+str(patch_w)+'_'+str(rec_w)+'_'+str(args.max_epochs)+'_'+str(args.batch_size*args.ngpus_per_node)+'_'+str(args.base_lr))

    
    epoch_loss_values = []
    lr_values=[]
    step_loss_values = []
    epoch_cl_loss_values = []
    epoch_cl_loss_class_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0

    chunk_num=2
    lr = args.base_lr*args.batch_size

    if (not os.path.exists(logdir_path)) and (args.rank == 0):
        os.mkdir(logdir_path)
    print('info',logdir_path)
    # Load Json & Append Root Path
    with open(json_Path, 'r') as json_f:
        json_Data = json.load(json_f)

    print(json_Data.keys())

    train_Data=json_Data['all_cect'] + json_Data['DeepLesion']


    print('Total Number of Training Data Samples: {}'.format(len(train_Data)))
    #print(train_Data)
    print('#' * 10)
    

    warmup_teacher_temp=0.04
    teacher_temp=0.07
    warmup_teacher_patch_temp=0.04
    teacher_patch_temp=0.07
    ars_epochs=args.max_epochs
    pred_start_epoch=30
    global_crops_number=2
    local_crops_number=0
    out_dimdim=8192
    patch_out_dim=8192
    warmup_teacher_temp_epochs=30

    #teacher_temp= 0.07 

    cls_patch_loss = Cls_Patch_Loss(
        out_dimdim,
        patch_out_dim,
        global_crops_number,
        local_crops_number,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_patch_temp,
        teacher_patch_temp,
        warmup_teacher_temp_epochs,
        ars_epochs,
        lambda1=cls_w,
        lambda2=patch_w,
        mim_start_epoch=pred_start_epoch,
    ).cuda()

    # Set Determinism
    set_determinism(seed=123)

    train_Transforms=MIMTransform(crop_num=2)
    Val_Transforms = train_Transforms

    check_ds = Dataset(data=train_Data, transform=train_Transforms)
    check_sampler = Sampler(check_ds, shuffle=False) if args.distributed else None
    check_loader = data.DataLoader(check_ds,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  sampler=check_sampler,
                                  pin_memory=True,
                                  persistent_workers=True)


    config = CONFIGS_TM['Trans-Small_SMIT_pre_train'] #
    print(config)
    # get the student model  shared weights

    #
    if ibot_head_share:
        model_S = Trans.Trans_SMIT_pre_train_Student(config)
        model_T = Trans.Trans_SMIT_pre_train_Teacher(config)

    else:
        model_S = Trans.Trans_SMIT_pre_train_Student_No_Share(config)
        model_T = Trans.Trans_SMIT_pre_train_Teacher_No_Share(config) 

    model_S.cuda(args.gpu), model_T.cuda(args.gpu)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model_S.cuda(args.gpu)
        model_T.cuda(args.gpu)
        model_S = torch.nn.parallel.DistributedDataParallel(model_S,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)

        model_T = torch.nn.parallel.DistributedDataParallel(model_T,
                                                            device_ids=[args.gpu],
                                                            output_device=args.gpu,
                                                            find_unused_parameters=True)
    else:
        model_S.cuda()
        model_T.cuda()


    # modle_T won't update parameters
    for p in model_T.parameters():
        p.requires_grad = False
    #print (model_S)
    print (model_T)
    print ('info: sharing head nor not ',ibot_head_share)
    # Define Hyper-paramters for training loop


    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model_S.parameters(), lr=lr)

    lrschedule='warmup_cosine'
    if lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=20,
                                                  max_epochs=ars_epochs)
    if lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=ars_epochs)

    train_ds = data.CacheDataset(
                    data=train_Data,
                    transform=train_Transforms,
                    cache_num=500,  #500 is good
                    cache_rate=1.0,
                    num_workers=args.workers,
                )

    train_sampler = Sampler(train_ds, shuffle=False) if args.distributed else None
    train_loader = data.DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  sampler=train_sampler,
                                  pin_memory=True,
                                  persistent_workers=True)

    momentum_teacher=0.996

    momentum_schedule = train_utils.cosine_scheduler(momentum_teacher, 1,
                                            args.max_epochs, len(train_loader))
    
    iter_num=0

    for epoch in range(args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        if args.rank == 0:
            print("-" * 10)
            print(f"epoch {epoch + 1}/{args.max_epochs}")
        model_S.train()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_recon_loss = 0
        epoch_cl_class_loss = 0
        step = 0

        steps_in_epoch=0
        #print the next batch


        for idx, (batch_data, train_mask_all1,train_mask_all2) in enumerate(train_loader):

            token_mask1,mask1=train_mask_all1[0],train_mask_all1[1]
            token_mask2,mask2=train_mask_all2[0],train_mask_all2[1]


            steps_in_epoch=steps_in_epoch+1
            step += 1
            iter_num +=1
            start_time = time.time()
            #print (batch_data)
            image1=batch_data[0][0]["image"]
            image2=batch_data[0][1]["image"]

            img_all=torch.cat((image1,image2),dim=0).cuda(args.rank)
            mask_all=torch.cat((mask1,mask2),dim=0).cuda(args.rank)
                
            token_mask_all=torch.cat((token_mask1,token_mask2),dim=0).cuda(args.rank)
            
            optimizer.zero_grad()

            student_token,x_rec = model_S(img_all,mask_all)
            teacher_token = model_T(img_all)
            #teacher_token_cls = model_T(img_all)

            # calculate the image reconstruction loss for student
            mask_interp = mask_all.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
            loss_recon = F.l1_loss(img_all, x_rec, reduction='none')
            loss_recon = (loss_recon * mask_interp).sum() / (mask_interp.sum() + 1e-5)
            loss_token,loss_token_cls = cls_patch_loss(student_token, teacher_token, token_mask_all, epoch,chunk_num)

            loss=loss_token+loss_token_cls+rec_w*loss_recon

            loss.backward()
            train_utils.cancel_gradients_last_layer(epoch, model_S,1)

            optimizer.step()
            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[steps_in_epoch]  # momentum parameter
                names_q, params_q, names_k, params_k = [], [], [], []
                for name_q, param_q in model_S.named_parameters():
                    names_q.append(name_q)
                    params_q.append(param_q)
                for name_k, param_k in model_T.named_parameters():
                    names_k.append(name_k)
                    params_k.append(param_k)
                names_common = list(set(names_q) & set(names_k))
                
                params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
                params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
                for param_q, param_k in zip(params_q, params_k):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


            val_img_save=image1[0].float()#.cuda()=
            val_img_save=val_img_save.data.cpu().numpy()
            val_img_save=np.squeeze(val_img_save)
            val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))
            pred_sv_name_img=logdir_path+'/train_debug_Input.nii.gz'
            if args.rank==0:
                nib.save(val_img_save, pred_sv_name_img)

            val_img_save=image2[0].float()#.cuda()=
            val_img_save=val_img_save.data.cpu().numpy()
            val_img_save=np.squeeze(val_img_save)
            val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))
            pred_sv_name_img=logdir_path+'/train_debug_Input2.nii.gz'
            if args.rank == 0:
                nib.save(val_img_save, pred_sv_name_img)

            mask_sv = mask_all.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
            val_img_save=mask_sv[0].float()#.cuda()=
            val_img_save=val_img_save.data.cpu().numpy()
            val_img_save=np.squeeze(val_img_save)
            val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))
            pred_sv_name_img=logdir_path+'/train_debug_Mask.nii.gz'
            if args.rank == 0:
                nib.save(val_img_save, pred_sv_name_img)

            val_img_save=x_rec[0].float()#.cuda()=
            val_img_save=val_img_save.data.cpu().numpy()
            val_img_save=np.squeeze(val_img_save)
            val_img_save = nib.Nifti1Image(val_img_save,np.eye(4))
            pred_sv_name_img=logdir_path+'/train_debug_Pred.nii.gz'
            if args.rank == 0:
                nib.save(val_img_save, pred_sv_name_img)

            torch.distributed.reduce(loss, 0), torch.distributed.reduce(loss_token, 0), torch.distributed.reduce(loss_recon, 0), torch.distributed.reduce(loss_token_cls, 0)

            if (args.rank==0):
                total_loss = loss
                cl_loss=loss_token#+loss_token2
                r_loss=loss_recon#+loss_recon2
                cls_loss=loss_token_cls#+loss_token_cls2
                #total_loss.backward()
                #optimizer.step()
                epoch_loss += total_loss.item()
                step_loss_values.append(total_loss.item())

                # CL & Recon Loss Storage of Value
                epoch_cl_loss += cl_loss.item()
                epoch_recon_loss += r_loss.item()
                epoch_cl_class_loss += cls_loss.item()

                if (step %20 ==0):
                    end_time = time.time()
                    print(
                        f"{step}/{len(train_ds) // train_loader.batch_size}, "
                        f"train_loss: {total_loss.item():.4f}, "
                        f"train_loss_token: {cl_loss.item():.4f}, "
                        f"train_loss_cls_token: {cls_loss.item():.4f}, "
                        f"train_loss_recon: {r_loss.item():.4f}, "
                    f"time taken: {end_time-start_time}s")

        if (args.rank == 0):
            epoch_loss /= step
            epoch_cl_loss /= step
            epoch_cl_class_loss /=step
            epoch_recon_loss /= step
            cur_lr=scheduler.get_last_lr()
            lr_values.append(cur_lr)
            epoch_loss_values.append(epoch_loss)
            epoch_cl_loss_values.append(epoch_cl_loss)
            epoch_cl_loss_class_values.append(epoch_cl_class_loss)
            epoch_recon_loss_values.append(epoch_recon_loss)

            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            checkpoint = {'epoch': 1,'state_dict': model_S.module.transformer.state_dict(),'optimizer': optimizer.state_dict()}
            # save the model
            torch.save(checkpoint, os.path.join(logdir_path, 'pre_train_model.pt'))

            # save and print loss
            plt.figure(1, figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epoch_loss_values)
            plt.grid()
            plt.title('Training Loss')

            plt.subplot(2, 2, 2)
            plt.plot(epoch_cl_loss_class_values)
            plt.grid()
            plt.title('Training Class Token Loss')

            plt.subplot(2, 2, 3)
            plt.plot(epoch_cl_loss_values)
            plt.grid()
            plt.title('Training Feature Token Loss')

            plt.subplot(2, 2, 4)
            plt.plot(epoch_recon_loss_values)
            plt.grid()
            plt.title('Training Image Pred Loss')

            plt.savefig(os.path.join(logdir_path, 'Training_loss_plots.png'))
            plt.close(1)

        scheduler.step()
    print('Done')
    return None

if __name__=="__main__":
    main()


