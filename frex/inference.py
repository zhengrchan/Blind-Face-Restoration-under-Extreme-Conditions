import argparse
from re import A

import cv2
import glob
import numpy as np
import os
import torch
import yaml
from tqdm import tqdm
import json

import archs

from basicsr.archs import build_network
from basicsr.utils.options import ordered_yaml
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils import imwrite
import imageio
from torchvision.transforms.functional import normalize

from camera_utils import LookAtPoseSampler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs/whole_imgs',
                        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('--has_ref', action='store_true', default=False)
    parser.add_argument('--gt', type=str, default='')
    parser.add_argument('-o', '--output', type=str,
                        default='results', help='Output folder. Default: results')
    parser.add_argument('--exp_path', type=str, help='target experiment path')
    parser.add_argument('--iter', type=str, default='300000', help='num of iter used for inference')
    parser.add_argument('--opt', type=str, default='./options/v35_fix_component_loss.yml')
    # parser.add_argument('--arch_version', type=str, default='v5')
    parser.add_argument('--sr_intermediate', action='store_true', default=False, help='Show toRGB images in each sr layers')
    parser.add_argument('--different_pose', action='store_true', default=False)
    parser.add_argument('--use_128', action='store_true', default=False)
    parser.add_argument('--use_3d_sft', action='store_true', default=False)
    parser.add_argument('--lq_3d_path', type=str, default=None)
    parser.add_argument('--crop_param_path', type=str, default=None)
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*.png')))
    
    os.makedirs(args.output, exist_ok=True)

    if args.crop_param_path:
        with open(args.crop_param_path, 'r') as f:
            crop_params = dict(json.load(f))
    else:
        crop_params = None

    # ------------------------ model ------------------------
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    model = build_network(opt['network_g'])

    model_path = os.path.join(args.exp_path, f'models/net_g_{args.iter}.pth')
    model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    model.eval().to(DEVICE)

    # ------------------------ inference ------------------------
    for img_path in tqdm(img_list):
        img_name = os.path.basename(img_path)

        # print(f'Processing {img_name} ...')

        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
       
        cropped_face_t = img2tensor(input_img / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(DEVICE)
        
        if args.lq_3d_path:
            lq_3d_input = cv2.imread(os.path.join(args.lq_3d_path, img_name), cv2.IMREAD_COLOR)
            lq_3d_input = img2tensor(lq_3d_input / 255., bgr2rgb=True, float32=True)
            normalize(lq_3d_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            lq_3d_input = lq_3d_input.unsqueeze(0).to(DEVICE)

        if args.different_pose:
            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=DEVICE).reshape(-1, 9)
            camera_lookat_point = torch.tensor([0, 0, 0.2], device=DEVICE)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, camera_lookat_point, radius=2.7, device=DEVICE)
            camera_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            use_difficulty = False if args.use_128 else True
            front_result = different_pose_forward(model, cropped_face_t, new_camera_params=camera_params, use_difficulty=use_difficulty, use_3d_sft=args.use_3d_sft)

        if args.lq_3d_path:
            result = model(cropped_face_t, lq_3d_input, crop_param=crop_params[img_name])
        else:
            result = model(cropped_face_t)

            
        output = result['image']
        lr_output = result['lr_image']

        imgs = [input_img]
        restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype('uint8')

        def process_lr_img(lr_img, hr_res=512):
            lr_image_eg3d = tensor2img(lr_img.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
            _, h, _ = lr_image_eg3d.shape
            boarder = (hr_res - h) // 2
            lr_image_eg3d = cv2.copyMakeBorder(lr_image_eg3d, boarder, boarder, boarder, boarder, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return lr_image_eg3d
        
        
        boarder_res = 128 if args.use_128 else 512
        imgs.append(process_lr_img(lr_output, hr_res=boarder_res))
        imgs.append(restored_face)
        
        if args.different_pose:
            front_output = front_result['image']
            front_lr_output = front_result['lr_image']
            imgs.append(process_lr_img(front_lr_output, hr_res=boarder_res))

            front_restored_face = tensor2img(front_output.squeeze(0), rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
            imgs.append(front_restored_face)

        if args.has_ref:
            ref_img_path = os.path.join(args.gt, os.path.basename(img_path))
            ref_img = cv2.imread(ref_img_path, cv2.IMREAD_COLOR)
            imgs.append(ref_img)


        # save comparison image
        cmp_img = np.concatenate(imgs, axis=1)
        imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}.png'))
        imwrite(restored_face, os.path.join(args.output, 'restored_faces', f'{basename}.png'))


def trans_to_img(img, drange):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img

def different_pose_forward(model, x, new_camera_params=None, new_difficulty=None, return_rgb=True, randomize_noise=True, use_difficulty=False, use_3d_sft=False):
    conditions = []
    unet_skips = []
    out_rgbs = []
    feat_3d_skip = []

    # encoder
    feat = model.conv_body_first(x)
    for i in range(model.log_size - 2):
        feat = model.conv_body_down[i](feat)
        unet_skips.insert(0, feat)

    feat = model.final_conv(feat)

    # style code
    style_code = model.final_linear(feat.view(feat.size(0), -1))

    if new_camera_params is not None:
        camera_params = new_camera_params
    else:
        camera_params = model.angle_linear(feat.view(feat.size(0), -1)) 
    style_code_sr = model.sr_linear(feat.view(feat.size(0), -1))
    if use_difficulty:
        difficulty = new_difficulty if new_difficulty != None else model.diff_linear(feat.view(feat.size(0), -1))

    # eg3d decoder
    output = model.eg3d_decoder(
        ws=style_code.unsqueeze(1).repeat([1, model.num_ws, 1]),
        c=camera_params,
        neural_rendering_resolution=128,
    )
    # lr_image: b * 3 * 128 * 128
    # feat: b * 32 * 128 * 128
    # conditions: 14list

    feat_3d, lr_image = output['feature'], output['lr_image']

    # feat 3d decoder
    for i in range(model.log_sr_in_size - 3):
        feat_3d = model.feat_3d_encoder[i](feat_3d)
        feat_3d_skip.insert(0, feat_3d.clone())

    if use_difficulty:
        restoration_difficulty = torch.cat((difficulty, camera_params), 1)
    else:
        restoration_difficulty = camera_params
    
    for i in range(model.log_size - 2):
        # add unet skip
        feat = feat + unet_skips[i]
        # ResUpLayer
        feat = model.conv_body_up[i](feat)
        # generate rgb images
        if return_rgb:
            out_rgbs.append(model.toRGB[i](feat))
        # generate scale and shift for SFT layers
        # 3D feat guide 8 ~ 64 resolution
        if i < model.log_sr_in_size - 3:
            feat_concat = torch.cat((feat, feat_3d_skip[i]), 1) if not use_3d_sft else feat_3d_skip[i]
            scale = model.condition_scale[i * 2](feat_concat, restoration_difficulty)
            scale = model.condition_scale[i * 2 + 1](scale, restoration_difficulty)
            conditions.append(scale.clone())

            shift = model.condition_shift[i * 2](feat_concat, restoration_difficulty)
            shift = model.condition_shift[i * 2 + 1](shift, restoration_difficulty)
            conditions.append(shift.clone())
        else:
            scale = model.condition_scale[i * 2](feat, restoration_difficulty)
            scale = model.condition_scale[i * 2 + 1](scale, restoration_difficulty)
            conditions.append(scale.clone())

            shift = model.condition_shift[i * 2](feat, restoration_difficulty)
            shift = model.condition_shift[i * 2 + 1](shift, restoration_difficulty)
            conditions.append(shift.clone())
            
        

    # decoder
    # image, _ = model.stylegan_decoder([style_code],
    #                                  conditions,
    #                                  return_latents=return_latents,
    #                                  input_is_latent=model.input_is_latent,
    #                                  randomize_noise=randomize_noise)

    # grdi_size, grid_c = model.setup_snapshot_image_grid()

    # feat_3d, lr_image = model.eg3d_decoder(
    #     z=style_code,
    #     angles=angle,
    #     cond=torch.zeros_like(angle),
    #     nerf_init_args={'img_size': 64},
    #     noise_mode='const',
    #     )

    
    

    if model.different_w:
        style_code_sr = style_code_sr.view(
            style_code_sr.size(0), -1, model.num_style_feat)

    image, _ = model.sr_decoder([style_code_sr],
                                conditions,
                                input_is_latent=model.input_is_latent,
                                randomize_noise=randomize_noise)
    return {
            'image': image,
            'out_rgbs':out_rgbs, 
            'ws': style_code, 
            'c': camera_params,
            'lr_image': lr_image,
            # 'diff': difficulty,
        }

# no difficulty part
def different_pose_forward_128_v02(model, x, new_camera_params=None, new_difficulty=None, return_rgb=True, randomize_noise=True):
    conditions = []
    unet_skips = []
    out_rgbs = []
    feat_3d_skip = []

    # encoder
    feat = model.conv_body_first(x)
    for i in range(model.log_size - 2):
        feat = model.conv_body_down[i](feat)
        unet_skips.insert(0, feat)

    feat = model.final_conv(feat)

    # style code
    style_code = model.final_linear(feat.view(feat.size(0), -1))
    # camera_params = new_camera_params if new_camera_params != None else model.angle_linear(feat.view(feat.size(0), -1))
    if new_camera_params is not None:
        camera_params = new_camera_params
    else:
        camera_params = model.angle_linear(feat.view(feat.size(0), -1)) 
    style_code_sr = model.sr_linear(feat.view(feat.size(0), -1))
    difficulty = new_difficulty if new_difficulty != None else model.diff_linear(feat.view(feat.size(0), -1))

    # eg3d decoder
    output = model.eg3d_decoder(
        ws=style_code.unsqueeze(1).repeat([1, model.num_ws, 1]),
        c=camera_params,
        # update_emas=False,
        neural_rendering_resolution=128,
        # cache_backbone=False, 
        # use_cached_backbone=False, 
        # **synthesis_kwargs
    )
    # lr_image: b * 3 * 128 * 128
    # feat: b * 32 * 128 * 128
    # conditions: 14list

    feat_3d, lr_image = output['feature'], output['lr_image']

    # feat 3d decoder
    for i in range(model.log_sr_in_size - 3):
        feat_3d = model.feat_3d_encoder[i](feat_3d)
        feat_3d_skip.insert(0, feat_3d.clone())

    # restoration_difficulty = torch.cat((difficulty, camera_params), 1)
    
    for i in range(model.log_size - 2):
        # add unet skip
        feat = feat + unet_skips[i]
        # ResUpLayer
        feat = model.conv_body_up[i](feat)
        # generate rgb images
        if return_rgb:
            out_rgbs.append(model.toRGB[i](feat))
        # generate scale and shift for SFT layers
        # 3D feat guide 8 ~ 64 resolution
        if i < model.log_sr_in_size - 3:
            feat_concat = torch.cat((feat, feat_3d_skip[i]), 1)
            scale = model.condition_scale[i * 2](feat_concat, camera_params)
            scale = model.condition_scale[i * 2 + 1](scale, camera_params)
            conditions.append(scale.clone())

            shift = model.condition_shift[i * 2](feat_concat, camera_params)
            shift = model.condition_shift[i * 2 + 1](shift, camera_params)
            conditions.append(shift.clone())
        else:
            scale = model.condition_scale[i * 2](feat, camera_params)
            scale = model.condition_scale[i * 2 + 1](scale, camera_params)
            conditions.append(scale.clone())

            shift = model.condition_shift[i * 2](feat, camera_params)
            shift = model.condition_shift[i * 2 + 1](shift, camera_params)
            conditions.append(shift.clone())
            
        

    # decoder
    # image, _ = model.stylegan_decoder([style_code],
    #                                  conditions,
    #                                  return_latents=return_latents,
    #                                  input_is_latent=model.input_is_latent,
    #                                  randomize_noise=randomize_noise)

    # grdi_size, grid_c = model.setup_snapshot_image_grid()

    # feat_3d, lr_image = model.eg3d_decoder(
    #     z=style_code,
    #     angles=angle,
    #     cond=torch.zeros_like(angle),
    #     nerf_init_args={'img_size': 64},
    #     noise_mode='const',
    #     )

    
    

    if model.different_w:
        style_code_sr = style_code_sr.view(
            style_code_sr.size(0), -1, model.num_style_feat)

    image, _ = model.sr_decoder([style_code_sr],
                                conditions,
                                input_is_latent=model.input_is_latent,
                                randomize_noise=randomize_noise)
    return {
            'image': image,
            'out_rgbs':out_rgbs, 
            'ws': style_code, 
            'c': camera_params,
            'lr_image': lr_image,
            'diff': difficulty,
        }

# def imwrite(img, file_path, params=None, auto_mkdir=True):
#     if auto_mkdir:
#         dir_name = os.path.abspath(os.path.dirname(file_path))
#         os.makedirs(dir_name, exist_ok=True)
#     imageio.imwrite(file_path, img)


if __name__ == '__main__':
    main()
