# Blind Face Restoration under Extreme Conditions: Leveraging 3D-2D Prior Fusion for Superior Structural and Texture Recovery(AAAI2024)

## Overview
FREx(Face Restoration under Extreme conditions) aims at incorporating 3D structural priors and 2D texture priors into the restoration process, leading to high-quality restoration results under extreme conditions.

![teaser](assets/teaser2.png)

## Dependences
- CUDA Version: 12.0
- python == 3.8.13
- pytorch == 1.10.0
- basicsr == 1.3.5

## Model Weights
- Pretrained_weights: [baidu drive](https://pan.baidu.com/s/1mZtJ_LXBW4J64F59RfMM_w?pwd=srts)
- FREx: [baidu drive](https://pan.baidu.com/s/1Sae37KA97fIchyqJYZS3Og?pwd=jnuh)
- Testing dataset: [baidu drive](https://pan.baidu.com/s/1QVUO2O0uf7jzVE4cIvUf7g?pwd=6ct6)

## Inference
```
python3 fr3d/inference.py 
--input ./datasets/CelebAHQ_test_balanced_pose_v2/lq_512/ 
--lq_3d_path ./datasets/CelebAHQ_test_balanced_pose_v2/lq_aligned_256  
--output ./inference_results/
--exp_path ./experiments/v12_inv 
--iter 230000 
--opt ./options/inv_v12.yml 
--crop_param_path ./datasets/celeba_crop_params.json
```

If you have any questions, feel free to contact zhengrchan@gmail.com






