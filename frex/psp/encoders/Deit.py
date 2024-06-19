import torch
import timm
import torch.nn as nn
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor


class CrossAttention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class CrossAttention_2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, batch_first=True):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, ):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)  
        return tgt



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")







class MyViT(nn.Module):
    def __init__(self, target_size, model_name, pretrained=False):
        super(MyViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        # 改成自己任务的图像类别数
        self.model.head = nn.Linear(n_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x

class MyDeiT(nn.Module):
    def __init__(self, target_size, pretrained=False):
        super(MyDeiT, self).__init__()
        self.model = torch.hub.load('facebookresearch/deit:main',
                                    'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained)

        n_features = self.model.head.in_features
        # 改成自己任务的图像类别数
        print(n_features)
        self.model.head = nn.Linear(n_features, target_size)
        self.model.head_dist = nn.Linear(n_features, target_size)

    def forward(self, x):
        x, x_dist = self.model(x)
        return x, x_dist




class DeiTEncoder(nn.Module):
    def __init__(self, target_size, mlp_layer=2, dropout=0.1, pretrained=False):
        super(DeiTEncoder, self).__init__()
        self.deit_coarse = torch.hub.load(
                                        'facebookresearch/deit:main',
                                        'deit_base_patch16_224',
                                        # 'deit_base_distilled_patch16_224',
                                    # 'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained
                                    )
        self.deit_mid= torch.hub.load(
                                        'facebookresearch/deit:main',
                                        'deit_base_patch16_224',
                                        # 'deit_base_distilled_patch16_224',
                                    # 'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained
                                    )
        self.deit_fine= torch.hub.load(
                                        'facebookresearch/deit:main',
                                        'deit_base_patch16_224',
                                        # 'deit_base_distilled_patch16_224',
                                    # 'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained
                                    )


        n_features_coarse = self.deit_coarse.head.in_features
        n_features_mid = self.deit_mid.head.in_features
        n_features_fine = self.deit_fine.head.in_features

        self.deit_coarse.head = nn.Linear(n_features_coarse, target_size)
        self.deit_coarse.head_dist = nn.Linear(n_features_coarse, target_size)


        self.deit_mid.head = nn.Linear(n_features_mid, target_size)
        self.deit_mid.head_dist = nn.Linear(n_features_mid, target_size)

        self.deit_fine.head = nn.Linear(n_features_fine, target_size)
        self.deit_fine.head_dist = nn.Linear(n_features_fine, target_size)


        self.mapper_coarse_list = nn.ModuleList()
        for i in range(7):
            module_list = nn.ModuleList()
            for j in range(mlp_layer):
                module_list.append(nn.Linear(target_size, target_size))
                module_list.append(nn.LeakyReLU())
            self.mapper_coarse_list.append(nn.Sequential(*module_list))


        self.mapper_mid_list = nn.ModuleList()
        for i in range(4):
            module_list = nn.ModuleList()
            for j in range(mlp_layer):
                module_list.append(nn.Linear(target_size, target_size))
                module_list.append(nn.LeakyReLU())
            self.mapper_mid_list.append(nn.Sequential(*module_list))


        self.mapper_fine_list = nn.ModuleList()
        for i in range(3):
            module_list = nn.ModuleList()
            for j in range(mlp_layer):
                module_list.append(nn.Linear(target_size, target_size))
                module_list.append(nn.LeakyReLU())
            self.mapper_fine_list.append(nn.Sequential(*module_list))





        self.SelfAtt_coarse = nn.MultiheadAttention(target_size, num_heads=4, dropout=dropout, batch_first=True)
        self.CrossAtt_mid = nn.MultiheadAttention(target_size, num_heads=4, dropout=dropout, batch_first=True)
        self.CrossAtt_fine = nn.MultiheadAttention(target_size, num_heads=4, dropout=dropout, batch_first=True)


        module_list = nn.ModuleList()
        for j in range(mlp_layer):
            module_list.append(nn.Linear(target_size, target_size))
            module_list.append(nn.LeakyReLU())
        self.after_att_mapper_coarse = nn.Sequential(*module_list)

        self.norm_coarse = nn.LayerNorm(target_size, elementwise_affine=False)
        self.dropout_coarse = nn.Dropout(dropout)


        module_list = nn.ModuleList()
        for j in range(mlp_layer):
            module_list.append(nn.Linear(target_size, target_size))
            module_list.append(nn.LeakyReLU())
        self.after_att_mapper_mid = nn.Sequential(*module_list)

        self.norm_mid = nn.LayerNorm(target_size, elementwise_affine=False)
        self.dropout_mid = nn.Dropout(dropout)


        module_list = nn.ModuleList()
        for j in range(mlp_layer):
            module_list.append(nn.Linear(target_size, target_size))
            module_list.append(nn.LeakyReLU())
        self.after_att_mapper_fine = nn.Sequential(*module_list)

        self.norm_fine = nn.LayerNorm(target_size, elementwise_affine=False)
        self.dropout_fine = nn.Dropout(dropout)



    def forward(self, x):

        ww = []

        # x_coarse, x_dist_coarse = self.deit_coarse(x)
        # latent_base_coarse = torch.stack((x_coarse, x_dist_coarse)).unsqueeze(1)

        latent_base_coarse = self.deit_coarse(x).unsqueeze(1)

        for mapper in self.mapper_coarse_list:
            ww.append(mapper(latent_base_coarse))



        # x_mid, x_dist_mid = self.deit_mid(x)
        # latent_base_mid = torch.stack((x_mid, x_dist_mid)).unsqueeze(1)

        latent_base_mid = self.deit_mid(x).unsqueeze(1)

        for mapper in self.mapper_mid_list:
            ww.append(mapper(latent_base_mid))

        # x_fine, x_dist_fine = self.deit_fine(x)
        # latent_base_fine = torch.stack((x_fine, x_dist_fine)).unsqueeze(1)

        latent_base_fine = self.deit_fine(x).unsqueeze(1)

        for mapper in self.mapper_fine_list:
            ww.append(mapper(latent_base_fine))

        
        

        # ws_14 = torch.cat(ww,dim=1)

        
        base_0_7 = torch.cat(ww[:7],dim=1)
        base_7_11 = torch.cat(ww[7:11],dim=1)
        base_11_14 = torch.cat(ww[11:],dim=1)

        # ww_new = []

        query_0_7 = self.norm_coarse(base_0_7)
        k_0_7 = v_0_7 = base_0_7
        ws_14_0_7 = self.SelfAtt_coarse(query=query_0_7, key=k_0_7, value=v_0_7)[0]
        base_0_7 = base_0_7 + self.dropout_coarse(ws_14_0_7)
        base_0_7 = self.after_att_mapper_coarse(base_0_7)

        query_7_11 = self.norm_mid(base_7_11)
        k_7_11 = v_7_11 = base_0_7
        ws_14_7_11 = self.SelfAtt_coarse(query=query_7_11, key=k_7_11, value=v_7_11)[0]

        base_7_11 = base_7_11 + self.dropout_mid(ws_14_7_11) 
        base_7_11 = self.after_att_mapper_mid(base_7_11)



        query_11_14 = self.norm_fine(base_11_14)
        k_11_14 = v_11_14 = base_7_11

        ws_14_11_14 = self.SelfAtt_coarse(query=query_11_14, key=k_11_14, value=v_11_14)[0]
        base_11_14 = base_11_14 + self.dropout_fine(ws_14_11_14)
        base_11_14 = self.after_att_mapper_fine(base_11_14)



        rec_ws_14 = torch.cat([base_0_7, base_7_11, base_11_14], dim=1)

        return rec_ws_14





class DeiTEncoder_dist(nn.Module):
    def __init__(self, target_size, mlp_layer=2, dropout=0.1, pretrained=False):
        super(DeiTEncoder_dist, self).__init__()
        self.deit_coarse = torch.hub.load(
                                        'facebookresearch/deit:main',
                                        # 'deit_base_patch16_224',
                                        'deit_base_distilled_patch16_224',
                                    # 'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained
                                    )
        self.deit_mid= torch.hub.load(
                                        'facebookresearch/deit:main',
                                        # 'deit_base_patch16_224',
                                        'deit_base_distilled_patch16_224',
                                    # 'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained
                                    )
        self.deit_fine= torch.hub.load(
                                        'facebookresearch/deit:main',
                                        # 'deit_base_patch16_224',
                                        'deit_base_distilled_patch16_224',
                                    # 'deit_tiny_distilled_patch16_224', 
                                    pretrained=pretrained
                                    )


        n_features_coarse = self.deit_coarse.head.in_features
        n_features_mid = self.deit_mid.head.in_features
        n_features_fine = self.deit_fine.head.in_features

        self.deit_coarse.head = nn.Linear(n_features_coarse, target_size)
        self.deit_coarse.head_dist = nn.Linear(n_features_coarse, target_size)


        self.deit_mid.head = nn.Linear(n_features_mid, target_size)
        self.deit_mid.head_dist = nn.Linear(n_features_mid, target_size)

        self.deit_fine.head = nn.Linear(n_features_fine, target_size)
        self.deit_fine.head_dist = nn.Linear(n_features_fine, target_size)


        self.mapper_coarse_list = nn.ModuleList()
        for i in range(7):
            module_list = nn.ModuleList()
            for j in range(mlp_layer):
                module_list.append(nn.Linear(target_size, target_size))
                module_list.append(nn.LeakyReLU())
            self.mapper_coarse_list.append(nn.Sequential(*module_list))


        self.mapper_mid_list = nn.ModuleList()
        for i in range(4):
            module_list = nn.ModuleList()
            for j in range(mlp_layer):
                module_list.append(nn.Linear(target_size, target_size))
                module_list.append(nn.LeakyReLU())
            self.mapper_mid_list.append(nn.Sequential(*module_list))


        self.mapper_fine_list = nn.ModuleList()
        for i in range(3):
            module_list = nn.ModuleList()
            for j in range(mlp_layer):
                module_list.append(nn.Linear(target_size, target_size))
                module_list.append(nn.LeakyReLU())
            self.mapper_fine_list.append(nn.Sequential(*module_list))





        self.SelfAtt_coarse = nn.MultiheadAttention(target_size, num_heads=4, dropout=dropout, batch_first=True)
        self.CrossAtt_mid = nn.MultiheadAttention(target_size, num_heads=4, dropout=dropout, batch_first=True)
        self.CrossAtt_fine = nn.MultiheadAttention(target_size, num_heads=4, dropout=dropout, batch_first=True)


        module_list = nn.ModuleList()
        for j in range(mlp_layer):
            module_list.append(nn.Linear(target_size, target_size))
            module_list.append(nn.LeakyReLU())
        self.after_att_mapper_coarse = nn.Sequential(*module_list)

        self.norm_coarse = nn.LayerNorm(target_size, elementwise_affine=False)
        self.dropout_coarse = nn.Dropout(dropout)


        module_list = nn.ModuleList()
        for j in range(mlp_layer):
            module_list.append(nn.Linear(target_size, target_size))
            module_list.append(nn.LeakyReLU())
        self.after_att_mapper_mid = nn.Sequential(*module_list)

        self.norm_mid = nn.LayerNorm(target_size, elementwise_affine=False)
        self.dropout_mid = nn.Dropout(dropout)


        module_list = nn.ModuleList()
        for j in range(mlp_layer):
            module_list.append(nn.Linear(target_size, target_size))
            module_list.append(nn.LeakyReLU())
        self.after_att_mapper_fine = nn.Sequential(*module_list)

        self.norm_fine = nn.LayerNorm(target_size, elementwise_affine=False)
        self.dropout_fine = nn.Dropout(dropout)



    def forward(self, x):
        B = x.shape[0]

        ww = []

        x_coarse, x_dist_coarse = self.deit_coarse(x)
        latent_base_coarse = torch.stack((x_coarse, x_dist_coarse)).reshape(B,2,-1).sum(1).unsqueeze(1)

        # latent_base_coarse = self.deit_coarse(x).unsqueeze(1)

        for mapper in self.mapper_coarse_list:
            ww.append(mapper(latent_base_coarse))



        x_mid, x_dist_mid = self.deit_mid(x)
        latent_base_mid = torch.stack((x_mid, x_dist_mid)).reshape(B,2,-1).sum(1).unsqueeze(1)

        # latent_base_mid = self.deit_mid(x).unsqueeze(1)

        for mapper in self.mapper_mid_list:
            ww.append(mapper(latent_base_mid))

        x_fine, x_dist_fine = self.deit_fine(x)
        latent_base_fine = torch.stack((x_fine, x_dist_fine)).reshape(B,2,-1).sum(1).unsqueeze(1)

        # latent_base_fine = self.deit_fine(x).unsqueeze(1)

        for mapper in self.mapper_fine_list:
            ww.append(mapper(latent_base_fine))

        
        

        # ws_14 = torch.cat(ww,dim=1)

        
        base_0_7 = torch.cat(ww[:7],dim=1)
        base_7_11 = torch.cat(ww[7:11],dim=1)
        base_11_14 = torch.cat(ww[11:],dim=1)

        # ww_new = []

        query_0_7 = self.norm_coarse(base_0_7)
        k_0_7 = v_0_7 = base_0_7
        ws_14_0_7 = self.SelfAtt_coarse(query=query_0_7, key=k_0_7, value=v_0_7)[0]
        base_0_7 = base_0_7 + self.dropout_coarse(ws_14_0_7)
        base_0_7 = self.after_att_mapper_coarse(base_0_7)

        query_7_11 = self.norm_mid(base_7_11)
        k_7_11 = v_7_11 = base_0_7
        ws_14_7_11 = self.SelfAtt_coarse(query=query_7_11, key=k_7_11, value=v_7_11)[0]

        base_7_11 = base_7_11 + self.dropout_mid(ws_14_7_11) 
        base_7_11 = self.after_att_mapper_mid(base_7_11)



        query_11_14 = self.norm_fine(base_11_14)
        k_11_14 = v_11_14 = base_7_11

        ws_14_11_14 = self.SelfAtt_coarse(query=query_11_14, key=k_11_14, value=v_11_14)[0]
        base_11_14 = base_11_14 + self.dropout_fine(ws_14_11_14)
        base_11_14 = self.after_att_mapper_fine(base_11_14)



        rec_ws_14 = torch.cat([base_0_7, base_7_11, base_11_14], dim=1)

        return rec_ws_14