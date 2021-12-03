import torch
from torch import nn
from modules.transformer import TransformerEncoder

class MULTModel(nn.Module):
        def __init__(self, hyp_params):
            
            super(MULTModel, self).__init__()
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
            self.d_l, self.d_a, self.d_v = 30, 30, 30
            self.num_heads = hyp_params.num_heads
            self.layers = hyp_params.layers
            self.attn_dropout = hyp_params.attn_dropout
            self.attn_dropout_a = hyp_params.attn_dropout_a
            self.attn_dropout_v = hyp_params.attn_dropout_v
            self.relu_dropout = hyp_params.relu_dropout
            self.res_dropout = hyp_params.res_dropout
            self.out_dropout = nn.Dropout(p=hyp_params.out_dropout)
            self.relu = nn.ReLU()
            self.embed_dropout = nn.Dropout(p=hyp_params.embed_dropout)
            self.attn_mask = hyp_params.attn_mask

            combined_dim = 2 * self.d_l

            # 1. Temporal convolutional layers
            self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
            self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
            self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

            # Crossmodel Attentions
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')

            # self.trans_a_with_l = self.get_network(self_type='al')
            # self.trans_a_with_v = self.get_network(self_type='av')

            # self.trans_v_with_l = self.get_network(self_type='vl')
            # self.trans_v_with_a = self.get_network(self_type='va')

            # Self Attention
            self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
            # self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
            # self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

            output_dim = hyp_params.output_dim

            # Projection layers
            self.proj1 = nn.Linear(combined_dim, combined_dim)
            self.proj2 = nn.Linear(combined_dim, combined_dim)
            self.out_layer = nn.Linear(combined_dim, output_dim)

        def get_network(self, self_type='l', layers=-1):
            if self_type in ['l', 'al', 'vl']:
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type in ['a', 'la', 'va']:
                embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
            elif self_type in ['v', 'lv', 'av']:
                embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
            elif self_type == 'l_mem':
                embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
            elif self_type == 'a_mem':
                embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
            elif self_type == 'v_mem':
                embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
            else:
                raise ValueError("Unknown network type")
            
            return TransformerEncoder(embed_dim=embed_dim,
                                    num_heads=self.num_heads,
                                    layers=max(self.layers, layers),
                                    attn_dropout=attn_dropout,
                                    relu_dropout=self.relu_dropout,
                                    res_dropout=self.res_dropout,
                                    embed_dropout=self.embed_dropout,
                                    attn_mask=self.attn_mask)
    
        def forward(self, x_l, x_a, x_v, mask_l, mask_a, mask_v, device):

            x_l = self.embed_dropout(x_l.transpose(1,2))
            x_a = x_a.transpose(1,2)
            x_v = x_v.transpose(1,2)

            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
            proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

            proj_x_a = proj_x_a.permute(2, 0, 1)
            proj_x_v = proj_x_v.permute(2, 0, 1)
            proj_x_l = proj_x_l.permute(2, 0, 1)

            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(device, proj_x_l, proj_x_a, proj_x_a, mask_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(device, proj_x_l, proj_x_v, proj_x_v, mask_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(device, x_in=h_ls, src_key_padding_mask=mask_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

            # # (L,V) --> A
            # h_a_with_ls = self.trans_a_with_l(device, proj_x_a, proj_x_l, proj_x_l, mask_l)
            # h_a_with_vs = self.trans_a_with_v(device, proj_x_a, proj_x_v, proj_x_v, mask_v)
            # h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            # h_as = self.trans_a_mem(device, x_in=h_as, src_key_padding_mask=mask_a)
            # if type(h_as) == tuple:
            #     h_as = h_as[0]
            # last_h_a = last_hs = h_as[-1]

            # # (L,A) --> V
            # h_v_with_ls = self.trans_v_with_l(device, proj_x_v, proj_x_l, proj_x_l, mask_l)
            # h_v_with_as = self.trans_v_with_a(device, proj_x_v, proj_x_a, proj_x_a, mask_a)
            # h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            # h_vs = self.trans_v_mem(device, x_in=h_vs, src_key_padding_mask=mask_v)
            # if type(h_vs) == tuple:
            #     h_vs = h_vs[0]
            # last_h_v = last_hs = h_vs[-1]

            last_hs = last_h_l

            last_hs_proj = self.proj2(self.out_dropout(self.relu(self.proj1(last_hs))))
            last_hs_proj += last_hs

            output = self.out_layer(last_hs_proj)
            return output
