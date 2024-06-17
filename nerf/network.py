from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class BoostedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias, omega):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.omega = omega
    def forward(self, x, **kwargs):
        return self.linear(x * self.omega)

class FinerActivation(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
    def generate_scale(self, x):
        with torch.no_grad():
            scale = torch.abs(x) + 1
        return scale

    def forward(self, x):
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out

class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 omega_0=30.0, first_bias_scale=None,
                 scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)


    def forward(self, input):
        return self.linear(input)



class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=4,
                 hidden_dim=182,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=182,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 embedding='pos',
                 sigma=1.0,
                 activation='ReLU',
                 omega=30.0,
                 omega_finer=30.0,
                 desired_resolution=2048,
                 level_dim=2,
                 num_levels=16,
                 log2_hashmap_size=19,
                 multires=10,
                 hashmap_high_values=False,
                 finer_high_values=False,
                 finer_k=None,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        if activation== 'ReLU':
            self.nonlinearity = partial(F.relu, inplace=True)
        elif activation == 'Sine':
            self.nonlinearity = torch.sin
        elif activation == 'Finer':
            self.nonlinearity = FinerActivation()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        # self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.first_nonlinearity = self.nonlinearity
        if activation == 'Finer':
            self.first_nonlinearity = FinerActivation(omega_finer)


        if embedding == 'pos':
            self.encoder, self.in_dim = get_encoder("frequency", multires=multires)
        elif embedding == 'id':
            self.encoder, self.in_dim = get_encoder("None", omega=omega)
        elif embedding == 'hashgrid':
            self.encoder, self.in_dim = get_encoder("hashgrid",
                                                    desired_resolution=desired_resolution,
                                                    log2_hashmap_size=log2_hashmap_size,
                                                    level_dim=level_dim,
                                                    num_levels=num_levels,
                                                    hashmap_high_values=hashmap_high_values)
        elif embedding == 'rff':
            import rff
            encoded_size = 256
            class GaussianEncoding(rff.layers.GaussianEncoding):
                def forward(self, x, bound):
                    return super().forward(x)

            self.encoder, self.in_dim = GaussianEncoding(
                    sigma=sigma, input_size=3, encoded_size=encoded_size
                ), encoded_size * 2

        bias = (activation == "Sine") or (activation == "Finer")
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim


            if activation not in {'Sine', 'Finer'}:
                sigma_net.append(nn.Linear(in_dim, out_dim, bias=bias))
            elif activation == 'Sine':
                hidden_omega = 30
                if l == 0:
                    hidden_omega = 1 # omega0 is in the embedding layer
                sigma_net.append(BoostedLinear(in_dim, out_dim, bias=bias, omega=hidden_omega))

                # use siren weight initialization
                if l == 0:
                    torch.nn.init.uniform_(sigma_net[-1].linear.weight, -1 / in_dim, 1 / in_dim)
                else:
                    l = -np.sqrt(6 / in_dim) / 30
                    r = np.sqrt(6 / in_dim) / 30
                    torch.nn.init.uniform_(sigma_net[-1].linear.weight, l, r)
            elif activation == 'Finer':
                sigma_net.append(FinerLayer(in_dim, out_dim, bias=True, is_first=l==0,
                                             omega_0=30.0, first_bias_scale=finer_k,
                                             scale_req_grad=False))


        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color


            if activation not in {'Sine', 'Finer'}:
                color_net.append(nn.Linear(in_dim, out_dim, bias=bias))
            elif activation == 'Sine':
                color_net.append(BoostedLinear(in_dim, out_dim, bias=bias, omega=30))

                torch.nn.init.uniform_(color_net[-1].linear.weight, -np.sqrt(6 / in_dim) / 30, np.sqrt(6 / in_dim) / 30)
            elif activation == 'Finer':
                hidden_omega_0 = 30.0
                if l < num_layers_color - 1:
                    color_net.append(FinerLayer(in_dim, out_dim,
                                                bias=True,
                                                is_first=False,
                                                omega_0=hidden_omega_0,
                                                first_bias_scale=None,
                                                scale_req_grad=False))
                else:
                    final_linear = nn.Linear(in_dim, out_dim)
                    with torch.no_grad():
                        final_linear.weight.uniform_(-np.sqrt(6 / in_dim) / hidden_omega_0,
                                                     np.sqrt(6 / in_dim) / hidden_omega_0)
                    if finer_high_values:
                        with torch.no_grad(): # initialise with higher values
                            final_linear.weight.uniform_(-np.sqrt(6 / in_dim),
                                                         np.sqrt(6 / in_dim))
                    color_net.append(final_linear)

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l == 0 and l != self.num_layers - 1:
                h = self.first_nonlinearity(h)
            elif l != self.num_layers - 1:
                h = self.nonlinearity(h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = self.nonlinearity(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l == 0 and l != self.num_layers - 1:
                h = self.first_nonlinearity(h)
            elif l != self.num_layers - 1:
                h = self.nonlinearity(h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = self.nonlinearity(h)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, use_viewdirs=True, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        if not use_viewdirs:
            # For FreSh, we don't use viewdirs at initialisation
            d = torch.zeros_like(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = self.nonlinearity(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
