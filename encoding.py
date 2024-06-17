import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_hash_encoder(input_dim, num_levels, level_dim,
                     base_resolution, log2_hashmap_size, desired_resolution,
                     align_corners, hashmap_high_values):
    from gridencoder import GridEncoder
    encoders = []
    while level_dim > 0:
        encoders.append(GridEncoder(input_dim=input_dim,
                              num_levels=num_levels,  ##
                              level_dim=min(level_dim, 8),
                              base_resolution=base_resolution,  #
                              log2_hashmap_size=log2_hashmap_size,  ##
                              desired_resolution=desired_resolution,  ##
                              gridtype='hash', align_corners=align_corners,
                                    hashmap_high_values=hashmap_high_values))
        level_dim -= 8

    class GridEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList(encoders)
            self.output_dim = sum([encoder.output_dim for encoder in self.encoders])

        def forward(self, x, *args, **kwargs):
            out = []
            for encoder in self.encoders:
                out.append(encoder(x, *args, **kwargs))
            return torch.cat(out, dim=-1)
    return GridEncoder()


def get_encoder(encoding, input_dim=3, 
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2,
                base_resolution=16, log2_hashmap_size=19,
                desired_resolution=2048, align_corners=False,
                omega=1, hashmap_high_values=False,
                **kwargs):

    if encoding == 'None':
        class Identity(torch.nn.Module):
            def forward(self, x, **kwargs):
                return omega*x
        return Identity(), input_dim
    
    elif encoding == 'frequency':
        #encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        encoder = get_hash_encoder(input_dim, num_levels, level_dim,
                     base_resolution, log2_hashmap_size, desired_resolution,
                     align_corners, hashmap_high_values)
    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    
    elif encoding == 'ash':
        from ashencoder import AshEncoder
        encoder = AshEncoder(input_dim=input_dim, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim