import torch
import torch.nn as nn
import einops
from rtml.models.base_model import BaseModel, BaseTrainer


class Transformer(BaseModel):
    def __init__(self, transformer_params, *args, **kwargs):
        super().__init__(transformer_params, *args, **kwargs)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=22, nhead=2)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.l1 = nn.Linear(1100, 500)
        self.l2 = nn.Linear(500, 100)
        
        # linear_layers = nn.Sequential(nn.Linear())

    def forward(self, x):
        x = einops.rearrange(x, 'n e s -> s n e')
        # print(x.shape)
        x = self.transformer(x)
        # print(x.shape)
        x = einops.rearrange(x, 's n e -> n (s e)')
        # print(x.shape)
        x = self.l1(x)
        x = self.l2(x)
        # print(x.shape)
        return x

class Transformer_Trainer(BaseTrainer):
    def __init__(
            self, model_params, name='Transformer', seed=None, verbose=False, model_dir="out/Transformer",
            notebook_mode=False, model=None, output_normalizer=None, *args, **kwargs
    ):
        super().__init__(model_params, name=name, seed=seed, verbose=verbose, output_normalizer=output_normalizer,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model, *args, **kwargs)
        self.model_class = Transformer
        print(self.model_class)
        self.name = name

if __name__=='__main__':
    n_feat = 128
    b = 64
    transformer_model = nn.Transformer(d_model=n_feat, nhead=16, num_encoder_layers=12)
    src = torch.rand((10, b, n_feat))
    tgt = torch.rand((20, b, n_feat+1))
    out = transformer_model(src, tgt)
    print(out.shape)

    x = torch.rand(50, 64, 22)
    model = Transformer()
    x = model.forward(x)
        
