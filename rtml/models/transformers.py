from typing import Dict, Union

import torch
import torch.nn as nn
import einops
from einops import repeat
from torch import Tensor

from rtml.models.MLP import MLPNet
from rtml.models.base_model import BaseModel, BaseTrainer
from rtml.models.column_handler import ColumnPreprocesser


class Transformer(BaseModel):
    def __init__(
            self,
            input_dim: Union[int, Dict[str, int]],
            out_dim: int,
            column_preprocesser: ColumnPreprocesser,
            n_layers: int = 5,  # 10
            hidden_dim: int = 256,  # 512
            nhead: int = 4,  # 8
            dim_feedforward: int = 512,  # 1024
            activation_function: str = 'Gelu',
            dropout: float = 0.0,
            norm_first: bool = False,
            use_out_norm: bool = True,
            out_pooling='cls',
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_preprocesser = column_preprocesser
        self.preprocess_func = self.column_preprocesser.get_preprocesser()
        raw_feature_in_dim = self.column_preprocesser.out_dim
        if raw_feature_in_dim != hidden_dim:
            self.lin_projector = nn.Linear(raw_feature_in_dim, hidden_dim, bias=True)
            self.preprocess_func1 = self.preprocess_func
            self.preprocess_func = lambda x: self.lin_projector(self.preprocess_func1(x))

        self.pos_embedding = nn.Parameter(torch.randn(1, column_preprocesser.n_lay + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        print(torch.__version__, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation_function.lower(),
          #  norm_first=norm_first,
          #  batch_first=True,
        )

        out_norm = nn.LayerNorm(hidden_dim) if use_out_norm else None
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=out_norm)

        self.out_pooling = out_pooling
        self.readout = MLPNet(
            input_dim=hidden_dim,
            hidden_dims=[256 for _ in range(1)],
            out_dim=100,
            activation_function='Gelu',
            net_normalization=None, #'layer_norm',
            dropout=0,
            output_normalization=False,
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        x = self.preprocess_func(x)

        b, n, _ = x.shape  # (batch-size, spatial-dim, hidden-dim), e.g. torch.Size([64, 49, 512])
        # print(b, n, self.cls_token.shape) # 64 49
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # (1, 1, 512) -> (64, 1, 512)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = einops.rearrange(x, 'b n d -> n b d')
        x = self.transformer(x) # torch.Size([50, 64, 512])
        x = x.mean(dim=0) if self.out_pooling == 'mean' else x[0, :, :]

        y = self.readout(x)         # torch.Size([64, 100])
        return y

    def forwardBfirst(self, x: Dict[str, Tensor]) -> Tensor:
        x = self.preprocess_func(x)  # (batch-size, spatial-dim, hidden-dim)
        print(x.shape)
        b, n, _ = x.shape
        print(b, n, self.cls_token.shape)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)
        print(x.shape)
        x = x.mean(dim=1) if self.out_pooling == 'mean' else x[:, 0]

        y = self.readout(x)
        print(y.shape)
        return y

class Transformer_Trainer(BaseTrainer):
    def __init__(
            self, model_params, column_preprocesser, name='Transformer', seed=None, verbose=False, model_dir="out/Transformer",
            notebook_mode=False, model=None, output_normalizer=None, *args, **kwargs
    ):
        super().__init__(model_params, name=name, seed=seed, verbose=verbose, output_normalizer=output_normalizer,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model, *args, **kwargs)
        self.model_class = Transformer
        print(self.model_class)
        self.name = name

        self.column_preprocesser = column_preprocesser

    def _get_model(self, get_new=False):
        if self.model is None or get_new:
            model = self.model_class(
                **self.model_params, column_preprocesser=self.column_preprocesser, normalizer=self.output_postprocesser
            )
            return model.to(self._device).float()
        return self.model


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
        
