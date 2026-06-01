import timm
import math
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .. import lorentz as Lorentz
from ..utils import distributed as dist
from ..modules.encoder import ResMLPEncoder


class HHAlignment(nn.Module):


    def __init__(
        self,
        image_dim: int,
        gene_dim: int,
        embed_dim: int,
        mlp_ratio:int,
        image_dropout:float = 0.1,
        gene_dropout:float = 0.1,
        entail_weight: float = 0.2,
        niche_project: bool = True,
        predict_norm:bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_encoder = ResMLPEncoder(in_features=image_dim, hidden_size=embed_dim,mlp_ratio=mlp_ratio,drop=image_dropout,depth=2)
        self.gene_encoder = ResMLPEncoder(in_features=gene_dim, hidden_size=embed_dim,mlp_ratio=mlp_ratio,drop=gene_dropout,depth=2)
        self.image_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gene_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.normal_(self.image_proj.weight, std=embed_dim **-0.5)
        nn.init.normal_(self.gene_proj.weight, std=embed_dim **-0.5)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        self._rank = dist.get_rank()

        self.niche_project = niche_project
        self.predict_norm = predict_norm

        if self.niche_project:
            self.ncihe_image_encoder = ResMLPEncoder(in_features=image_dim, hidden_size=embed_dim,mlp_ratio=mlp_ratio,drop=image_dropout,depth=2)
            self.ncihe_gene_encoder = ResMLPEncoder(in_features=gene_dim, hidden_size=embed_dim,mlp_ratio=mlp_ratio,drop=gene_dropout,depth=2)
            self.niche_image_proj = self.image_proj
            self.niche_gene_proj = self.gene_proj
            
        else:
            self.ncihe_image_encoder = self.image_encoder
            self.ncihe_gene_encoder = self.gene_encoder
            self.niche_image_proj = self.image_proj
            self.niche_gene_proj = self.gene_proj
        
        
        
        self.curv = nn.Parameter(
            torch.tensor(1.0).log(), requires_grad=True
        )

        self._curv_minmax = {
            "max": math.log(1.0 * 10),
            "min": math.log(1.0 / 10),
        }


        self.entail_weight = entail_weight

        self.image_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.gene_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())


    @property
    def device(self) -> torch.device:
        return self.logit_scale.device
    
    def forward(
        self, 
        image_emb: torch.Tensor, niche_image_emb: torch.Tensor,
        gene_emb: torch.Tensor, niche_gene_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()



        self.image_alpha.data = torch.clamp(self.image_alpha.data, max=0.0)
        self.gene_alpha.data = torch.clamp(self.gene_alpha.data, max=0.0)

        image_emb = self.image_encoder(image_emb)
        image_feats = self.image_proj(image_emb)
        gene_emb = self.gene_encoder(gene_emb)
        gene_feats = self.gene_proj(gene_emb)
        niche_image_emb = self.ncihe_image_encoder(niche_image_emb)
        niche_image_feats = self.niche_image_proj(niche_image_emb)
        niche_gene_emb = self.ncihe_gene_encoder(niche_gene_emb)
        niche_gene_feats = self.niche_gene_proj(niche_gene_emb)
        image_feats_inner = image_feats * self.image_alpha.exp()
        gene_feats_inner = gene_feats * self.gene_alpha.exp()

        niche_image_feats_inner = niche_image_feats * self.image_alpha.exp()
        niche_gene_feats_inner = niche_gene_feats * self.gene_alpha.exp()

        with torch.autocast(self.device.type, dtype=torch.float32):
            image_feats_L = Lorentz.exp_map0(image_feats_inner, self.curv.exp())
            gene_feats_L = Lorentz.exp_map0(gene_feats_inner, self.curv.exp())
            niche_image_feats_L = Lorentz.exp_map0(niche_image_feats_inner, self.curv.exp())
            niche_gene_feats_L = Lorentz.exp_map0(niche_gene_feats_inner, self.curv.exp())

        all_image_feats_L = dist.gather_across_processes(image_feats_L)
        all_gene_feats_L = dist.gather_across_processes(gene_feats_L)

        all_batch_size = [i.shape[0] for i in all_image_feats_L]
        all_batch_size = np.cumsum(all_batch_size)[:-1]

        all_batch_size = [0] + all_batch_size.tolist()

        all_image_feats_L = torch.cat(all_image_feats_L, dim=0)
        all_gene_feats_L = torch.cat(all_gene_feats_L, dim=0)


        with torch.autocast(self.device.type, dtype=torch.float32):
            image_logits = -Lorentz.pairwise_dist(image_feats_L, all_gene_feats_L, _curv)
            gene_logits = -Lorentz.pairwise_dist(gene_feats_L, all_image_feats_L, _curv)
            niche_image_logits = -Lorentz.pairwise_dist(niche_image_feats_L, all_gene_feats_L, _curv)
            niche_gene_logits = -Lorentz.pairwise_dist(niche_gene_feats_L, all_image_feats_L, _curv)


            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + all_batch_size[self._rank]

            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * gene_logits, targets)
                + nn.functional.cross_entropy(_scale * niche_image_logits, targets)
                + nn.functional.cross_entropy(_scale * niche_gene_logits, targets)
            )

            _angle = Lorentz.oxy_angle(image_feats_L, gene_feats_L, _curv)
            _aperture = Lorentz.half_aperture(image_feats_L, _curv)

            _niche_angle = Lorentz.oxy_angle(niche_image_feats_L, niche_gene_feats_L, _curv)
            _niche_aperture = Lorentz.half_aperture(niche_image_feats_L, _curv)

            _cross_image_angle = Lorentz.oxy_angle(image_feats_L, niche_image_feats_L, _curv)
            _image_aperture = Lorentz.half_aperture(image_feats_L, _curv)

            _cross_gene_angle = Lorentz.oxy_angle(gene_feats_L, niche_gene_feats_L, _curv)
            _gene_aperture = Lorentz.half_aperture(gene_feats_L, _curv)

            _global_aperture_thresh = 0.7   
            _local_aperture_thresh = 1.2   

            image_gene_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            niche_image_gene_entailment_loss = torch.clamp(_niche_angle - _global_aperture_thresh * _niche_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _image_aperture, min=0).mean()
            cross_gene_entailment_loss = torch.clamp(_cross_gene_angle - _local_aperture_thresh * _gene_aperture, min=0).mean()

            entailment_loss = 0.5 * (
                image_gene_entailment_loss 
                + niche_image_gene_entailment_loss 
                + cross_image_entailment_loss 
                + cross_gene_entailment_loss
            )

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        if self.predict_norm:
            return {
                "loss": loss,
                "logging": {
                    "contrastive_loss": contrastive_loss,
                    "image_gene_entailment_loss": image_gene_entailment_loss,
                    "niche_image_gene_entailment_loss": niche_image_gene_entailment_loss,
                    "cross_image_entailment_loss": cross_image_entailment_loss,
                    "cross_gene_entailment_loss": cross_gene_entailment_loss,
                    "entailment_loss": entailment_loss,
                    "logit_scale": _scale,
                    "curv": _curv,
                }, 
                "emb" : { 
                    "image_feats": image_feats_inner,
                    "niche_image_feats": niche_image_feats_inner,
                    "gene_feats": gene_feats_inner,
                    "niche_gene_feats": niche_gene_feats_inner
                }
            }
        else:
            return {
                "loss": loss,
                "logging": {
                    "contrastive_loss": contrastive_loss,
                    "image_gene_entailment_loss": image_gene_entailment_loss,
                    "niche_image_gene_entailment_loss": niche_image_gene_entailment_loss,
                    "cross_image_entailment_loss": cross_image_entailment_loss,
                    "cross_gene_entailment_loss": cross_gene_entailment_loss,
                    "entailment_loss": entailment_loss,
                    "logit_scale": _scale,
                    "curv": _curv,
                }, 
                "emb" : { 
                    "image_feats": image_feats,
                    "niche_image_feats": niche_image_feats,
                    "gene_feats": gene_feats,
                    "niche_gene_feats": niche_gene_feats
                }
            }