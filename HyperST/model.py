import torch
from torch import nn
from torch.nn import functional as F
from peft import get_peft_model, LoraConfig
from .modules.alignment import HHAlignment
from .modules.encoder import ResMLPEncoder
def encoder_lora(image_encoder_name:str, image_encoder, last_layer:int=3, is_lora_ffn:bool=False, lora_rank:int=8, lora_alpha:int=16, lora_dropout:float=0.1, logger=None):

    if last_layer > 0:

        if image_encoder_name == 'uni':
            length = range(24)
            lora_list = [f'blocks.{i}.attn.qkv'  for i in length[-last_layer:]]
            if is_lora_ffn:
                mlp_fc1_list = [f'blocks.{i}.mlp.fc1'  for i in length[-last_layer:]]
                mlp_fc2_list = [f'blocks.{i}.mlp.fc2'  for i in length[-last_layer:]]
            else:
                mlp_fc1_list = []
                mlp_fc2_list = []
        else:
            raise ValueError(f'image_encoder_name {image_encoder_name} is not support')
        lora_list = lora_list + mlp_fc1_list + mlp_fc2_list

        lora_config = LoraConfig(
            r=lora_rank,  
            lora_alpha=lora_alpha,  
            target_modules=lora_list,  
            lora_dropout=lora_dropout,  
            bias="none", 
        )

        peft_image_encoder = get_peft_model(image_encoder, lora_config)

        return peft_image_encoder
    else:
        for param in image_encoder.parameters():
            param.requires_grad = False
        return image_encoder

    
class HyperST(nn.Module):
    def __init__(
        self,
        image_dim:int,
        gene_dim:int,
        emb_dim:int,
        num_outputs:int,
        mlp_ratio:float,
        image_dropout:float=0.1,
        gene_dropout:float=0,
        decoder_dropout:float=0.,

        entail_weight: float = 0.4,
        niche_project: bool = True,
        predict_norm: bool = False,
        alignment_beta: float = 0.2,


        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        image_encoder = None,
        logger=None,
        image_encoder_name:str = 'uni',
        last_layer:int = 3,
        is_lora_ffn:bool = False
        ):
        super().__init__()

        assert image_encoder_name in ['uni']
        self.alignment_beta = alignment_beta
        self.predict_norm = predict_norm


        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.last_layer = last_layer
        self.is_lora_ffn = is_lora_ffn

        self.image_encoder_name = image_encoder_name
        
        self.image_encoder = encoder_lora(
            image_encoder_name=image_encoder_name, image_encoder=image_encoder, 
            last_layer=last_layer, is_lora_ffn=is_lora_ffn, lora_rank=lora_rank, 
            lora_alpha=lora_alpha, lora_dropout=lora_dropout, logger=logger
        )

      
        self.image_projector = nn.Linear(
            in_features=image_dim,
            out_features=emb_dim
        )

        self.niche_image_projector = nn.Linear(
            in_features=image_dim,
            out_features=emb_dim
        )


        self.gene_encoder = nn.Linear(
            in_features=gene_dim,
            out_features=emb_dim
        )

        self.niche_gene_encoder = nn.Linear(
            in_features=gene_dim,
            out_features=emb_dim
        )
        self.alignment = HHAlignment(
            image_dim=emb_dim,
            gene_dim=emb_dim,
            embed_dim=emb_dim,
            mlp_ratio=mlp_ratio,
            image_dropout=image_dropout,
            gene_dropout=gene_dropout,
            entail_weight=entail_weight,
            niche_project=niche_project,
            predict_norm=predict_norm
        )

        self.gene_decoder = ResMLPEncoder(
            in_features=emb_dim * 2,
            hidden_size=emb_dim,
            mlp_ratio=mlp_ratio,
            drop=decoder_dropout,
            depth=2,
        )


        self.fc = nn.Linear(emb_dim, num_outputs)


    def forward(
            self,
            spot_img: torch.Tensor, niche_image: torch.Tensor,
            spot_gene_ebd: torch.Tensor, niche_gene_ebd: torch.Tensor,
            label:torch.Tensor
    ):

        image_emb = self.image_encoder(spot_img)
        niche_image_emb = self.image_encoder(niche_image)
        

        image_emb = self.image_projector(image_emb)

        niche_image_emb = self.niche_image_projector(niche_image_emb)

        if self.alignment_beta > 0:

            gene_emb = self.gene_encoder(spot_gene_ebd)

            niche_gene_emb = self.niche_gene_encoder(niche_gene_ebd)

            align_result = self.alignment(
                image_emb=image_emb,
                niche_image_emb=niche_image_emb,
                gene_emb=gene_emb,
                niche_gene_emb=niche_gene_emb
            )
            align_loss = align_result['loss']

            image_feats = align_result['emb']['image_feats']
            niche_image_feats = align_result['emb']['niche_image_feats']
            image_predict_emb = torch.cat([image_feats, niche_image_feats], dim=-1)

        else:
            align_result = None
            image_predict_emb = torch.cat([image_emb, niche_image_emb] , dim=-1)
          


        gene_pred = self.fc(self.gene_decoder(image_predict_emb))

        predict_loss = F.mse_loss(gene_pred, label)

        loss = predict_loss
        if self.alignment_beta > 0:
            loss = loss + self.alignment_beta * align_loss

        return {
            'loss': loss, 
            'logits': gene_pred, 
            'logging': {
                'predict_loss': predict_loss,
                'alignment_beta': self.alignment_beta,
                'align' : align_result
            }
        }