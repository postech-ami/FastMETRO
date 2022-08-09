# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see https://github.com/facebookresearch/detr/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Transformer encoder-decoder architecture in FastMETRO model.
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor

class Transformer(nn.Module):
    """Transformer encoder-decoder"""
    def __init__(self, model_dim=512, nhead=8, num_enc_layers=3, num_dec_layers=3, 
                feedforward_dim=2048, dropout=0.1, activation="relu"):
        """
        Parameters:
            - model_dim: The hidden dimension size in the transformer architecture
            - nhead: The number of attention heads in the attention modules
            - num_enc_layers: The number of encoder layers in the transformer encoder
            - num_dec_layers: The number of decoder layers in the transformer decoder
            - feedforward_dim: The hidden dimension size in MLP
            - dropout: The dropout rate in the transformer architecture
            - activation: The activation function used in MLP
        """
        super().__init__()
        self.model_dim = model_dim
        self.nhead = nhead

        # transformer encoder
        encoder_layer = TransformerEncoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        # transformer decoder
        decoder_layer = TransformerDecoderLayer(model_dim, nhead, feedforward_dim, dropout, activation)
        decoder_norm = nn.LayerNorm(model_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers, decoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_features, cam_token, jv_tokens, pos_embed, attention_mask=None):
        device = img_features.device
        hw, bs, _ = img_features.shape
        mask = torch.zeros((bs, hw), dtype=torch.bool, device=device)

        # Transformer Encoder 
        zero_mask = torch.zeros((bs, 1), dtype=torch.bool, device=device)
        mem_mask = torch.cat([zero_mask, mask], dim=1)
        cam_with_img = torch.cat([cam_token, img_features], dim=0)
        e_outputs = self.encoder(cam_with_img, src_key_padding_mask=mem_mask, pos=pos_embed)
        cam_features, enc_img_features = e_outputs.split([1, hw], dim=0) 

        # Transformer Decoder
        zero_tgt = torch.zeros_like(jv_tokens)
        jv_features = self.decoder(jv_tokens, enc_img_features, tgt_mask=attention_mask, 
                                   memory_key_padding_mask=mask, pos=pos_embed, query_pos=zero_tgt)

        return cam_features, enc_img_features, jv_features


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.norm = norm
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # tensor[0] is for a camera token (no positional encoding)
        return tensor if pos is None else torch.cat([tensor[:1], (tensor[1:] + pos)], dim=0)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, model_dim, nhead, feedforward_dim=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, nhead, dropout=dropout)

        # MLP
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        # Layer Normalization & Dropout
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_transformer(transformer_config):
    return Transformer(model_dim=transformer_config['model_dim'],
                       dropout=transformer_config['dropout'],
                       nhead=transformer_config['nhead'],
                       feedforward_dim=transformer_config['feedforward_dim'],
                       num_enc_layers=transformer_config['num_enc_layers'],
                       num_dec_layers=transformer_config['num_dec_layers'])