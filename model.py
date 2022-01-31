import torch.nn as nn
from modules import Encoder, Decoder

class Rec_Transformer(nn.Module):
    def __init__(self, input_size=1600, in_chans=1, encoder_embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], depths=[2, 2, 2, 2], sum_encoder_embed_dims=960, rec_channels=1, rec_size=500):
        super(Rec_Transformer, self).__init__()
        self.Encoder = Encoder(input_size=input_size, in_chans=in_chans, embed_dims=encoder_embed_dims, num_heads=num_heads, depths=depths)
        self.Decoder = Decoder(sum_encoder_embed_dims=sum_encoder_embed_dims, rec_channels=rec_channels, rec_size=rec_size)

    def forward(self, x):
        x= self.Encoder(x)
        x= self.Decoder(x)
        return x