import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
import math

#ENCODER---------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, stride, in_chans, embed_dim):
        super(OverlapPatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.BN= nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x.type(torch.cuda.FloatTensor))
        _, _, H, W = x.shape
        x = self.BN(x)

        return x


# AxialAttention module refer to: https://github.com/csrhddlam/axial-deeplab
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups, kernel_size,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1  # (14,14)
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # (N, W, C, H)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)  # (128,14,14)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,kij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class Encoder_Block(nn.Module):

    def __init__(self, dim, heads, kernel_size):
        super(Encoder_Block, self).__init__()
        self.height_attn = AxialAttention(in_planes=dim, out_planes=dim, groups=heads,
                                          kernel_size=kernel_size, width=False)
        self.width_attn = AxialAttention(in_planes=dim, out_planes=dim, groups=heads,
                                         kernel_size=kernel_size, width=True)

        self.BN= nn.BatchNorm2d(dim)
        self.conv1x1_up = nn.Conv2d(dim, dim*4, kernel_size=1, stride=1, bias=False)
        self.conv1x1_down = nn.Conv2d(dim*4, dim, kernel_size=1, stride=1, bias=False)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        h = x
        x = self.BN(x)
        x = self.height_attn(x)
        x = self.width_attn(x)
        x = x + h

        h = x
        x = self.BN(x)
        x = self.conv1x1_up(x)
        x = self.act(x)
        x = self.conv1x1_down(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, input_size=1600, in_chans=1, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], depths=[2, 2, 2, 2]):
        super(Encoder,self).__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=input_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=input_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=input_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=input_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # encoder
        self.block1=[]
        for _ in range(depths[0]):
            self.block1.append(Encoder_Block(dim=embed_dims[0], heads=num_heads[0], kernel_size=input_size//4))
        self.block1=nn.Sequential(*self.block1)
        self.norm1 = nn.BatchNorm2d(embed_dims[0])

        self.block2=[]
        for _ in range(depths[1]):
            self.block2.append(Encoder_Block(dim=embed_dims[1], heads=num_heads[1], kernel_size=input_size//8))
        self.block2=nn.Sequential(*self.block2)
        self.norm2 = nn.BatchNorm2d(embed_dims[1])

        self.block3=[]
        for _ in range(depths[2]):
            self.block3.append(Encoder_Block(dim=embed_dims[2], heads=num_heads[2], kernel_size=input_size//16))
        self.block3=nn.Sequential(*self.block3)
        self.norm3 = nn.BatchNorm2d(embed_dims[2])

        self.block4=[]
        for _ in range(depths[3]):
            self.block4.append(Encoder_Block(dim=embed_dims[3], heads=num_heads[3], kernel_size=input_size//32))
        self.block4=nn.Sequential(*self.block4)
        self.norm4 = nn.BatchNorm2d(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        # stage 1
        x1=self.patch_embed1(x)
        x1=self.block1(x1)
        x1 = self.norm1(x1)

        # stage 2
        x2=self.patch_embed2(x1)
        x2=self.block2(x2)
        x2 = self.norm2(x2)

        # stage 3
        x3=self.patch_embed3(x2)
        x3=self.block3(x3)
        x3 = self.norm3(x3)

        # stage 4
        x4=self.patch_embed4(x3)
        x4=self.block4(x4)
        x4 = self.norm4(x4)

        return x1, x2, x3, x4

#DECODER---------------------------------------------------
class FNN(nn.Module):
    def __init__(self, sum_encoder_embed_dims=960,rec_channels=1):
        super(FNN, self).__init__()
        self.rec_channels = rec_channels

        self.dropout = nn.Dropout(0.3)
        self.act_Lrelu=nn.LeakyReLU(negative_slope=0.2)
        self.act_relu = nn.ReLU()
        self.act_tanh=nn.Tanh()

        self.f = nn.Sequential(
            nn.Conv2d(sum_encoder_embed_dims, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            #self.act_Lrelu,
            self.act_relu,
            #self.dropout,

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            #self.act_Lrelu,
            self.act_relu,

            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #self.act_Lrelu,
            self.act_relu,

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            #self.act_Lrelu,
            self.act_relu,

            nn.Conv2d(64, self.rec_channels, kernel_size=3, padding=1, bias=False),
            #self.act_tanh
        )

    def forward(self, x):
        x=self.f(x)
        return x

class Decoder(nn.Module):
    def __init__(self, sum_encoder_embed_dims=960,rec_channels=1, rec_size=500):
        super(Decoder, self).__init__()

        self.rec_size = rec_size

        self.FNN1007=FNN(sum_encoder_embed_dims=sum_encoder_embed_dims,rec_channels=rec_channels)
    def forward(self, x):
        s1, s2, s3, s4 = x

        s1=F.interpolate(s1, size=(self.rec_size, self.rec_size), mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, size=(self.rec_size, self.rec_size), mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=(self.rec_size, self.rec_size), mode='bilinear', align_corners=False)
        s4 = F.interpolate(s4, size=(self.rec_size, self.rec_size), mode='bilinear', align_corners=False)
        s = torch.cat((s1, s2, s3, s4),1).type(torch.cuda.FloatTensor)
        #print(s.size())
        s = self.FNN1007(s)
        return s

