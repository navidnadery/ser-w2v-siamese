from typing import Optional
from torch import nn, FloatTensor, Tensor, cat, std_mean
import torch.nn.functional as F

class model_emo(nn.Module):
    def __init__(self, num_classes=4, Di1=16, Di2=32, Drc=32, Fc1=64, Fc2=64, device='cpu'):
        super(model_emo, self).__init__()
        self.cred = nn.Conv1d(Di2, Drc, 1)
        self.relu = nn.LeakyReLU(0.2)
        # fully connected layers
        self.fc1 = nn.Linear(Di1 * Drc, Fc1)
        self.fc2 = nn.Linear(Fc1, Fc2)
        self.layer_norm = nn.LayerNorm(Fc2, elementwise_affine=False)
        self.W = nn.Parameter(FloatTensor(num_classes, Fc2).uniform_(-0.25, 0.25).to(device), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def Linear(self, input):
      return F.linear(self.layer_norm(input), F.normalize(self.W))

    def forward(self, x):
        x = self.cred(x.transpose(2,1))
        layer1 = self.relu(self.fc1(x.view(x.shape[0], -1)))
        layer2 = self.relu(self.fc2(layer1))
        Ylogits = self.Linear(layer2)
        Ylogits = self.softmax(Ylogits)
        return Ylogits


class MultiheadAttentionKQV(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        if out_dim is not None:
            self.out_proj = nn.Linear(embed_dim, out_dim, bias=True)
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        xk: Tensor,
        xq: Tensor,
        xv: Tensor,
        attention_mask: Optional[Tensor] = None,) -> Tensor:
        
        batch_size, channels, length, embed_dim = xk.size()
        shape = (batch_size, channels*length, self.num_heads, self.head_dim)
        shape_ = (batch_size, self.num_heads, self.head_dim)
        q = self.q_proj(xq).view(*shape_)
        k = self.k_proj(xk).view(*shape).permute(0, 1, 3, 2)  # B, nH, Hd
        v = self.v_proj(xv).view(*shape)
        weights = self.scaling * (q.unsqueeze(1) @ k)  # B, nH
        if attention_mask is not None:
            weights += attention_mask

        weights = nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v  # B, nH, Hd
        output = output.reshape(batch_size, channels, length, embed_dim)
        output = self.out_proj(output)
        return output

# In[17]:
class MultiheadAttention(nn.Module):
    """Multihead Self Attention module
    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probabiliry on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        if out_dim is not None:
            self.out_proj = nn.Linear(embed_dim, out_dim, bias=True)
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or None, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
        Returns:
            Tensor: The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``
        """
        if x.ndim != 4 or x.shape[3] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, _, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, 2, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(3, 2)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 1, 3, 4, 2)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(3, 2)  # B, nH, L, Hd

        weights = self.scaling * (q @ k)  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask

        weights = nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd
        output = output.transpose(3, 2).reshape(batch_size, 2, length, embed_dim)

        output = self.out_proj(output)
        return output


class model_emo_mh(nn.Module):
    def __init__(self, num_classes=4, Di1=16, Di2=32, Drc=32, Fc1=64, Fc2=64, device="cpu"):
        super(model_emo_mh, self).__init__()
        self.mhatt = MultiheadAttention(embed_dim = Di2, num_heads = 8, out_dim = Fc2)
        self.relu = nn.LeakyReLU(0.2)
        # fully connected layers
        self.layer_norm = nn.LayerNorm(4*Fc2, elementwise_affine=False) #TODO check the dim
        self.W = nn.Parameter(FloatTensor(1, 4*Fc2).uniform_(-0.25, 0.25).to(device), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def Linear(self, input):
      return F.linear(self.layer_norm(input), F.normalize(self.W)) #TODO check the dim of normalization
      
    def forward(self, x1, x2):
        att = self.mhatt(cat((x1.unsqueeze(1), x2.unsqueeze(1)), 1))
        std, avg = std_mean(att, unbiased=False, dim=2)
        seqr = cat((avg, std), dim=2)
        Ylogits = self.Linear(seqr.view(seqr.shape[0], -1))
        Ylogits = self.sigmoid(Ylogits)
        return Ylogits