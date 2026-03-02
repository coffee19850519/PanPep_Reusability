"""Learner module with complete support for standard layers and ResNet-style residual blocks.

This is the final version that combines:
- Original learner.py: Complete self_attention with mask, bias, multi-head attention
- learner_resnet.py: res_block and res_attention_block implementations
"""

import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class Learner(nn.Module):
    """
    Configurable neural network supporting both standard layers and ResNet-style residual blocks.

    Supported layer types:
    - 'conv2d': [out_ch, in_ch, kernel, kernel, stride, padding]
    - 'linear': [out_features, in_features]
    - 'bn': [num_features]
    - 'relu', 'tanh', 'sigmoid', 'leakyrelu', 'gelu'
    - 'max_pool2d', 'avg_pool2d': [kernel, stride, padding]
    - 'flatten': []
    - 'self_attention': [[in_ch, key_dim, value_dim], ...] (with mask support)
    - 'res_block': [[out_ch, in_ch, kernel, stride, padding, dilation], ...]
    - 'res_attention_block': [[in_ch, key_dim, value_dim], ...]
    - 'res_multi_head_attention_block': [[in_ch, key_dim, value_dim, num_heads], ...]
    - 'attention_stack': [[in_ch, key_dim, value_dim], ...] (multi-layer attention with bn+relu)
    - 'conv_stack': [[out_ch, in_ch, kernel, stride, padding], ...] (multi-layer conv with bn+relu)
    - 'transformer_stack': [[d_model, n_heads, d_ff], ...] (multi-layer transformer with residual)
    """

    def __init__(self, config):
        super(Learner, self).__init__()

        self.config = config
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):

            # Conv2d layer
            if name is 'conv2d' or name == 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # Linear layer
            elif name is 'linear' or name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # Batch normalization
            elif name is 'bn' or name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            # Self attention layer (original implementation with bias and mask)
            elif name is 'self_attention' or name == 'self_attention':
                Q = nn.Parameter(torch.ones(param[0]))
                K = nn.Parameter(torch.ones(param[1]))
                V = nn.Parameter(torch.ones(param[2]))
                torch.nn.init.kaiming_normal_(Q)
                torch.nn.init.kaiming_normal_(K)
                torch.nn.init.kaiming_normal_(V)
                self.vars.append(Q)
                self.vars.append(nn.Parameter(torch.zeros(param[0][:2])))
                self.vars.append(K)
                self.vars.append(nn.Parameter(torch.zeros(param[1][:2])))
                self.vars.append(V)
                self.vars.append(nn.Parameter(torch.zeros(param[2][:2])))
            # ResNet-style residual block for convolution (conv_stack + residual)
            elif name == 'res_block':
                num_layers = len(param)
                if num_layers == 0:
                    raise ValueError("res_block requires at least one layer")

                first_in_ch = param[0][1]
                last_out_ch = param[-1][0]

                # Calculate total stride (product of all strides)
                total_stride = 1
                for conv_param in param:
                    stride = conv_param[3]
                    total_stride *= stride

                # Projection layer for skip connection if needed
                # (channel mismatch OR stride > 1 changes spatial dimensions)
                if first_in_ch != last_out_ch or total_stride > 1:
                    # 1x1 conv for channel matching (will use stride in forward)
                    w = nn.Parameter(torch.ones(last_out_ch, first_in_ch, 1, 1))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(last_out_ch)))
                    # BN for projection
                    self.vars.append(nn.Parameter(torch.ones(last_out_ch)))
                    self.vars.append(nn.Parameter(torch.zeros(last_out_ch)))
                    self.vars_bn.extend([
                        nn.Parameter(torch.zeros(last_out_ch), requires_grad=False),
                        nn.Parameter(torch.ones(last_out_ch), requires_grad=False)
                    ])

                # Same as conv_stack: Conv + BN for each layer
                for layer_idx, conv_param in enumerate(param):
                    # Conv2d layer
                    out_ch, in_ch, kernel = conv_param[0], conv_param[1], conv_param[2]
                    w = nn.Parameter(torch.ones(out_ch, in_ch, kernel, kernel))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(out_ch)))

                    # Batch normalization after conv
                    self.vars.append(nn.Parameter(torch.ones(out_ch)))
                    self.vars.append(nn.Parameter(torch.zeros(out_ch)))
                    running_mean = nn.Parameter(torch.zeros(out_ch), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(out_ch), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])
                    # ReLU will be applied in forward (except for last layer)

            # ResNet-style residual block for attention (attention_stack + residual)
            elif name == 'res_attention_block':
                num_layers = len(param)
                if num_layers == 0:
                    raise ValueError("res_attention_block requires at least one layer")

                # att_param = [num_heads, proj_dim, input_dim]
                # actual input dim = param[0][2], actual output dim = param[-1][1]
                actual_in_dim = param[0][2]
                actual_out_dim = param[-1][1]
                if actual_out_dim != actual_in_dim:
                    proj_w = nn.Parameter(torch.ones(actual_out_dim, actual_in_dim))
                    torch.nn.init.xavier_uniform_(proj_w)
                    self.vars.append(proj_w)
                    self.vars.append(nn.Parameter(torch.zeros(actual_out_dim)))

                # Same as attention_stack: Q, K, V for each layer + BN
                for layer_idx, att_param in enumerate(param):
                    # Self-attention layer (Q, K, V with biases) - same as attention_stack
                    Q = nn.Parameter(torch.ones(att_param))
                    K = nn.Parameter(torch.ones(att_param))
                    V = nn.Parameter(torch.ones(att_param))
                    torch.nn.init.kaiming_normal_(Q)
                    torch.nn.init.kaiming_normal_(K)
                    torch.nn.init.kaiming_normal_(V)
                    self.vars.append(Q)
                    self.vars.append(nn.Parameter(torch.zeros(att_param[:2])))  # Qb
                    self.vars.append(K)
                    self.vars.append(nn.Parameter(torch.zeros(att_param[:2])))  # Kb
                    self.vars.append(V)
                    self.vars.append(nn.Parameter(torch.zeros(att_param[:2])))  # Vb

                    # Batch normalization after attention
                    # attention output dim = att_param[1] (proj_dim after matmul and mean)
                    bn_dim = att_param[1]
                    self.vars.append(nn.Parameter(torch.ones(bn_dim)))
                    self.vars.append(nn.Parameter(torch.zeros(bn_dim)))
                    running_mean = nn.Parameter(torch.zeros(bn_dim), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(bn_dim), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])

            elif name == 'res_multi_head_attention_block':
                num_layers = len(param)
                if num_layers == 0:
                    raise ValueError("res_multi_head_attention_block requires at least one layer")

                first_in_ch = param[0][0]
                last_value_dim = param[-1][2]

                if last_value_dim != first_in_ch:
                    proj_w = nn.Parameter(torch.ones(last_value_dim, first_in_ch))
                    torch.nn.init.xavier_uniform_(proj_w)
                    self.vars.append(proj_w)
                    self.vars.append(nn.Parameter(torch.zeros(last_value_dim)))
                    self.vars.append(nn.Parameter(torch.ones(last_value_dim)))
                    self.vars.append(nn.Parameter(torch.zeros(last_value_dim)))

                for layer_idx, att_param in enumerate(param):
                    if len(att_param) != 4:
                        raise ValueError("res_multi_head_attention_block expects [in_ch, key_dim, value_dim, num_heads]")
                    in_ch, key_dim, value_dim, num_heads = att_param
                    if key_dim % num_heads != 0:
                        raise ValueError(f"key_dim {key_dim} must be divisible by num_heads {num_heads}")
                    if value_dim % num_heads != 0:
                        raise ValueError(f"value_dim {value_dim} must be divisible by num_heads {num_heads}")

                    # Q projection
                    w = nn.Parameter(torch.ones(key_dim, in_ch))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(key_dim)))
                    # K projection
                    w = nn.Parameter(torch.ones(key_dim, in_ch))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(key_dim)))
                    # V projection
                    w = nn.Parameter(torch.ones(value_dim, in_ch))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(value_dim)))
                    # Output projection
                    w = nn.Parameter(torch.ones(value_dim, value_dim))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(value_dim)))
                    # LayerNorm
                    self.vars.append(nn.Parameter(torch.ones(value_dim)))
                    self.vars.append(nn.Parameter(torch.zeros(value_dim)))

            # Attention stack: multi-layer self-attention with bn+relu between layers
            elif name == 'attention_stack':
                num_layers = len(param)
                if num_layers == 0:
                    raise ValueError("attention_stack requires at least one layer")

                for layer_idx, att_param in enumerate(param):
                    # Self-attention layer (Q, K, V with biases)
                    Q = nn.Parameter(torch.ones(att_param))
                    K = nn.Parameter(torch.ones(att_param))
                    V = nn.Parameter(torch.ones(att_param))
                    torch.nn.init.kaiming_normal_(Q)
                    torch.nn.init.kaiming_normal_(K)
                    torch.nn.init.kaiming_normal_(V)
                    self.vars.append(Q)
                    self.vars.append(nn.Parameter(torch.zeros(att_param[:2])))  # Qb
                    self.vars.append(K)
                    self.vars.append(nn.Parameter(torch.zeros(att_param[:2])))  # Kb
                    self.vars.append(V)
                    self.vars.append(nn.Parameter(torch.zeros(att_param[:2])))  # Vb

                    # Batch normalization after attention (for output dimension)
                    # Note: attention outputs [batch, seq_len, value_dim], BN on value_dim
                    out_dim = att_param[2]  # value_dim is the output dimension
                    self.vars.append(nn.Parameter(torch.ones(out_dim)))
                    self.vars.append(nn.Parameter(torch.zeros(out_dim)))
                    running_mean = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(out_dim), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])
                    # ReLU will be applied in forward (except for last layer)

            # Conv stack: multi-layer conv2d with bn+relu between layers
            elif name == 'conv_stack':
                num_layers = len(param)
                if num_layers == 0:
                    raise ValueError("conv_stack requires at least one layer")

                for layer_idx, conv_param in enumerate(param):
                    # Conv2d layer
                    out_ch, in_ch, kernel = conv_param[0], conv_param[1], conv_param[2]
                    w = nn.Parameter(torch.ones(out_ch, in_ch, kernel, kernel))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(out_ch)))

                    # Batch normalization after conv
                    self.vars.append(nn.Parameter(torch.ones(out_ch)))
                    self.vars.append(nn.Parameter(torch.zeros(out_ch)))
                    running_mean = nn.Parameter(torch.zeros(out_ch), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(out_ch), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])
                    # ReLU will be applied in forward (except for last layer)

            # Transformer stack: multi-layer transformer with residual connections
            elif name == 'transformer_stack':
                num_layers = len(param)
                if num_layers == 0:
                    raise ValueError("transformer_stack requires at least one layer")

                for layer_idx, tf_param in enumerate(param):
                    d_model, n_heads, d_ff = tf_param
                    head_dim = d_model // n_heads

                    if d_model % n_heads != 0:
                        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

                    # Multi-head Self-Attention
                    # Q, K, V projections
                    for _ in range(3):  # Q, K, V
                        w = nn.Parameter(torch.ones(d_model, d_model))
                        torch.nn.init.xavier_uniform_(w)
                        self.vars.append(w)
                        self.vars.append(nn.Parameter(torch.zeros(d_model)))

                    # Output projection
                    w = nn.Parameter(torch.ones(d_model, d_model))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(d_model)))

                    # LayerNorm 1 (after attention)
                    self.vars.append(nn.Parameter(torch.ones(d_model)))
                    self.vars.append(nn.Parameter(torch.zeros(d_model)))

                    # Feed-Forward Network
                    # FFN Layer 1: d_model -> d_ff
                    w = nn.Parameter(torch.ones(d_ff, d_model))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(d_ff)))

                    # FFN Layer 2: d_ff -> d_model
                    w = nn.Parameter(torch.ones(d_model, d_ff))
                    torch.nn.init.xavier_uniform_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(d_model)))

                    # LayerNorm 2 (after FFN)
                    self.vars.append(nn.Parameter(torch.ones(d_model)))
                    self.vars.append(nn.Parameter(torch.zeros(d_model)))

            # No-parameter layers
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'gelu']:
                continue
            else:
                raise NotImplementedError(f"Layer type '{name}' not implemented")

    def forward(self, x, vars=None, bn_training=True, return_embedding=False):
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch, seq_len, features]
            vars: Optional parameter list (for meta-learning)
            bn_training: Whether to use batch statistics for BN
            return_embedding: Return features before final layer

        Returns:
            Output tensor
        """
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        # Compute mask for padding positions
        mask = torch.abs(x).sum(dim=-1) == 0

        # Check if we should return embedding before last layer
        for i, (name, param) in enumerate(self.config):
            if return_embedding and i == len(self.config) - 1 and (name == 'linear' or name is 'linear'):
                return x

        # Main forward pass
        for i, (name, param) in enumerate(self.config):

            # Conv2d
            if name is 'conv2d' or name == 'conv2d':
                if len(x.size()) < 4:
                    x = x.unsqueeze(1)
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2

            # Linear
            elif name is 'linear' or name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            # Batch Normalization
            elif name is 'bn' or name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            # Self Attention (original implementation with mask)
            elif name is 'self_attention' or name == 'self_attention':
                Q, Qb = vars[idx], vars[idx + 1]
                K, Kb = vars[idx + 2], vars[idx + 3]
                V, Vb = vars[idx + 4], vars[idx + 5]
                idx += 6
                q_value = torch.matmul(x.unsqueeze(1), Q.transpose(1, 2)) + Qb.unsqueeze(1)
                k_value = torch.matmul(x.unsqueeze(1), K.transpose(1, 2)) + Kb.unsqueeze(1)
                v_value = torch.matmul(x.unsqueeze(1), V.transpose(1, 2)) + Vb.unsqueeze(1)
                score = torch.matmul(q_value, k_value.transpose(-2, -1))
                with torch.no_grad():
                    mask2 = mask.repeat(1, x.size()[1]).view(x.size()[0], x.size()[1], -1)
                    mask2 = mask2 + mask2.transpose(-2, -1)
                    score[mask2.unsqueeze(1).repeat(1, score.size()[1], 1, 1)] = -1e9
                att = F.softmax(score, dim=-1)
                x = torch.matmul(att, v_value)
                x[mask.unsqueeze(1).repeat(1, score.size()[1], 1)] = 0
                x = torch.mean(x, dim=1)

            # ResNet-style Residual Block (conv)
            elif name == 'res_block':
                # ResNet-style: conv_stack + residual connection
                num_layers = len(param)
                first_in_ch = param[0][1]
                last_out_ch = param[-1][0]

                # Calculate total stride (product of all strides)
                total_stride = 1
                for conv_param in param:
                    stride = conv_param[3]
                    total_stride *= stride

                # Add channel dimension if needed (before saving identity)
                if len(x.size()) < 4:
                    x = x.unsqueeze(1)

                # Save identity for residual connection
                identity = x

                # Projection for skip connection if needed
                # (channel mismatch OR stride > 1 changes spatial dimensions)
                if first_in_ch != last_out_ch or total_stride > 1:
                    # Check if actual input channels match expected
                    actual_in_ch = identity.size(1)
                    if actual_in_ch != first_in_ch:
                        raise RuntimeError(
                            f"res_block: config expects {first_in_ch} input channels, "
                            f"but got {actual_in_ch} channels. "
                            f"Please update config first layer to [{last_out_ch}, {actual_in_ch}, ...]"
                        )
                    proj_w, proj_b = vars[idx], vars[idx + 1]
                    # Use total_stride to match spatial dimensions
                    identity = F.conv2d(identity, proj_w, proj_b, stride=total_stride)
                    idx += 2
                    bn_w, bn_b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    identity = F.batch_norm(identity, running_mean, running_var,
                                           weight=bn_w, bias=bn_b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                # Apply conv_stack logic (same as conv_stack)
                for layer_idx, conv_param in enumerate(param):

                    # Conv2d
                    out_ch, in_ch, kernel_size, stride, padding = conv_param
                    conv_w, conv_b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, conv_w, conv_b, stride=stride, padding=padding)
                    idx += 2

                    # Batch normalization
                    bn_w, bn_b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var,
                                     weight=bn_w, bias=bn_b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                    # ReLU (except for last layer before residual)
                    if layer_idx < num_layers - 1:
                        x = F.relu(x)  # 不用inplace，避免破坏梯度

                # Add residual connection
                x = x + identity
                # Final ReLU after residual
                x = F.relu(x)  # 也不能用inplace

            # ResNet-style Residual Block (attention)
            elif name == 'res_attention_block':
                # ResNet-style: attention_stack + residual connection
                num_layers = len(param)
                # att_param = [num_heads, proj_dim, input_dim]
                actual_in_dim = param[0][2]
                actual_out_dim = param[-1][1]

                # Save identity for residual connection
                identity = x

                # Project identity if input/output dims differ
                if actual_out_dim != actual_in_dim:
                    proj_w, proj_b = vars[idx], vars[idx + 1]
                    identity = F.linear(identity, proj_w, proj_b)
                    idx += 2

                # Apply attention_stack logic
                for layer_idx, att_param in enumerate(param):
                    # Self-attention layer
                    Q, Qb = vars[idx], vars[idx + 1]
                    K, Kb = vars[idx + 2], vars[idx + 3]
                    V, Vb = vars[idx + 4], vars[idx + 5]
                    idx += 6

                    q_value = torch.matmul(x.unsqueeze(1), Q.transpose(1, 2)) + Qb.unsqueeze(1)
                    k_value = torch.matmul(x.unsqueeze(1), K.transpose(1, 2)) + Kb.unsqueeze(1)
                    v_value = torch.matmul(x.unsqueeze(1), V.transpose(1, 2)) + Vb.unsqueeze(1)

                    score = torch.matmul(q_value, k_value.transpose(-2, -1))

                    # Apply mask to prevent attention to padding
                    with torch.no_grad():
                        mask2 = mask.repeat(1, x.size()[1]).view(x.size()[0], x.size()[1], -1)
                        mask2 = mask2 + mask2.transpose(-2, -1)
                        score[mask2.unsqueeze(1).repeat(1, score.size()[1], 1, 1)] = -1e9

                    att = F.softmax(score, dim=-1)
                    x = torch.matmul(att, v_value)
                    x[mask.unsqueeze(1).repeat(1, score.size()[1], 1)] = 0
                    x = torch.mean(x, dim=1)

                    # Batch normalization
                    # x shape: [batch, 40, 5] -> need to transpose for BN
                    bn_w, bn_b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = x.transpose(1, 2)  # [batch, 40, 5] -> [batch, 5, 40]
                    x = F.batch_norm(x, running_mean, running_var,
                                     weight=bn_w, bias=bn_b, training=bn_training)
                    x = x.transpose(1, 2)  # [batch, 5, 40] -> [batch, 40, 5]
                    idx += 2
                    bn_idx += 2

                    # ReLU (except for last layer before residual)
                    if layer_idx < num_layers - 1:
                        x = F.relu(x)

                # Add residual connection
                x = x + identity
                # Final ReLU after residual
                x = F.relu(x)

            elif name == 'res_multi_head_attention_block':
                num_layers = len(param)
                first_in_ch = param[0][0]
                last_value_dim = param[-1][2]

                if x.dim() != 3:
                    batch_size = x.size(0)
                    x = x.view(batch_size, -1, first_in_ch)
                batch_size, seq_len, current_dim = x.size()
                if current_dim != first_in_ch:
                    x = x.view(batch_size, -1, first_in_ch)
                    seq_len = x.size(1)

                identity = x
                if last_value_dim != first_in_ch:
                    proj_w, proj_b = vars[idx], vars[idx + 1]
                    identity = F.linear(identity, proj_w, proj_b)
                    idx += 2
                    ln_w, ln_b = vars[idx], vars[idx + 1]
                    identity = F.layer_norm(identity, [last_value_dim], weight=ln_w, bias=ln_b)
                    idx += 2

                valid_mask = None
                if mask is not None and mask.dim() == 2 and mask.size(1) == seq_len:
                    valid_mask = mask

                for layer_idx, att_param in enumerate(param):
                    _, key_dim, value_dim, num_heads = att_param
                    head_dim_q = key_dim // num_heads
                    head_dim_v = value_dim // num_heads

                    q_w, q_b = vars[idx], vars[idx + 1]
                    k_w, k_b = vars[idx + 2], vars[idx + 3]
                    v_w, v_b = vars[idx + 4], vars[idx + 5]
                    idx += 6

                    q = F.linear(x, q_w, q_b)
                    k = F.linear(x, k_w, k_b)
                    v = F.linear(x, v_w, v_b)

                    q = q.view(batch_size, seq_len, num_heads, head_dim_q).transpose(1, 2)
                    k = k.view(batch_size, seq_len, num_heads, head_dim_q).transpose(1, 2)
                    v = v.view(batch_size, seq_len, num_heads, head_dim_v).transpose(1, 2)

                    att_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim_q)
                    if valid_mask is not None:
                        mask_k = valid_mask.unsqueeze(1).unsqueeze(2)
                        att_score = att_score.masked_fill(mask_k, float('-inf'))

                    att_weight = F.softmax(att_score, dim=-1)
                    att_out = torch.matmul(att_weight, v)
                    att_out = att_out.transpose(1, 2).contiguous().view(batch_size, seq_len, value_dim)

                    out_proj_w, out_proj_b = vars[idx], vars[idx + 1]
                    x = F.linear(att_out, out_proj_w, out_proj_b)
                    idx += 2

                    ln_w, ln_b = vars[idx], vars[idx + 1]
                    x = F.layer_norm(x, [value_dim], weight=ln_w, bias=ln_b)
                    idx += 2

                    if valid_mask is not None:
                        x = x.masked_fill(valid_mask.unsqueeze(-1), 0)

                    if layer_idx < num_layers - 1:
                        x = F.relu(x)

                x = F.relu(x + identity)

            # Attention stack: multi-layer self-attention with bn+relu
            elif name == 'attention_stack':
                num_layers = len(param)

                for layer_idx, att_param in enumerate(param):
                    # Self-attention layer
                    Q, Qb = vars[idx], vars[idx + 1]
                    K, Kb = vars[idx + 2], vars[idx + 3]
                    V, Vb = vars[idx + 4], vars[idx + 5]
                    idx += 6

                    q_value = torch.matmul(x.unsqueeze(1), Q.transpose(1, 2)) + Qb.unsqueeze(1)
                    k_value = torch.matmul(x.unsqueeze(1), K.transpose(1, 2)) + Kb.unsqueeze(1)
                    v_value = torch.matmul(x.unsqueeze(1), V.transpose(1, 2)) + Vb.unsqueeze(1)

                    score = torch.matmul(q_value, k_value.transpose(-2, -1))

                    # Apply mask to prevent attention to padding
                    with torch.no_grad():
                        mask2 = mask.repeat(1, x.size()[1]).view(x.size()[0], x.size()[1], -1)
                        mask2 = mask2 + mask2.transpose(-2, -1)
                        score[mask2.unsqueeze(1).repeat(1, score.size()[1], 1, 1)] = -1e9

                    att = F.softmax(score, dim=-1)
                    x = torch.matmul(att, v_value)
                    x[mask.unsqueeze(1).repeat(1, score.size()[1], 1)] = 0
                    x = torch.mean(x, dim=1)

                    # Batch normalization
                    # x shape: [batch, 40, 5] -> need to transpose for BN
                    bn_w, bn_b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = x.transpose(1, 2)  # [batch, 40, 5] -> [batch, 5, 40]
                    x = F.batch_norm(x, running_mean, running_var,
                                     weight=bn_w, bias=bn_b, training=bn_training)
                    x = x.transpose(1, 2)  # [batch, 5, 40] -> [batch, 40, 5]
                    idx += 2
                    bn_idx += 2

                    # ReLU (except for last layer)
                    if layer_idx < num_layers - 1:
                        x = F.relu(x)

            # Conv stack: multi-layer conv2d with bn+relu
            elif name == 'conv_stack':
                num_layers = len(param)

                for layer_idx, conv_param in enumerate(param):
                    # Add channel dimension if needed (first conv layer)
                    if layer_idx == 0 and len(x.size()) < 4:
                        x = x.unsqueeze(1)

                    # Conv2d
                    out_ch, in_ch, kernel_size, stride, padding = conv_param
                    conv_w, conv_b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, conv_w, conv_b, stride=stride, padding=padding)
                    idx += 2

                    # Batch normalization
                    bn_w, bn_b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var,
                                     weight=bn_w, bias=bn_b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                    # ReLU (except for last layer)
                    if layer_idx < num_layers - 1:
                        x = F.relu(x)

            # Transformer stack: multi-layer transformer with residual connections
            elif name == 'transformer_stack':
                num_layers = len(param)

                for layer_idx, tf_param in enumerate(param):
                    d_model, n_heads, d_ff = tf_param
                    head_dim = d_model // n_heads
                    batch_size = x.size(0)
                    seq_len = x.size(1)

                    # Save input for residual connection
                    residual = x

                    # === Multi-head Self-Attention ===
                    # Q, K, V projections
                    q_w, q_b = vars[idx], vars[idx + 1]
                    k_w, k_b = vars[idx + 2], vars[idx + 3]
                    v_w, v_b = vars[idx + 4], vars[idx + 5]
                    idx += 6

                    q = F.linear(x, q_w, q_b)  # [batch, seq_len, d_model]
                    k = F.linear(x, k_w, k_b)  # [batch, seq_len, d_model]
                    v = F.linear(x, v_w, v_b)  # [batch, seq_len, d_model]

                    # Reshape for multi-head: [batch, seq_len, d_model] -> [batch, n_heads, seq_len, head_dim]
                    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                    k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                    v = v.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)

                    # Scaled dot-product attention
                    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)

                    # Apply mask to prevent attention to padding
                    with torch.no_grad():
                        # Create 2D mask: mask2[i,j] = True if query[i] OR key[j] is padding
                        # mask: [batch, seq_len] -> [batch, seq_len, seq_len]
                        mask_2d = mask.unsqueeze(2) | mask.unsqueeze(1)  # [batch, seq_len, seq_len]
                        # Expand for multi-head: [batch, n_heads, seq_len, seq_len]
                        mask_expanded = mask_2d.unsqueeze(1).expand(batch_size, n_heads, seq_len, seq_len)

                    # Apply mask (must be outside no_grad to preserve gradients)
                    scores = scores.masked_fill(mask_expanded, -1e9)
                    attn_weights = F.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]

                    # Concatenate heads: [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

                    # Output projection
                    out_w, out_b = vars[idx], vars[idx + 1]
                    idx += 2
                    attn_output = F.linear(attn_output, out_w, out_b)

                    # Add & Norm 1
                    x = attn_output + residual
                    ln1_w, ln1_b = vars[idx], vars[idx + 1]
                    idx += 2
                    x = F.layer_norm(x, [d_model], weight=ln1_w, bias=ln1_b)

                    # Save for second residual connection
                    residual = x

                    # === Feed-Forward Network ===
                    # FFN Layer 1
                    ffn1_w, ffn1_b = vars[idx], vars[idx + 1]
                    idx += 2
                    ffn_output = F.linear(x, ffn1_w, ffn1_b)
                    ffn_output = F.gelu(ffn_output)  # GELU activation

                    # FFN Layer 2
                    ffn2_w, ffn2_b = vars[idx], vars[idx + 1]
                    idx += 2
                    ffn_output = F.linear(ffn_output, ffn2_w, ffn2_b)

                    # Add & Norm 2
                    x = ffn_output + residual
                    ln2_w, ln2_b = vars[idx], vars[idx + 1]
                    idx += 2
                    x = F.layer_norm(x, [d_model], weight=ln2_w, bias=ln2_b)

            # Flatten
            elif name is 'flatten' or name == 'flatten':
                x = x.view(x.size(0), -1)
                if return_embedding:
                    return x

            # Reshape
            elif name is 'reshape' or name == 'reshape':
                x = x.view(x.size(0), *param)

            # Activation functions
            elif name is 'relu' or name == 'relu':
                x = F.relu(x, inplace=param[0])

            elif name is 'gelu' or name == 'gelu':
                x = F.gelu(x)

            elif name is 'leakyrelu' or name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name is 'tanh' or name == 'tanh':
                x = F.tanh(x)

            elif name is 'sigmoid' or name == 'sigmoid':
                x = torch.sigmoid(x)

            # Pooling
            elif name is 'max_pool2d' or name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

            elif name is 'avg_pool2d' or name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            elif name is 'upsample' or name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])

            else:
                raise NotImplementedError(f"Layer '{name}' not implemented in forward")

        # Verify all parameters were used
        assert idx == len(vars), f"Parameter mismatch: used {idx}/{len(vars)}"
        assert bn_idx == len(self.vars_bn), f"BN parameter mismatch: used {bn_idx}/{len(self.vars_bn)}"

        return x

    def zero_grad(self, vars=None):
        """Zero gradients for parameters."""
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """Return learnable parameters."""
        return self.vars
