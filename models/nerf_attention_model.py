import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFAttentionModel(nn.Module):
    def __init__(self, nerf_model, attention_model, embed_ln):
        super(NeRFAttentionModel, self).__init__()
        self.nerf_model = nerf_model
        self.attention_model = attention_model
        self.embed_ln = embed_ln

    def foward(self, inputs):
        # inputs = [n_pts, embedding_length]
        nerf_inputs = inputs[0]
        # indices: n_views, n_pts, 2 (int32); image_coords: n_views, n_pts, 2 (urounded, float32)
        local = inputs[1]

        embeded_pts = nerf_inputs[..., self.embed_ln]
        embeded_pts = torch.broadcast_to(embeded_pts[None], (local.shape[0], local.shape[1], embeded_pts.shape[-1]))

        attention_output = self.attention_model(torch.transpose(local, [1, 0, 2]),
                                                torch.transpose(embeded_pts, [1, 0, 2]))
        decoder_input = torch.cat([attention_output, nerf_inputs], -1)

        return self.nerf_model(decoder_input), decoder_input
