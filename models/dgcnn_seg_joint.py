""" Define the segmentation model structure for joint training """

import torch
import torch.nn as nn
from models.dgcnn import DGCNN

# Joint Training Using DGCNN segmentation model
class DGCNNSeg(nn.Module):
    def __init__(self, args, num_classes):
        super(DGCNNSeg, self).__init__()
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        in_dim = args.dgcnn_mlp_widths[-1]
        for edgeconv_width in args.edgeconv_widths:
            in_dim += edgeconv_width[-1]
        self.segmentor = nn.Sequential(
                            nn.Conv1d(in_dim, 256, 1, bias=False),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(0.2),
                            nn.Conv1d(256, 128, 1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Conv1d(128, num_classes, 1)
                         )

    def forward(self, pc):
        num_points = pc.shape[2]
        edgeconv_feats, point_feat = self.encoder(pc)
        global_feat = point_feat.max(dim=-1, keepdim=True)[0]
        edgeconv_feats.append(global_feat.expand(-1,-1,num_points))
        pc_feat = torch.cat(edgeconv_feats, dim=1)
        logits = self.segmentor(pc_feat)
        return logits
