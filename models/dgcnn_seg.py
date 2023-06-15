""" Define the overall class-incremental segmentation model structure and functions
    for geometry-aware relations and uncertainty estimation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dgcnn import DGCNN
from torch_cluster import fps
from utils.pc_utils import nn_distance, index_point
from sklearn.neighbors import NearestNeighbors

# Define the model structure except the final last output layer for point-wise classification
class DGCNNSeg(nn.Module):
    def __init__(self, args, option='no_uncertain'):
        super(DGCNNSeg, self).__init__()
        self.k_relations = 12
        self.k_probs = 12
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        in_dim = args.dgcnn_mlp_widths[-1]
        for edgeconv_width in args.edgeconv_widths:
            in_dim += edgeconv_width[-1]

        self.option = option
        if option == 'uncertain':
            self.segmentor = nn.Sequential(
                nn.Conv1d(in_dim, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
            )
        else:
            self.segmentor = nn.Sequential(
                nn.Conv1d(in_dim, 256, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            )
        # Final output classifier layer conv1d(128, num_classes, 1) is in Class Classifier(nn.Module)

    def forward(self, pc, relation=False):
        num_points = pc.shape[2]
        edgeconv_feats, point_feat = self.encoder(pc)

        global_feat = point_feat.max(dim=-1, keepdim=True)[0]
        edgeconv_feats.append(global_feat.expand(-1,-1,num_points))
        pc_feat = torch.cat(edgeconv_feats, dim=1)

        if relation:
            pc_relations = (self.geometry_aware_structure(pc[:, :3, :], pc_feat[:, :, :])).mean(2)
        else:
            pc_relations = None

        if self.option == 'uncertain':
            logits = self.segmentor(pc_feat)
            logits = F.dropout(logits, p=0.3, training=True)
        else:
            logits = self.segmentor(pc_feat)

        return pc_relations, logits

    # Model the geometry-aware point relations for distillation
    def geometry_aware_structure(self, xyz, feat):
        xyz, feat = xyz.transpose(1, 2), feat.transpose(1, 2)
        fps_point_list = []
        fps_feat_list = []
        for batch in range(xyz.shape[0]):
            fps_index = fps(xyz[batch,:, :], None, ratio=0.25, random_start=False).unique()
            fps_point_list.append(xyz[batch, fps_index, :].unsqueeze(0))
            fps_feat_list.append(feat[batch, fps_index, :].unsqueeze(0))

        fps_point = torch.cat(fps_point_list, dim=0)
        fps_feat = torch.cat(fps_feat_list, dim=0)
        pc_dist = nn_distance(fps_point, xyz, pc_distance=True)
        pc_index = (pc_dist.sort(-1, False)[1])[:, :, :self.k_relations]
        index_points_xyz = index_point(xyz, pc_index)
        index_points_features = index_point(feat, pc_index)
        pc_xyz_rel = index_points_xyz - fps_point.unsqueeze(dim=2)
        pc_feat_rel = index_points_features - fps_feat.unsqueeze(dim=2)
        pc_relations = torch.cat([pc_xyz_rel, pc_feat_rel], dim=-1)
        del pc_dist, pc_index, index_points_features
        return pc_relations

    # Estimate the uncertainties of point predictions
    def uncertainty_estimation(self, xyz, output_prob):
        num_classes = output_prob.shape[-1]

        xyz = xyz.reshape(-1, 3)
        output_prob = output_prob.reshape(-1, num_classes)
        nbrs = NearestNeighbors(n_neighbors=self.k_probs, algorithm='ball_tree', n_jobs=-1).fit(xyz.cpu().numpy())
        _, knn_indices = nbrs.kneighbors(xyz.cpu().numpy())
        index_probs = output_prob[knn_indices.reshape(1, -1)].reshape(-1, self.k_probs, num_classes)
        xyzs = xyz[knn_indices.reshape(1, -1)].reshape(-1, self.k_probs, 3)

        fz = torch.sum(xyzs * (xyzs[:, 0, :].reshape(-1, 1, 3)), 2)
        fm = torch.sqrt(torch.sum((xyzs[:, 0, :].reshape(-1, 1, 3)) ** 2, 2)) * torch.sqrt(
                torch.sum(xyzs ** 2, 2))
        similar = fz / fm
        similar = 0.5 * ((similar - torch.min(similar, 1)[0].reshape(-1, 1)) / (
                    torch.max(similar, 1)[0].reshape(-1, 1) - torch.min(similar, 1)[0].reshape(-1, 1)) + 1)
        similar = similar / torch.sum(similar, 1).reshape(-1, 1)
        probs = index_probs * torch.as_tensor(similar.reshape(-1, self.k_probs, 1))
        output_mean = torch.sum(probs, 1) / self.k_probs
        square_mean = torch.sum(probs ** 2, 1) / self.k_probs
        entropy_mean = torch.sum(-(probs * torch.log(probs)).mean(2), 1) / self.k_probs

        # A Series of methods optional for uncertainty estimation: Mean STD, Mean Entropy, Variation Ratios, BALD (we use)
        # Mean STD
        # uncertain = (square_mean - output_mean.pow(2)).mean(1)

        # Max Entropy
        # uncertain = -(output_mean * torch.log(output_mean)).mean(1)

        # Variation Ratios
        # uncertain = 1 - output_mean.max(1)[0]

        # BALD
        uncertain = -(output_mean * torch.log(output_mean)).mean(1) - entropy_mean
        uncertain_knn = (uncertain.reshape(-1, 1))[knn_indices.reshape(1, -1)].reshape(-1, self.k_probs, 1)
        return index_probs, uncertain, uncertain_knn

# Define the final output layer for different situations
class Classifer(nn.Module):
    def __init__(self, num_classes, initial_old_classifer_weights=None):
        super(Classifer, self).__init__()
        self.num_classes = num_classes
        if initial_old_classifer_weights != None: # Only initialize the novel weights
            num_base_class, _, _ = initial_old_classifer_weights.size()
            classifer_weights_novel = torch.empty(num_classes - num_base_class + 1, 128, 1).cuda()
            torch.nn.init.kaiming_normal_(classifer_weights_novel)
            classifer_weights = torch.cat((initial_old_classifer_weights[1:, :, :], classifer_weights_novel), dim=0)
            self.classifer_weights = nn.Parameter(classifer_weights, requires_grad=True)
        else: # Random initialization
            self.classifer_weights = nn.Parameter(torch.empty(num_classes, 128, 1).cuda(), requires_grad=True)
            torch.nn.init.kaiming_normal_(self.classifer_weights)

    def forward(self, logits, classifer_weights=None):
        if classifer_weights is None:
            classifer_weights = self.classifer_weights
        out_classifer = F.conv1d(logits, classifer_weights, bias=None)
        return out_classifer

    def enroll_weights(self, old_weights):
        classifer_weights = self.classifer_weights.detach()
        classifer_weights = torch.cat((old_weights[1:, :, :], classifer_weights[1:, :, :]), dim=0)
        return classifer_weights.cuda()

