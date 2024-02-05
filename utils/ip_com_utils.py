# constants related to part segmentation
SMPL_PART_BOUNDS = '../ipman-r/data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_segments_bounds.pkl'
FID_TO_PART = '../ipman-r/data/essentials/yogi_segments/smpl/part_meshes_ply/fid_to_part.pkl'
PART_VID_FID = '../ipman-r/data/essentials/yogi_segments/smpl/part_meshes_ply/smpl_part_vid_fid.pkl'
HD_SMPL_MAP  = '../ipman-r/data/essentials/hd_model/smpl/smpl_neutral_hd_sample_from_mesh_out.pkl'
ESSENTIALS_DIR = '../ipman-r/data/essentials'
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
"""
This script is used to close part meshes.
"""
from tempfile import gettempdir

import torch
import trimesh
import torch.nn as nn
import numpy as np
import pickle
import os
import pickle as pkl
# from .utils.mesh import winding_numbers

class SMPLMesh(nn.Module):
    def __init__(self, vertices, faces):
        super(SMPLMesh, self).__init__()

        self.vertices = vertices
        self.faces = faces

class PartVolume(nn.Module):
    def __init__(self,
                 part_name,
                 vertices,
                 faces):
        super(PartVolume, self).__init__()

        self.part_name = part_name
        self.smpl_mesh = SMPLMesh(vertices, faces)

        self.part_triangles = None
        self.device = vertices.device

        self.new_vert_ids = []
        self.new_face_ids = []

    def close_mesh(self, boundary_vids):
        # find the center of the boundary
        mean_vert = self.smpl_mesh.vertices[:, boundary_vids, :].mean(dim=1, keepdim=True)
        self.smpl_mesh.vertices = torch.cat([self.smpl_mesh.vertices, mean_vert], dim=1)
        new_vert_idx = self.smpl_mesh.vertices.shape[1]-1
        self.new_vert_ids.append(new_vert_idx)
        # add faces
        new_faces = [[boundary_vids[i + 1], boundary_vids[i], new_vert_idx] for i in range(len(boundary_vids) - 1)]
        self.smpl_mesh.faces = torch.cat([self.smpl_mesh.faces, torch.tensor(new_faces, dtype=torch.long, device=self.device)], dim=0)
        self.new_face_ids += list(range(self.smpl_mesh.faces.shape[0]-len(new_faces), self.smpl_mesh.faces.shape[0]))

    def extract_part_triangles(self, part_vids, part_fid):
        # make sure that inputs are from a watertight part mesh
        batch_size = self.smpl_mesh.vertices.shape[0]

        part_vertices = self.smpl_mesh.vertices[:, part_vids, :]
        part_faces = self.smpl_mesh.faces[part_fid, :]

        part_mean = part_vertices.mean(dim=1, keepdim=True)

        # subtract vert mean because volume computation only applies if origin is inside the triangles
        self.smpl_mesh.vertices = self.smpl_mesh.vertices - part_mean

        # compute triangle
        if self.part_triangles is None:
            # self.part_triangles = torch.index_select(self.smpl_mesh.vertices, 1, self.smpl_mesh.faces.view(-1)).reshape(batch_size, -1, 3, 3)
            self.part_triangles = torch.index_select(self.smpl_mesh.vertices, 1, part_faces.view(-1)).reshape(batch_size, -1, 3, 3)
        else:
            self.part_triangles = torch.cat([self.part_triangles,
                                             torch.index_select(self.smpl_mesh.vertices, 1,
                                                     part_faces.view(-1)).reshape(batch_size, -1, 3, 3)], dim=1)
        # add back vert mean
        self.smpl_mesh.vertices = self.smpl_mesh.vertices + part_mean

    def part_volume(self):
        # Note: the mesh should be enclosing the origin (mean-subtracted)
        # compute volume of triangles by drawing tetrahedrons
        # https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
        x = self.part_triangles[:, :, :, 0]
        y = self.part_triangles[:, :, :, 1]
        z = self.part_triangles[:, :, :, 2]
        volume = (
                         -x[:, :, 2] * y[:, :, 1] * z[:, :, 0] +
                         x[:, :, 1] * y[:, :, 2] * z[:, :, 0] +
                         x[:, :, 2] * y[:, :, 0] * z[:, :, 1] -
                         x[:, :, 0] * y[:, :, 2] * z[:, :, 1] -
                         x[:, :, 1] * y[:, :, 0] * z[:, :, 2] +
                         x[:, :, 0] * y[:, :, 1] * z[:, :, 2]
                 ).sum(dim=1).abs() / 6.0
        return volume

class BodySegment(nn.Module):
    def __init__(self,
                 name,
                 faces,
                 segments_folder,
                 model_type='smpl',
                 append_idx=None):
        super(BodySegment, self).__init__()

        self.name = name
        self.append_idx = faces.max().item() if append_idx is None \
            else append_idx

        self.model_type = model_type
        sb_path = f'{segments_folder}/{model_type}_segments_bounds.pkl'
        sxseg = pickle.load(open(sb_path, 'rb'))

        # read mesh and find faces of segment
        segment_path = f'{segments_folder}/{model_type}_segment_{name}.ply'
        bandmesh = trimesh.load(segment_path, process=False)
        segment_vidx = torch.from_numpy(np.where(
            np.array(bandmesh.visual.vertex_colors[:,0]) == 255)[0])
        self.register_buffer('segment_vidx', segment_vidx)

        # read boundary information
        self.bands = [x for x in sxseg[name].keys()]
        self.bands_verts = [x for x in sxseg[name].values()]
        self.num_bounds = len(self.bands_verts)
        for idx, bv in enumerate(self.bands_verts):
            self.register_buffer(f'bands_verts_{idx}', torch.tensor(bv))
        self.bands_faces = self.create_band_faces()

        # read mesh and find
        segment_faces_ids = np.where(np.isin(faces.cpu().numpy(),
            segment_vidx).sum(1) == 3)[0]
        segment_faces = faces[segment_faces_ids,:]
        segment_faces = torch.cat((faces[segment_faces_ids,:],
            self.bands_faces), 0)
        self.register_buffer('segment_faces', segment_faces)

        # create vector to select vertices form faces
        tri_vidx = []
        for ii in range(faces.max().item()+1):
            tri_vidx += [torch.nonzero(faces==ii)[0].tolist()]
        self.register_buffer('tri_vidx', torch.tensor(tri_vidx))

    def create_band_faces(self):
        """
            create the faces that close the segment.
        """
        bands_faces = []
        for idx, k in enumerate(self.bands):
            new_vert_idx = self.append_idx + 1 + idx
            new_faces = [[self.bands_verts[idx][i+1], \
                self.bands_verts[idx][i], new_vert_idx] \
                for i in range(len(self.bands_verts[idx])-1)]
            bands_faces += new_faces

        bands_faces_tensor = torch.tensor(
            np.array(bands_faces).astype(np.int64), dtype=torch.long)

        return bands_faces_tensor

    def get_closed_segment(self, vertices):
        """
            create the closed segment mesh from SMPL-X vertices.
        """
        vertices = vertices.detach().clone()
        # append vertices to smpl, that close the segment and compute faces
        for idx in range(self.num_bounds):
            bv = eval(f'self.bands_verts_{idx}')
            close_segment_vertices = torch.mean(vertices[:, bv,:], 1,
                                    keepdim=True)
            vertices = torch.cat((vertices, close_segment_vertices), 1)
        segm_triangles = vertices[:, self.segment_faces, :]

        return segm_triangles

    # def has_self_isect_verts(self, vertices, thres=0.99):
    #     """
    #         check if segment (its vertices) are self intersecting.
    #     """
    #     segm_triangles = self.get_closed_segment(vertices)
    #     segm_verts = vertices[:,self.segment_vidx,:]
    #
    #     # do inside outside segmentation
    #     exterior = winding_numbers(segm_verts, segm_triangles) \
    #                 .le(thres)
    #
    #     return exterior
    #
    # def has_self_isect_points(self, points, triangles, thres=0.99):
    #     """
    #         check if points on segment are self intersecting.
    #     """
    #     smpl_verts = triangles[:,self.tri_vidx[:,0], self.tri_vidx[:,1],:]
    #     segm_triangles = self.get_closed_segment(smpl_verts)
    #
    #     # do inside outside segmentation
    #     exterior = winding_numbers(points, segm_triangles) \
    #                 .le(thres)
    #
    #     return exterior

class BatchBodySegment(nn.Module):
    def __init__(self,
                 names,
                 faces,
                 segments_folder,
                 model_type='smpl',
                 device='cuda'):
        super(BatchBodySegment, self).__init__()
        self.names = names
        self.num_segments = len(names)
        self.nv = faces.max().item()

        self.model_type = model_type
        sb_path = f'{segments_folder}/{model_type}_segments_bounds.pkl'
        sxseg = pickle.load(open(sb_path, 'rb'))

        self.append_idx = [len(b) for a,b in sxseg.items() \
            for c,d in b.items() if a in self.names]
        self.append_idx = np.cumsum(np.array([self.nv] + self.append_idx))

        self.segmentation = {}
        for idx, name in enumerate(names):
            self.segmentation[name] = BodySegment(name, faces, segments_folder,
                model_type).to('cuda')

    def batch_has_self_isec_verts(self, vertices):
        """
            check is mesh is intersecting with itself
        """
        exteriors = []
        for k, segm in self.segmentation.items():
            exteriors += [segm.has_self_isect_verts(vertices)]
        return exteriors


class HDfier():
    def __init__(self, model_type='smpl', device='cuda'):
        hd_operator_path = os.path.join(ESSENTIALS_DIR, 'hd_model', model_type,
                                    f'{model_type}_neutral_hd_vert_regressor_sparse.npz')
        hd_operator = np.load(hd_operator_path)
        self.hd_operator = torch.sparse.FloatTensor(
            torch.tensor(hd_operator['index_row_col']),
            torch.tensor(hd_operator['values']),
            torch.Size(hd_operator['size']))
        self.model_type = model_type
        self.device = device

    def hdfy_mesh(self, vertices):
        """
        Applies a regressor that maps SMPL vertices to uniformly distributed vertices
        """
        # device = body.vertices.device
        # check if vertices ndim are 3, if not , add a new axis
        if vertices.dim() != 3:
            # batchify the vertices
            vertices = vertices[None, :, :]

        # check if vertices are an ndarry, if yes, make pytorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(self.device)

        vertices = vertices.to(torch.double)

        if self.hd_operator.device != vertices.device:
            self.hd_operator = self.hd_operator.to(vertices.device)
        hd_verts = sparse_batch_mm(self.hd_operator, vertices).to(torch.float)
        return hd_verts
    
def sparse_batch_mm(m1, m2):
    """
    https://github.com/pytorch/pytorch/issues/14489

    m1: sparse matrix of size N x M
    m2: dense matrix of size B x M x K
    returns m1@m2 matrix of size B x N x K
    """

    batch_size = m2.shape[0]
    # stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
    m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
    result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
               .transpose(1, 0)
    return result

class StabilityLossCoP(nn.Module):
    def __init__(self,
                 faces,
                 cop_w = 10,
                 cop_k = 100,
                 contact_thresh=0.1,
                 model_type='smpl',
                 device='cuda',
    ):
        super().__init__()
        """
        Loss that ensures that the COM of the SMPL mesh is close to the center of support 
        """
        if model_type == 'smpl':
            num_faces = 13776
            num_verts_hd = 20000

        assert faces is not None, 'Faces tensor is none'
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long).to(device)
        self.register_buffer('faces', faces)

        self.cop_w = cop_w
        self.cop_k = cop_k
        self.contact_thresh = contact_thresh

        self.hdfy_op = HDfier(model_type=model_type)

        with open(SMPL_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        with open(PART_VID_FID, 'rb') as f:
            self.part_vid_fid = pkl.load(f)

        # mapping between vid_hd and fid
        with open(HD_SMPL_MAP, 'rb') as f:
            faces_vert_is_sampled_from = pkl.load(f)['faces_vert_is_sampled_from']
        index_row_col = torch.stack(
            [torch.LongTensor(np.arange(0, num_verts_hd)), torch.LongTensor(faces_vert_is_sampled_from)], dim=0)
        values = torch.ones(num_verts_hd, dtype=torch.float)
        size = torch.Size([num_verts_hd, num_faces])
        hd_vert_on_fid = torch.sparse.FloatTensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, 'rb') as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros([len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.

        # mapping between vid_hd and part label
        self.hd_vid_in_part = self.vertex_id_to_part_mapping(hd_vert_on_fid, fid_to_part)

    def compute_triangle_area(self, triangles):
        ### Compute the area of each triangle in the mesh
        # Compute the cross product of the two vectors of each triangle
        # Then compute the length of the cross product
        # Finally, divide by 2 to get the area of each triangle

        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1,0).to(vertices.device)

    def vertex_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vertex_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def forward(self, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vertex_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        # # get COM of the SMPLX mesh
        # triangles = torch.index_select(vertices, 1, self.faces.view(-1)).reshape(batch_size, -1, 3, 3)
        # triangle_centroids = torch.mean(triangles, dim=2)
        # triangle_area = self.compute_triangle_area(triangles)
        # com_naive = torch.einsum('bij,bi->bj', triangle_centroids, triangle_area) / torch.sum(triangle_area, dim=1)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 1] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (1-self.cop_k*vertex_height) + outside_mask *  torch.exp(-self.cop_w * vertex_height)
        cop = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (torch.sum(pressure_weights, dim=1, keepdim=True) +eps)

        # naive center of support
        # vertex_height_robustified = GMoF_unscaled(rho=self.gmof_rho)(vertex_height)
        contact_confidence = torch.sum(pressure_weights, dim=1)
        # contact_mask = (vertex_height < self.contact_thresh).float()
        # num_contact_verts = torch.sum(contact_mask, dim=1)
        # contact_centroid_naive = torch.sum(vertices_hd * contact_mask[:, :, None], dim=1) / (torch.sum(contact_mask, dim=1) + eps)

        # project com, cop to ground plane (x-z plane)
        # weight loss by number of contact vertices to zero out if zero vertices in contact
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        contact_centroid_xz = torch.stack([cop[:, 0], torch.zeros_like(cop)[:, 0], cop[:, 2]], dim=1)
        # stability_loss = (contact_confidence * torch.norm(com_xz - contact_centroid_xz, dim=1)).sum(dim=-1)
        stability_loss = (torch.norm(com_xz - contact_centroid_xz, dim=1))
        com_np = com.detach().cpu().numpy()
        cop_np = cop.detach().cpu().numpy()
        com_xz_np = com_xz.detach().cpu().numpy()
        return stability_loss, com_np[0], cop_np[0], com_xz_np[0]