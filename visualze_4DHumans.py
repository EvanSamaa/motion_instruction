import pickle as pkl
import gzip
# joblib is used to load the model outputs
import joblib
import polyscope as ps
import polyscope.imgui as psim
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath("/scratch/ondemand27/evanpan/motion_instruction/smpl"))
from smpl_np import SMPLModel
# from smpl.smpl_webuser.serialization import load_model
# from phalp.utils import smpl_utils

if __name__ == "__main__":

    motionData_path = "/scratch/ondemand27/evanpan/motion_instruction/4D-Humans/outputs/results/demo_biggerspin1.pkl"
    motiondata = joblib.load(open(motionData_path, "rb"))
    # load smpl model
    smpl = SMPLModel('/scratch/ondemand27/evanpan/motion_instruction/smpl/models/model.pkl')

    frames = sorted(list(motiondata.keys()))
main_guy = 1
# for i in range(0, len(frames)):
for i in range(0, 1):    
    frame_i = motiondata[frames[i]]
    # print(frame_i.keys())
    pose_i = frame_i["pose"] # this stores a list of poses for characters in the 
    center_i = frame_i["center"]
    scale_i = frame_i["scale"]
    smpl_i = frame_i["smpl"]
    ids_i = frame_i['tid']
    threeD_joints_i = frame_i['3d_joints']

    # only care about the guy of interest
    pose_i_main = pose_i[ids_i.index(main_guy)]
    threeD_joints_i_main = threeD_joints_i[ids_i.index(main_guy)] # shape is (43, 3)
    smpl_i_main = smpl_i[ids_i.index(main_guy)] # dictionary with keys ['global_orient', 'body_pose', 'betas']
    
    # get the smpl parameters
    global_orient = smpl_i_main["global_orient"] # defined root orentation (rotation matrix )
    body_pose = smpl_i_main["body_pose"] # defines the rotation matrix of the 23 joints (generated using Rodrigues formula)
    beta = smpl_i_main["betas"] # shape is (10,)

    # convert rotation matrices in body pose to axis-angle representation
    body_pose_aa = np.zeros((body_pose.shape[0], 3))
    for i in range(body_pose.shape[0]):
        body_pose_aa[i] = cv2.Rodrigues(body_pose[i])[0].squeeze()
    global_orient_aa = np.array([cv2.Rodrigues(global_orient[0])[0].squeeze()])
    body_pose_aa = np.concatenate([global_orient_aa, body_pose_aa], axis=0)
    smpl.set_params(beta=beta, pose=body_pose_aa, trans=np.zeros([3, ]))
    vertices = smpl.verts
    faces = smpl.faces
    # use polyscope to show the mesh
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_program_name("Interactive Viewer")
    ps.set_ground_plane_mode("none")
    ps.set_view_projection_mode("orthographic")
    ps.set_autocenter_structures(False)
    ps.set_autoscale_structures(False)
    ps.set_front_dir("z_front")
    ps.set_background_color([0,0,0])
    ps.init()
    SM0 = ps.register_surface_mesh("original", vertices, faces, color=[0.9,0.9,0.9], smooth_shade=True, edge_width=0.25, material="normal")
    ps.show()
    # fig = plt.figure() 
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(threeD_joints_i_main[:, 0], threeD_joints_i_main[:, 1], threeD_joints_i_main[:, 2])
    # plt.show()