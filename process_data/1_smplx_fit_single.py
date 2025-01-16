''' Convert data from IMavatar to SCARF data format
'''
from ipaddress import ip_address
import os
from pathlib import Path
from re import S
from socket import IP_DEFAULT_MULTICAST_LOOP
import sys
import os.path as osp
import numpy as np
from glob import glob
import pickle
import torch
import argparse
from skimage.io import imread, imsave
import cv2
from pytorch3d.io import load_obj
from tqdm import tqdm
import shutil
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.renderer import (look_at_view_transform, FoVOrthographicCameras, PointsRasterizationSettings,
                                PointsRenderer, PulsarPointsRenderer, PointsRasterizer, AlphaCompositor,
                                NormWeightedCompositor, PerspectiveCameras)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    look_at_rotation,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    SoftSilhouetteShader,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.utils.deca_render import SRenderY
from lib.utils import camera_util
from lib.utils import util, lossfunc
from lib.model.smplx import SMPLX
from lib.model.FLAME import FLAMETex, texture_flame2smplx
from lib.core.config import cfg, update_cfg
from lib.utils import rotation_converter
from lib.render.mesh_helper import pytorch3d_rasterize, render_shape, render_texture
from lib.dataset.face_video_old import NerfDataset
from lib.utils.convert_camera_params import load_K_Rt_from_P, convert_camera_params, convert_opencv_to_pytorch3d
from pytorch3d.renderer import join_cameras_as_batch
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

model_cfg = cfg.model


class PoseModel(nn.Module):

    def __init__(self,
                 dataset,
                 optimize_cam=False,
                 use_perspective=False,
                 use_appearance=False,
                 appearance_dim=0,
                 use_deformation=False,
                 use_light=False,
                 n_lights=3,
                 deformation_dim=0,
                 cameras = None):
        super(PoseModel, self).__init__()
        self.device = 'cuda:0'
        self.subject_id = dataset.subject_id
        self.use_perspective = use_perspective
        self.cameras = cameras
        #print(self.posemodel.state_dict())

        init_exp = torch.zeros([1, model_cfg.n_exp]).float().to(self.device)
        init_light = torch.zeros([1, 9, 3]).float().to(self.device)

        init_full_pose = torch.zeros([55, 3]).float().to(self.device) + 0.00001
        init_full_pose[0, 0] = np.pi
        ## init shoulder
        init_full_pose[16,2] = -np.pi*60/180
        init_full_pose[17,2] = np.pi*60/180

        init_full_pose = rotation_converter.batch_euler2axis(init_full_pose)[None, ...]
        self.init_full_pose = init_full_pose

        print(f"Dataset: {dataset.data}\n")
        original_filenames = sorted(cameras.keys(), key=lambda x: int(x))  # Sort keys numerically
        processed_filenames = sorted(dataset.data)  # Sorted list of processed images from dataset

        # Ensure lengths match
        if len(processed_filenames) > len(original_filenames):
            raise ValueError("More processed images than available camera parameters.")
        if len(original_filenames) > len(processed_filenames):
            print("Warning: Some original frames are not processed.")

        filename_mapping = {
            Path(processed_filenames[i]).name: original_filenames[i] for i in range(len(processed_filenames))
        }
        print(f"Filename mapping (processed -> original): {list(filename_mapping.items())[:10]}")

        processed_camera_params = {
            Path(processed_filenames[i]).name: cameras[original_filenames[i]]
            for i in range(len(processed_filenames))
            if original_filenames[i] in cameras
        }

        print(f"Filtered camera parameters: {len(processed_camera_params)} available for processed frames.")

        if self.use_perspective:
            for processed_name, camera_param in processed_camera_params.items():
                print(f"\nProcessed frame: {processed_name}")
                name = self.subject_id

                P_original = camera_param.transpose(0, 1)[:3, :4].numpy()
                print(f"Original Projection Matrix (P_original):\n{P_original}")

                original_width, original_height = 1920, 1080  # Original resolution
                new_width, new_height = 512, 512  # Target resolution

                # projection_matrix = convert_camera_params(P_original, original_width, original_height, new_width, new_height)

                projection_matrix = convert_camera_params(P_original, scale_x=512, scale_y=512)
                P = projection_matrix[:3, :4]
                print(f"Converted Projection Matrix (P):\n{P}")

                cam_intr, pose = load_K_Rt_from_P(P=P)
                cam_extr = np.linalg.inv(pose)

                print(f"Camera Intrinsics:\n{cam_intr}")
                print(f"Camera Pose:\n{pose}")
                print(f"Camera Extrinsics:\n{cam_extr}")

                frame_id = int(Path(processed_name).stem.split('_f')[-1])
                print(f"Frame ID: {frame_id}")

                self.register_buffer(f'{name}_cam_intr_{frame_id}', torch.from_numpy(cam_intr).float())
                self.register_buffer(f'{name}_cam_extr_{frame_id}', torch.from_numpy(cam_extr).float())

                self.register_parameter(f'{name}_pose_{frame_id}', torch.nn.Parameter(init_full_pose))
                self.register_parameter(f'{name}_light_{frame_id}', torch.nn.Parameter(init_light))
                self.register_parameter(f'{name}_exp_{frame_id}', torch.nn.Parameter(init_exp))
        else:
            ## initialize
            # assume: global pose zero (body facing front), change head pose
            # optimize: cam, beta, head/neck pose, exp
            lmk = dataset[0]['lmk']
            ymin = lmk[:, 1].min()
            ymax = lmk[:, 1].max()
            yrange = ymax - ymin
            init_scale = 7 * yrange      
            print('init_scale', init_scale)   
            init_cam = torch.tensor([
                init_scale,
                0.02,
                # init_cam = torch.tensor([6.2, 0.02,
                1
            ]).float().reshape(1, 3).to(self.device)  # good initialization
            for imagepath in dataset.data:
                imagename = Path(imagepath).stem
                frame_id = int(imagename.split('_f')[-1])
                name = self.subject_id
                # init cam
                self.register_parameter(f'{name}_cam_{frame_id}', torch.nn.Parameter(init_cam))
                # init full pose
                self.register_parameter(f'{name}_pose_{frame_id}', torch.nn.Parameter(init_full_pose))
                self.register_parameter(f'{name}_light_{frame_id}', torch.nn.Parameter(init_light))
                self.register_parameter(f'{name}_exp_{frame_id}', torch.nn.Parameter(init_exp))


    def forward(self, batch, extra_fix_idx=None):
        # return poses of given frame_ids
        name = self.subject_id
        if 'cam_id' in batch.keys():
            cam_id = batch['cam_id']
            names = [f'{name}_{cam}' for cam in cam_id]
        else:
            names = [name] * len(batch['frame_id'])
        frame_ids = batch['frame_id']
        batch_size = len(frame_ids)

        # Prepare pose for each frame in the batch
        batch_pose = torch.cat([
            getattr(self, f'{names[i]}_pose_{frame_ids[i]}') for i in range(batch_size)
            ])
        batch_pose = rotation_converter.batch_axis2matrix(batch_pose.reshape(-1, 3)).reshape(batch_size, 55, 3, 3)
        batch['init_full_pose'] = rotation_converter.batch_axis2matrix(self.init_full_pose.clone().reshape(
            -1, 3)).reshape(batch_size, 55, 3, 3)
        batch['full_pose'] = batch_pose
        # do not optimize body pose
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        fix_idx = list(range(1, 12)) + [13, 14, 18, 19, 20, 21] + list(range(25, 55))
        if extra_fix_idx is not None:
            fix_idx += extra_fix_idx
        batch['full_pose'][:, fix_idx] = batch[
            'init_full_pose'][:,
                              fix_idx]
        
        if self.use_perspective:
            # Retrieve camera intrinsics and extrinsics
            cam_intr_list = [getattr(self, f'{name}_cam_intr_{frame_ids[i]}') for i in range(batch_size)]
            cam_extr_list = [getattr(self, f'{name}_cam_extr_{frame_ids[i]}') for i in range(batch_size)]
            batch['cam_intr'] = torch.stack(cam_intr_list).to(self.device)
            batch['cam_extr'] = torch.stack(cam_extr_list).to(self.device)
        else:
            batch['cam'] = torch.cat([getattr(self, f'{names[i]}_cam_{frame_ids[i]}') for i in range(batch_size)])
            print(f"Camera parameters for batch: {batch['cam']}")
        
        batch['exp'] = torch.cat([getattr(self, f'{names[i]}_exp_{frame_ids[i]}') for i in range(batch_size)])
        batch['light'] = torch.cat([getattr(self, f'{names[i]}_light_{frame_ids[i]}') for i in range(batch_size)])
        return batch


class SMPLX_optimizer(torch.nn.Module):

    def __init__(self, dataset=None, device='cuda:0', image_size=512, light_type='SH', use_perspective=False, cameras=None):
        super(SMPLX_optimizer, self).__init__()
        self.cfg = cfg
        self.device = device
        self.image_size = image_size
        self.light_type = light_type
        # smplx_cfg.model.n_shape = 100
        # smplx_cfg.model.n_exp = 50
        self.dataset = dataset
        self.use_perspective = use_perspective
        self.cameras = cameras
        self._setup_model()
        self._setup_renderer()
        self._setup_loss_weight()
        self.configure_optimizers()
        # loss
        # self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=model_cfg.fr_model_path)

    def _setup_model(self):
        ## pose model
        self.posemodel = PoseModel(dataset=self.dataset,
                                   optimize_cam=True,
                                   #use_perspective=False,
                                   use_perspective=self.use_perspective,
                                   use_appearance=False,
                                   use_deformation=False,
                                   use_light=True,
                                   cameras=self.cameras,
                                   n_lights=9).to(self.device)

        ## smplx model
        self.smplx = SMPLX(model_cfg).to(self.device)
        self.flametex = FLAMETex(model_cfg).to(self.device)
        self.verts = self.smplx.v_template
        self.faces = self.smplx.faces_tensor
        ## iris index
        self.idx_iris = [9503, 10049]  #right, left
        ## part index
        ##--- load vertex mask
        with open(self.cfg.model.mano_ids_path, 'rb') as f:
            hand_idx = pickle.load(f)
        flame_idx = np.load(self.cfg.model.flame_ids_path)
        with open(self.cfg.model.flame_vertex_masks_path, 'rb') as f:
            flame_vertex_mask = pickle.load(f, encoding='latin1')
        # verts = torch.nn.Parameter(self.v_template, requires_grad=True)
        exclude_idx = []
        exclude_idx += list(hand_idx['left_hand'])
        exclude_idx += list(hand_idx['right_hand'])
        exclude_idx += list(flame_vertex_mask['face'])
        exclude_idx += list(flame_vertex_mask['left_eyeball'])
        exclude_idx += list(flame_vertex_mask['right_eyeball'])
        exclude_idx += list(flame_vertex_mask['left_ear'])
        exclude_idx += list(flame_vertex_mask['right_ear'])
        all_idx = range(self.smplx.v_template.shape[1])
        face_idx = list(flame_vertex_mask['face'])
        body_idx = [i for i in all_idx if i not in face_idx]
        self.part_idx_dict = {
            'face': flame_vertex_mask['face'],
            'hand': list(hand_idx['left_hand']) + list(hand_idx['right_hand']),
            'exclude': exclude_idx,
            'body': body_idx
        }

    def configure_optimizers(self):
        ###--- optimizer for training nerf model.
        # whether to use apperace code
        # nerf_params = list(self.model.mlp_coarse.parameters()) + list(self.model.mlp_fine.parameters())
        init_beta = torch.zeros([1, model_cfg.n_shape]).float().to(self.device)
        self.register_parameter('beta', torch.nn.Parameter(init_beta))
        init_tex = torch.zeros([1, model_cfg.n_tex]).float().to(self.device)
        self.register_parameter('tex', torch.nn.Parameter(init_tex))
        parameters = [{'params': [self.beta, self.tex], 'lr': 1e-3}]


        pose_parameters = [param for name, param in self.posemodel.named_parameters() if 'pose' in name or 'light' in name or 'exp' in name]
        parameters.append({'params': pose_parameters, 'lr': 1e-3})
        
        # if not self.use_perspective: #check if this is helpful for perspective
        #     cam_parameters = [param for name, param in self.posemodel.named_parameters() if 'cam' in name]
        #     parameters.append({'params': cam_parameters, 'lr': 1e-2})
        
        cam_parameters = [param for name, param in self.posemodel.named_parameters() if 'cam' in name]
        parameters.append({'params': cam_parameters, 'lr': 1e-2})

        self.optimizer = torch.optim.Adam(params=parameters)

    def _setup_renderer(self):
        ## setup raterizer
        uv_size = 1024
        topology_path = self.cfg.model.topology_path
        # cache data for smplx texture
        self.smplx_texture = imread(self.cfg.model.smplx_tex_path) / 255.
        self.cached_data = np.load(self.cfg.model.flame2smplx_cached_path, allow_pickle=True, encoding='latin1').item()

        self.render = SRenderY(self.image_size,
                               obj_filename=topology_path,
                               uv_size=uv_size,
                               rasterizer_type='pytorch3d').to(self.device)
        mask = imread(self.cfg.model.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_eye_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)

        # face region mask in flame texture map
        mask = imread(self.cfg.model.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.flame_face_mask = F.interpolate(mask, [self.cfg.model.uv_size, self.cfg.model.uv_size]).to(self.device)

        ########### silhouette rendering
        ## camera
        if self.use_perspective:
            self.cameras = None

        else:
            R = torch.eye(3).unsqueeze(0)
            T = torch.zeros([1, 3])
            batch_size = 1
            self.cameras = pytorch3d.renderer.cameras.FoVOrthographicCameras(
                R=R.expand(batch_size, -1, -1),
                T=T.expand(batch_size, -1),
                znear=0.0).to(self.device)

        blend_params = BlendParams(sigma=1e-7, gamma=1e-4)
        raster_settings = RasterizationSettings(image_size=self.image_size,
                                                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                                                faces_per_pixel=50,
                                                bin_size=0)
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        self.silhouette_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras,
                                                                          raster_settings=raster_settings),
                                                shader=SoftSilhouetteShader(blend_params=blend_params))

    def forward_model(self, batch, returnMask=False, returnRendering=False, returnNormal=False):
        ''' forward SMPLX model
        Args:
            batch: dict, batch data
                'beta': [B, n_shape(200)], shape parameters
                'exp': [B, n_exp(100)], expression parameters
                'full_pose': [B, n_pose(55), 3, 3], pose parameters, in Rotatation matrix format
                'cam': [B, 3], camera parameters, [scale, tx, ty], use orthographic projection
            returnMask: bool, whether to return mask
            returnRendering: bool, whether to return rendering
            returnNormal: bool, whether to return normal
        Returns:
            opdict: dict, output dict
            
        '''
        opdict = {}
        verts, landmarks, joints = self.smplx(shape_params=batch['beta'],
                                              expression_params=batch['exp'],
                                              full_pose=batch['full_pose'])
        #cameras = None

        print(f"FORWARD_MODEL BEFORE IF use_perspective")
        print(f"Verts: {verts}")
        print(f"Landmarks: {landmarks}")


        cams = []
        if self.use_perspective:
            print("\n=== Starting Perspective Projection ===\n")
            cam_intr = batch['cam_intr']
            cam_extr = batch['cam_extr']


            batch_size = cam_intr.shape[0]

            # height, width = 1920, 1080
            height, width = 512, 512

            image_size = torch.tensor([[height, width]] * batch_size, dtype=torch.float32).to(self.device)

            K = cam_intr[:, :3, :3]
            R = cam_extr[:, :3, :3]
            T = cam_extr[:, :3, 3]
            cameras = cameras_from_opencv_projection(
            camera_matrix=K,
            R=R,
            tvec=T,
            image_size=image_size).cuda()

            print("\nCreated PyTorch3D PerspectiveCameras object.")
            cams.append(cameras)
            cameras = join_cameras_as_batch(cams).to(self.device)
            
            trans_verts = cameras.transform_points_screen(verts)
            pred_lmk = cameras.transform_points_screen(landmarks)

            print(f"Transformed Vertices: {trans_verts.shape}")
            print(f"Transformed Landmarks: {pred_lmk.shape}")

            # pred_lmk = pred_lmk[..., :2]
            pred_lmk = pred_lmk / self.image_size
            pred_lmk = pred_lmk * 2. - 1
            pred_lmk = pred_lmk[..., :2]

            # convert smpl-x landmarks to flame landmarks (right order)
            #pred_lmk = torch.cat([pred_lmk[:, -17:], pred_lmk[:, :-17]], dim=1)
            #pred_lmk = (pred_lmk + 1) * (self.image_size / 2.0)
            print(f"Predicted Landmarks after Scaling:\n{pred_lmk}")
            print(f"pred_lmk shape in forward_model: {pred_lmk.shape}")
            print(f"Transformed Vertices: {trans_verts.shape}")
            
        else:
            cam = batch['cam']
            trans_verts = util.batch_orth_proj(verts, cam)
            pred_lmk = util.batch_orth_proj(landmarks, cam)[:, :, :2]
            # convert smpl-x landmarks to flame landmarks (right order)
            pred_lmk = torch.cat([pred_lmk[:, -17:], pred_lmk[:, :-17]], dim=1)

        opdict['verts'] = verts
        opdict['trans_verts'] = trans_verts
        opdict['pred_lmk'] = pred_lmk

        print(f"FORWARD_MODEL AFTER BEFORE IF use_perspective")
        print(f"Verts: {verts}")
        print(f"Landmarks: {landmarks}")

        print(f"\nFinal Output Shapes:")
        print(f"verts: {verts.shape}")
        print(f"trans_verts: {trans_verts.shape}")
        print(f"pred_lmk: {pred_lmk.shape}")

        if 'iris' in batch.keys():
            iris_verts = verts[:, self.idx_iris, :]
            pred_iris = cameras.transform_points_screen(iris_verts)
            # pred_iris = pred_iris[..., :2]
            pred_iris = pred_iris / self.image_size
            pred_iris = pred_iris * 2. - 1
            pred_iris = pred_iris[..., :2]
            #pred_iris = pred_iris / (self.image_size / 2.0) - 1.0
            opdict['pred_iris'] = pred_iris

        # render mask for silhouette loss
        batch_size = verts.shape[0]
        faces = self.faces.unsqueeze(0).expand(batch_size,-1,-1)
        if returnMask:
            trans_verts_mask = trans_verts.clone()
            trans_verts_mask[:,:,:2] = -trans_verts_mask[:,:,:2]
            trans_verts_mask[:,:,-1] = -trans_verts_mask[:,:,-1] + 50
            mesh = Meshes(
                verts = trans_verts_mask,
                faces = faces
            )
            if self.use_perspective:
                render_cameras = cameras
                if render_cameras is None:
                    raise ValueError("Cameras are not defined for perspective projection.")
            else:
                render_cameras = self.cameras
            #mesh_mask = self.silhouette_renderer(meshes_world=mesh).permute(0, 3, 1, 2)[:,3:]
            mesh_mask = self.silhouette_renderer(meshes_world=mesh, cameras=render_cameras).permute(0, 3, 1, 2)[:, 3:]
            opdict['mesh_mask'] = mesh_mask
        # render image for image loss
        if returnRendering:
            # calculate normal for shading
            normal_verts = trans_verts.clone()
            normal_verts[..., 0] = -normal_verts[..., 0]
            normals = util.vertex_normals(normal_verts, faces)                    
            trans_verts[..., -1] = trans_verts[..., -1] + 50
            albedo = self.flametex(batch['tex'])
            rendering_out = self.render(verts,
                                trans_verts,
                                albedo,
                                lights=batch['light'],
                                light_type=self.light_type,
                                given_normal=normals)
            opdict['image'] = rendering_out['images']
            opdict['albedo_image'] = rendering_out['albedo_images']
            opdict['shading_image'] = rendering_out['shading_images']
        
        return opdict

    def _setup_loss_weight(self):
        loss_cfg = cfg.loss
        loss_cfg.lmk = 1.
        loss_cfg.eyed = 2.
        loss_cfg.lipd = 0.5
        # mask
        loss_cfg.inside_mask = 1.
        loss_cfg.mesh_mask = 1.
        # image
        loss_cfg.image = 2.
        loss_cfg.albedo = 2.
        
        self.loss_cfg = loss_cfg

    def optimize(self, batch, iters, vis_step=100, vispath=None, 
                    data_type='else',
                    lmk_only=True,
                    use_mask = False,
                    use_rendering = False,
                    use_normal = False):
        ''' optimize the pose and shape parameters of the model, using lmk loss only
        # global: 0, neck: 12, head: 15, leftarm: 16, rightarm: 17, jaw: 22, lefteye: 23, righteye: 24
        '''
        os.makedirs(vispath, exist_ok=True)
        for iter in tqdm(range(iters)):
            # first stage, only optimize global and neck pose
            if data_type == 'fix_shoulder':
                extra_fix_idx = [15, 16, 17, 23, 24, 22]
            elif data_type == 'fix_neck':
                extra_fix_idx = [12, 15, 16, 17, 23, 24, 22]
            else:
                extra_fix_idx = []
            image = batch['image']
            batch_size = image.shape[0]
            batch['beta'] = self.beta.expand(batch_size, -1)
            batch['tex'] = self.tex.expand(batch_size, -1)
            batch = self.posemodel(batch, extra_fix_idx=extra_fix_idx)
            if 'cam' in batch:
                print(f"Optimized camera parameters: {batch['cam']}")
            elif 'cam_intr' in batch and 'cam_extr' in batch:
                print("Using provided camera intrinsics and extrinsics.")
            else:
                print("Camera parameters not found in batch.")
            #print(f"Optimized camera parameters: {batch['cam']}")
            opdict = self.forward_model(batch, returnMask=use_mask, returnRendering=use_rendering, returnNormal=use_normal)

            print(f"OPTIMIZE METHOD BEFORE 'losses dict' ")
            print(f"opdict['verts']: {opdict['trans_verts']}")
            print(f"opdict['trans_verts']: {opdict['verts']}")
            print(f"opdict['pred_lmk']: {opdict['pred_lmk']}")
            losses = {}
            if 'iris' in batch.keys():
                print(f"----Iris in batch keys---")
                #pred_iris = opdict['trans_verts'][:, self.idx_iris, :2]
                pred_iris = opdict['pred_iris']
                pred_lmk = torch.cat([opdict['pred_lmk'], pred_iris], dim=1)
                print(f"SHAPE LANDMARKS BATCH: {batch['lmk'].shape}")
                print(f"SHAPE IRIS BATCH: {batch['iris'].shape}")
                #gt_lmk = torch.cat([batch['lmk'][:, :], batch['iris'][:, :]], dim=1)
                # gt_lmk = torch.cat([batch['lmk'], batch['iris'][:, :, :3]], dim=1)
                # print(f"pred_lmk shape in optimize: {pred_lmk.shape}")
                if batch['lmk'].dim() == 2:
                    batch['lmk'] = batch['lmk'].unsqueeze(0)
                if batch['iris'].dim() == 2:
                    batch['iris'] = batch['iris'].unsqueeze(0)
                gt_lmk = torch.cat([batch['lmk'], batch['iris'][:, :, :3]], dim=1)

            else:
                print(f"----No iris in batch keys---")
                pred_lmk = opdict['pred_lmk']
                gt_lmk = batch['lmk']
                if gt_lmk.dim() == 2:
                    gt_lmk = gt_lmk.unsqueeze(0)

            print(f"OPTIMIZE METHOD AFTER 'losses dict' ")
            print(f"opdict['verts']: {opdict['trans_verts']}")
            print(f"opdict['trans_verts']: {opdict['verts']}")
            print(f"opdict['pred_lmk']: {opdict['pred_lmk']}")

            print(f"Batch: {batch['lmk']}")
            print(f"Ground truth landmarks shape (gt_lmk): {gt_lmk.shape}")
            print(f"Predicted landmarks shape (pred_lmk): {pred_lmk.shape}")
            print(f"Ground truth landmark range (min/max): {gt_lmk.min()}, {gt_lmk.max()}")
            print(f"Predicted landmark range (min/max): {pred_lmk.min()}, {pred_lmk.max()}")
            print(f"Ground truth landmarks (gt_lmk): {gt_lmk}")
            print(f"Predicted landmarks (pred_lmk): {pred_lmk}")
            print(f"Check dim shape predlmk: {pred_lmk.shape[-1]}")

            if gt_lmk.dim() == 2:
                print(f"gt_lmk dim == 2")
                gt_lmk = gt_lmk.unsqueeze(0)

            gt_pixel = (gt_lmk + 1.0) * (self.image_size / 2.0)
            pred_pixel = (pred_lmk + 1.0) * (self.image_size / 2.0)

            plt.figure(figsize=(8, 8))
            plt.scatter(gt_pixel[0, :, 0].cpu().numpy(), gt_pixel[0, :, 1].cpu().numpy(), label='Ground Truth (Pixels)', color='blue')
            plt.scatter(pred_pixel[0, :, 0].detach().cpu().numpy(), pred_pixel[0, :, 1].detach().cpu().numpy(), label='Predicted (Pixels)', color='red')
            plt.gca().invert_yaxis()  # Invert Y for image coordinates
            plt.legend()
            plt.title('Landmarks in Pixel Space')
            plt.show()

            # lmk loss, use confidence, if confidence is 0, then ignore this points (e.g. especially for iris points)
            losses['lmk'] = lossfunc.batch_kp_2d_l1_loss(gt_lmk, pred_lmk)*self.loss_cfg.lmk
            # eye distance loss from DECA
            if self.cfg.loss.eyed > 0.:
                losses['eyed'] = lossfunc.eyed_loss(pred_lmk[:,:68,:2], gt_lmk[:,:68,:2])*self.cfg.loss.eyed
            
            if use_mask:
                losses['mesh_inside_mask'] = (torch.relu(opdict['mesh_mask'] - batch['mask'])).abs().mean()*self.loss_cfg.inside_mask
                mesh_mask = opdict['mesh_mask']
                non_skin_mask = 1-mesh_mask.detach()
                hair_only = non_skin_mask*batch['hair_mask']
                mesh_mask = mesh_mask + hair_only
                opdict['mesh_mask'] = mesh_mask
                losses['mesh_mask'] = lossfunc.huber(opdict['mesh_mask'], batch['mask'])*self.loss_cfg.mesh_mask
            if use_rendering:
                losses['image'] = (batch['face_mask'] *(batch['image'] - opdict['image'])).abs().mean() * self.loss_cfg.image
                losses['image_albedo'] = (batch['face_mask'] *
                                        (batch['image'] - opdict['albedo_image'])).abs().mean() * self.loss_cfg.albedo
            
            losses['reg_shape'] = (torch.sum(batch['beta']**2) / 2).mean() * 1e-5
            losses['reg_exp'] = (torch.sum(batch['exp']**2) / 2).mean() * 1e-5
            losses['reg_tex'] = (torch.sum(batch['tex']**2) / 2).mean() * 5e-5
            # regs in init shoulder
            shoulder_pose = batch['full_pose'][:,16:18]
            shoulder_pose_axis = rotation_converter.batch_matrix2axis(shoulder_pose.reshape(-1, 3, 3)).reshape(batch_size, -1, 3)
            losses['reg_shoulder'] = (shoulder_pose_axis - self.posemodel.init_full_pose[:,16:18,:].detach()).mean().abs()*1000
            
            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            if vis_step < 1000 and iter % vis_step == 0:
                loss_info = f"Iter: {iter}/{iters}: "
                for k, v in losses.items():
                    loss_info = loss_info + f'{k}: {v:.6f}, '
                print(loss_info)
                visdict = {
                    'inputs': image,
                    'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=True),
                    'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=True)
                }
                # render shape
                faces = self.smplx.faces_tensor
                shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                        faces=faces.expand(batch_size, -1, -1),
                                        image_size=image.shape[-1],
                                        background=image)
                visdict['shape'] = shape_image
                for key in opdict.keys():
                    if 'image' in key:
                        visdict[key] = opdict[key]
                util.visualize_grid(visdict, os.path.join(vispath, f'{iter:06}.png'), return_gird=False)
                
        visdict = {
                    'inputs': image,
                    'lmk_gt': util.tensor_vis_landmarks(image, gt_lmk, isScale=True),
                    'lmk_pred': util.tensor_vis_landmarks(image, pred_lmk, isScale=True)
                }
        # render shape
        faces = self.smplx.faces_tensor
        shape_image = render_shape(vertices=opdict['trans_verts'].detach(),
                                faces=faces.expand(batch_size, -1, -1),
                                image_size=image.shape[-1],
                                background=image)
        visdict['shape'] = shape_image
        for key in opdict.keys():
            if 'image' in key:
                visdict[key] = opdict[key]
        util.visualize_grid(visdict, os.path.join(vispath, f'{iter:06}.png'), return_gird=False)
        ## save obj
        util.write_obj(os.path.join(vispath, f'{iter:06}.obj'),
                    opdict['trans_verts'][0].detach().cpu().numpy(), 
                    faces.cpu().numpy())
        print(os.path.join(vispath, f'{iter:06}.obj'))
        return pred_lmk
            
    def run(self,
            savepath=None,
            iter_list=[500, 500, 2000],
            batch_size=1,
            vis_step=100,
            data_type='fix_shoulder',
            args=None):
        '''
        Args:
            iter_list: list, number of iterations for each stage
            savepath: str, path to save the results
            args: dict, additional arguments
        '''
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=min(batch_size, 4),
                                                 pin_memory=True,
                                                 drop_last=False)
        subject = self.dataset.subject_id
        for batch in dataloader:
            print(f"Batch keys: {batch.keys()}")
            #if 'lmk' in batch:
            #    print(f"Landmarks shape: {batch['lmk'].shape}")
            #    print(f"Landmarks data: {batch['lmk']}")
            #break
            print(batch)
            util.move_dict_to_device(batch, self.device)
            frame_id = batch['frame_id'][0]
            name = f'{frame_id:06}'
            self.optimize(batch,
                        iter_list[0],
                        vis_step=vis_step,
                        vispath=os.path.join(savepath, name, 'stage_0'),
                        data_type=data_type)

            self.optimize(batch,
                        iter_list[1],
                        vis_step=vis_step,
                        vispath=os.path.join(savepath, name, 'stage_1'),
                        use_mask=True)
            self.optimize(batch,
                        iter_list[2],
                        vis_step=vis_step,
                        vispath=os.path.join(savepath, name, 'stage_2'),
                        use_mask=True,
                        use_rendering=True)

            pixie_param = {
                'shape': batch['beta'],
                'full_pose': batch['full_pose'],
                'light': batch['light'],
                #'cam': batch['cam'],
                'exp': batch['exp'],
                'tex': batch['tex']
            }

            # Include camera parameters based on projection type
            if 'cam' in batch:
                pixie_param['cam'] = batch['cam']
            elif 'cam_intr' in batch and 'cam_extr' in batch:
                pixie_param['cam_intr'] = batch['cam_intr']
                pixie_param['cam_extr'] = batch['cam_extr']
            else:
                print("Warning: Camera parameters not found in batch.")
            util.save_params(os.path.join(savepath, f'{subject}_f{frame_id:06}_param.pkl'),
                            pixie_param, ind=0)
            
    def save(self, savepath, dataloader):
        subject = self.posemodel.subject_id
        for batch in tqdm(dataloader):
            util.move_dict_to_device(batch, self.device)
            batch = self.posemodel(batch)
            image = batch['image']
            batch_size = image.shape[0]
            for i in range(batch_size):
                frame_id = batch['frame_id'][i]
                pixie_param = {
                    'shape': self.beta,
                    'full_pose': batch['full_pose'][i],
                    'light': batch['light'][i],
                    'cam': batch['cam'][i],
                    'exp': batch['exp'][i],
                    'tex': self.tex
                }
                util.save_params(os.path.join(savepath, f'{subject}_f{frame_id:06}_param.pkl'), pixie_param)


''' for face parsing from https://github.com/zllrunning/face-parsing.PyTorch/issues/12
 [0 'backgruond' 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--camera_params',
                    type=str,
                    #required=True,
                    default=None,
                    help='Path to the camera parameters file (.pkl)')

    parser.add_argument('--datapath',
                        type=str,
                        default='/is/cluster/yfeng/Data/Projects-data/DELTA/datasets/face')
    parser.add_argument('--data_cfg', type=str, default=None)
    parser.add_argument('--subject', type=str, default='person_0000')
    parser.add_argument('--data_type', type=str, default='fix_shoulder')
    parser.add_argument('--batch_idx', type=int, default=0)
    parser.add_argument('--batch_num', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--iters', type=str, default='2000,500,2000')
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--light_type', type=str, default='SH')
    parser.add_argument('--seed', type=int, default=9988)
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--use_id", default=False, action="store_true")
    # only optimize training frames only if set train_only
    parser.add_argument("--train_only", default=False, action="store_true")
    args = parser.parse_args()

    use_perspective = args.camera_params is not None

    if args.data_cfg is None:
        subject = args.subject
        data_cfg_file = os.path.join('/home/yfeng/Projects/DELTA/configs/data/face', f'{args.subject}.yml')
    else:
        subject = args.subject
        data_cfg_file = args.data_cfg

    # Load camera parameters from the specified file
    if use_perspective:
        with open(args.camera_params, 'rb') as f:
            cameras = pickle.load(f)
    else:
        cameras = None

    data_cfg = update_cfg(cfg, data_cfg_file).dataset
    data_cfg.image_size = args.image_size
    data_cfg.load_lmk = True
    data_cfg.load_normal = True
    data_cfg.load_fits = True
    # batch for cluster running
    imagepath_list = glob(os.path.join(data_cfg.path, subject, 'matting', f'{args.subject}_*.png'))
    imagepath_list = sorted(imagepath_list)
    if args.train_only:
        imagepath_list = imagepath_list[data_cfg.train.frame_start:data_cfg.train.frame_end:data_cfg.train.frame_step]
    imagepath_list = imagepath_list[args.batch_idx * args.batch_num:(args.batch_idx + 1) * args.batch_num]
    savepath = os.path.join(args.datapath, subject, f'smplx_single')
    iters = [int(x) for x in args.iters.split(',')]
    print(f'run tracking, results will be saved in {savepath}')
    for i in range(len(imagepath_list)):
        current_imagepath_list = imagepath_list[i:i + 1]
        dataset = NerfDataset(data_cfg, given_imagepath_list=current_imagepath_list)

        optimizer = SMPLX_optimizer(dataset, args.device,
                                    light_type=args.light_type,
                                    use_perspective=use_perspective,
                                    cameras=cameras)
        optimizer.run(savepath, iters, args.batch_size, vis_step=args.vis_step, data_type=args.data_type)
        
# python smplx_detail_tracking_single.py  --batch_idx 500 --batch_num 1 --batch_size 1  --iters 2000