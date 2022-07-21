import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch3d
from smplx.lbs import batch_rodrigues

from predict.predict_hrnet import predict_hrnet

from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer
import csv
import math
from utils.image_utils import batch_add_rgb_background, batch_crop_pytorch_affine, batch_crop_opencv_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import rot6d_to_rotmat, aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling, joints2D_error_sorted_verts_sampling

# def writeToCSV(image_fname, data,name):
#            with open('./csv_files/'+name, 'a') as f:
#            # create the csv writer
#                writer = csv.writer(f)

# write a row to the csv file
#           writer.writerow([image_fname.split('.')[0], data])


def predict_poseMF_shapeGaussian_net(pose_shape_model,
                                     pose_shape_cfg,
                                     smpl_model,
                                     hrnet_model,
                                     hrnet_cfg,
                                     edge_detect_model,
                                     device,
                                     image_dir,
                                     save_dir,
                                     object_detect_model=None,
                                     joints2Dvisib_threshold=0.75,
                                     visualise_wh=512,
                                     visualise_uncropped=True,
                                     visualise_samples=False):
    """
    Predictor for SingleInputKinematicPoseMFShapeGaussianwithGlobCam on unseen test data.
    Input --> ResNet --> image features --> FC layers --> MF over pose and Diagonal Gaussian over shape.
    Also get cam and glob separately to distribution predictor.
    Pose predictions follow the kinematic chain.
    """
    # Setting up body visualisation renderer
    body_vis_renderer = TexturedIUVRenderer(device=device,
                                            batch_size=1,
                                            img_wh=visualise_wh,
                                            projection_type='orthographic',
                                            render_rgb=True,
                                            bin_size=32)
    plain_texture = torch.ones(1, 1200, 800, 3, device=device).float() * 0.7
    lights_rgb_settings = {'location': torch.tensor([[0., -0.8, -2.0]], device=device, dtype=torch.float32),
                           'ambient_color': 0.5 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'diffuse_color': 0.3 * torch.ones(1, 3, device=device, dtype=torch.float32),
                           'specular_color': torch.zeros(1, 3, device=device, dtype=torch.float32)}
    fixed_cam_t = torch.tensor([[0., -0.2, 2.5]], device=device)
    fixed_orthographic_scale = torch.tensor([[0.95, 0.95]], device=device)

    hrnet_model.eval()
    pose_shape_model.eval()
    if object_detect_model is not None:
        object_detect_model.eval()
    for image_fname in tqdm(sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])):
        with torch.no_grad():
            # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------
            vis_save_path = os.path.join(save_dir, image_fname)

            image = cv2.cvtColor(cv2.imread(os.path.join(
                image_dir, image_fname)), cv2.COLOR_BGR2RGB)
            orig_image = image.copy()
            image = torch.from_numpy(image.transpose(
                2, 0, 1)).float().to(device) / 255.0
            # Predict Person Bounding Box + 2D Joints
            hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                         hrnet_config=hrnet_cfg,
                                         object_detect_model=object_detect_model,
                                         image=image,
                                         object_detect_threshold=pose_shape_cfg.DATA.BBOX_THRESHOLD,
                                         bbox_scale_factor=pose_shape_cfg.DATA.BBOX_SCALE_FACTOR)

            # Transform predicted 2D joints and image from HRNet input size to input proxy representation size
            hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                                hrnet_output['cropped_image'].shape[2]]],
                                              dtype=torch.float32,
                                              device=device) * 0.5
            hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                              dtype=torch.float32,
                                              device=device)
            cropped_for_proxy = batch_crop_pytorch_affine(input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
                                                          output_wh=(
                                                              pose_shape_cfg.DATA.PROXY_REP_SIZE, pose_shape_cfg.DATA.PROXY_REP_SIZE),
                                                          num_to_crop=1,
                                                          device=device,
                                                          joints2D=hrnet_output['joints2D'][None, :, :],
                                                          rgb=hrnet_output['cropped_image'][None, :, :, :],
                                                          bbox_centres=hrnet_input_centre,
                                                          bbox_heights=hrnet_input_height,
                                                          bbox_widths=hrnet_input_height,
                                                          orig_scale_factor=1.0)

            # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
            edge_detector_output = edge_detect_model(cropped_for_proxy['rgb'])
            proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if pose_shape_cfg.DATA.EDGE_NMS else edge_detector_output['thresholded_grad_magnitude']
            proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                             img_wh=pose_shape_cfg.DATA.PROXY_REP_SIZE,
                                                                             std=pose_shape_cfg.DATA.HEATMAP_GAUSSIAN_STD)

            torch.save(proxy_rep_img, './edge_detect_output/' +
                       image_fname.split('.')[0]+'_edge.pt')
            hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
            data = hrnet_joints2Dvisib.cpu().numpy()
            # print(data)
            count = 0
            # write a row to the csv file
            for item in data:
                if item == True:
                    count += 1
            # Only removing joints [7, 8, 9, 10, 13, 14, 15, 16] if occluded
            hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6, 11, 12]] = True
            proxy_rep_heatmaps = proxy_rep_heatmaps * \
                hrnet_joints2Dvisib[None, :, None, None]
            # (1, 18, img_wh, img_wh)
            proxy_rep_input = torch.cat(
                [proxy_rep_img, proxy_rep_heatmaps], dim=1).float()

            # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
            pred_pose_F, pred_pose_U, pred_pose_S, pred_pose_V, pred_pose_rotmats_mode, \
                pred_shape_dist, pred_glob, pred_cam_wp = pose_shape_model(
                    proxy_rep_input)
            n = 10
            samples_of_glob_rots = torch.zeros([n, 3])
            average_of_sample_rots = torch.zeros([1, 3])
            for i in range(0, n):
                f, u, s, v, pose, shape, glob, cam_wp = pose_shape_model(
                    proxy_rep_input)
                if glob.shape[-1] == 3:
                    _pred_glob_rotmats = batch_rodrigues(glob)  # (1, 3, 3)
                elif glob.shape[-1] == 6:
                    _pred_glob_rotmats = rot6d_to_rotmat(glob)
                xyz = pytorch3d.transforms.matrix_to_euler_angles(
                    _pred_glob_rotmats, 'XYZ').cpu()
                samples_of_glob_rots[i] = xyz
                average_of_sample_rots += xyz
                print(average_of_sample_rots * (1/n))

            #samples_of_glob_rots = samples_of_glob_rots * (1/n)
            print(
                "-------------------n samples of global rotation matrix-------------------")
            print(average_of_sample_rots)
            print("----------- SAMPLES---------")
            print(samples_of_glob_rots)
            # print(samples_of_glob_rots)
            # Pose F, U, V and rotmats_mode are (bsize, 23, 3, 3) and Pose S is (bsize, 23, 3)
            if pred_glob.shape[-1] == 3:
                pred_glob_rotmats = batch_rodrigues(pred_glob)  # (1, 3, 3)
            elif pred_glob.shape[-1] == 6:
                pred_glob_rotmats = rot6d_to_rotmat(pred_glob)  # (1, 3, 3)
            global_angles = pytorch3d.transforms.matrix_to_euler_angles(
                pred_glob_rotmats, 'XYZ')
            #print(samples_of_glob_rots - pred_glob_rotmats.cpu())
            pred_smpl_output_mode = smpl_model(body_pose=pred_pose_rotmats_mode,
                                               global_orient=pred_glob_rotmats.unsqueeze(
                                                   1),
                                               betas=pred_shape_dist.loc,
                                               pose2rot=False)
        # -------------------------------------------- Angles in radians ---------------------------#
        # --END--#
            pred_vertices_mode = pred_smpl_output_mode.vertices  # (1, 6890, 3)
            # Need to flip pred_vertices before projecting so that they project the right way up.
            pred_vertices_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                      axes=torch.tensor(
                                                                          [1., 0., 0.], device=device),
                                                                      angles=np.pi,
                                                                      translations=torch.zeros(3, device=device))
            # -------------------------------------------------PREDICTED  VERTICES ARE USED TO SHOW GREGY 3D model ------------------------------------------------------------------------#
            # Rotating 90° about vertical axis for visualisation
            pred_vertices_rot90_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_mode,
                                                                            axes=torch.tensor(
                                                                                [0., 1., 0.], device=device),
                                                                            angles=-np.pi / 2.,
                                                                            translations=torch.zeros(3, device=device))
            pred_vertices_rot180_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot90_mode,
                                                                             axes=torch.tensor(
                                                                                 [0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))
            pred_vertices_rot270_mode = aa_rotate_translate_points_pytorch3d(points=pred_vertices_rot180_mode,
                                                                             axes=torch.tensor(
                                                                                 [0., 1., 0.], device=device),
                                                                             angles=-np.pi / 2.,
                                                                             translations=torch.zeros(3, device=device))

            pred_reposed_smpl_output_mean = smpl_model(
                betas=pred_shape_dist.loc)
            # (1, 6890, 3)
            pred_reposed_vertices_mean = pred_reposed_smpl_output_mean.vertices
            # Need to flip pred_vertices before projecting so that they project the right way up.
            pred_reposed_vertices_flipped_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_mean,
                                                                                      axes=torch.tensor(
                                                                                          [1., 0., 0.], device=device),
                                                                                      angles=np.pi,
                                                                                      translations=torch.zeros(3, device=device))
            # Rotating 90° about vertical axis for visualisation
            pred_reposed_vertices_rot90_mean = aa_rotate_translate_points_pytorch3d(points=pred_reposed_vertices_flipped_mean,
                                                                                    axes=torch.tensor(
                                                                                        [0., 1., 0.], device=device),
                                                                                    angles=-np.pi / 2.,
                                                                                    translations=torch.zeros(3, device=device))

            # -------------------------------------- VISUALISATION --------------------------------------

            # Predicted camera corresponding to proxy rep input
            orthographic_scale = pred_cam_wp[:, [0, 0]]
            cam_t = torch.cat([pred_cam_wp[:, 1:],
                               torch.ones(pred_cam_wp.shape[0], 1, device=device).float() * 2.5],
                              dim=-1)

            # Estimate per-vertex uncertainty (variance) by sampling SMPL poses/shapes and computing corresponding vertex meshes
            per_vertex_3Dvar, pred_vertices_samples, pred_joints_samples = compute_vertex_uncertainties_by_poseMF_shapeGaussian_sampling(
                pose_U=pred_pose_U,
                pose_S=pred_pose_S,
                pose_V=pred_pose_V,
                shape_distribution=pred_shape_dist,
                glob_rotmats=pred_glob_rotmats,
                num_samples=50,
                smpl_model=smpl_model,
                use_mean_shape=True)
            #print("Predicted Shape Distance Loc:")
            # print(pred_shape_dist.loc)
            # print("#==========================================================#")
            ppf = pred_pose_rotmats_mode.cpu().numpy()
            ppfA = []
            for x in ppf:
                for rm in x:
                    # print(x)

                    rm = torch.from_numpy(rm)
                    xyz = pytorch3d.transforms.matrix_to_euler_angles(
                        rm, 'XYZ')
                    df = [math.degrees(xyz[0]), math.degrees(
                        xyz[1]), math.degrees(xyz[2])]
                    #print("JOINT ROTATION")
                    # print(df)
                    # print('==================================')
                    ppfA.append(df)

            if visualise_samples:
                num_samples = 8
                # Prepare vertex samples for visualisation
                pred_vertices_samples = joints2D_error_sorted_verts_sampling(pred_vertices_samples=pred_vertices_samples,
                                                                             pred_joints_samples=pred_joints_samples,
                                                                             input_joints2D_heatmaps=proxy_rep_input[
                                                                                 :, 1:, :, :],
                                                                             pred_cam_wp=pred_cam_wp)[:num_samples, :, :]  # (8, 6890, 3)
                # Need to flip pred_vertices before projecting so that they project the right way up.
                pred_vertices_samples = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples,
                                                                             axes=torch.tensor(
                                                                                 [1., 0., 0.], device=device),
                                                                             angles=np.pi,
                                                                             translations=torch.zeros(3, device=device))
                pred_vertices_rot90_samples = aa_rotate_translate_points_pytorch3d(points=pred_vertices_samples,
                                                                                   axes=torch.tensor(
                                                                                       [0., 1., 0.], device=device),
                                                                                   angles=-np.pi / 2.,
                                                                                   translations=torch.zeros(3, device=device))

                pred_vertices_samples = torch.cat(
                    [pred_vertices_mode, pred_vertices_samples], dim=0)  # (9, 6890, 3)
                pred_vertices_rot90_samples = torch.cat(
                    [pred_vertices_rot90_mode, pred_vertices_rot90_samples], dim=0)  # (9, 6890, 3)
            # Generate per-vertex uncertainty colourmap
            vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            vvc = vertex_var_norm(per_vertex_3Dvar.cpu().detach().numpy())
            large_num = 0
#---------------------------------Maxs uncertainty------------------------------#
            # print(vvc.count())
            # print(vvc)
            # print(vvc.count_nonzero())
            for number in vvc:
                if(number > large_num):
                    large_num = number

#---------------------------------Average uncertainty------------------------------#
            den = 0
            avg = 0
            camm = cam_t.cpu().numpy()
            for number in vvc:
                den += 1
                avg += number
#---------------------------------- Adding all the data to a single array, preparing to write to CSV file 'all_data.csv' -----------------#
            #uDat = [23]
            sDat = calculatePredS(pred_pose_S)
            DTA = [image_fname.split('.')[0], large_num, avg/den, count,
                   camm.item(0), camm.item(1), camm.item(2), str(math.degrees(global_angles[0, 0].item())), str(math.degrees(global_angles[0, 1].item())), str(math.degrees(global_angles[0, 2].item()))]
            #DTA = AppendMultipleToDTA(DTA,sDat)
            # print(ppfA)
            DTA = appendFJointsToDTA(DTA, ppfA)
            #DTA = appendREuclideanToDTA(DTA, mag_for_joints)
            DTA = AppendMultipleToDTA(DTA, hrnet_output['joints2Dconfs'])
            writeToCSV(DTA)

            vertex_var_norm = plt.Normalize(vmin=0.0, vmax=0.2, clip=True)
            vertex_var_colours = plt.cm.jet(vertex_var_norm(
                per_vertex_3Dvar.cpu().detach().numpy()))[:, :3]
            vertex_var_colours = torch.from_numpy(
                vertex_var_colours[None, :, :]).to(device).float()

            #vertex_var_colours = plt.cm.jet(vertex_var_norm(per_vertex_3Dvar.cpu().detach().numpy()))[:, :3]
            body_vis_output = body_vis_renderer(vertices=pred_vertices_mode,
                                                cam_t=cam_t,
                                                orthographic_scale=orthographic_scale,
                                                lights_rgb_settings=lights_rgb_settings,
                                                verts_features=vertex_var_colours)
            cropped_for_proxy_rgb = torch.nn.functional.interpolate(cropped_for_proxy['rgb'],
                                                                    size=(
                                                                        visualise_wh, visualise_wh),
                                                                    mode='bilinear',
                                                                    align_corners=False)
            body_vis_rgb = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                    rgb=body_vis_output['rgb_images'].permute(
                                                        0, 3, 1, 2).contiguous(),
                                                    seg=body_vis_output['iuv_images'][:, :, :, 0].round())
            body_vis_rgb = body_vis_rgb.cpu().detach().numpy()[
                0].transpose(1, 2, 0)

            body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_vertices_rot90_mode,
                                                   cam_t=fixed_cam_t,
                                                   orthographic_scale=fixed_orthographic_scale,
                                                   lights_rgb_settings=lights_rgb_settings,
                                                   verts_features=vertex_var_colours)['rgb_images'].cpu().detach().numpy()[0]
            body_vis_rgb_rot180 = body_vis_renderer(vertices=pred_vertices_rot180_mode,
                                                    cam_t=fixed_cam_t,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings,
                                                    verts_features=vertex_var_colours)['rgb_images'].cpu().detach().numpy()[0]
            body_vis_rgb_rot270 = body_vis_renderer(vertices=pred_vertices_rot270_mode,
                                                    textures=plain_texture,
                                                    cam_t=fixed_cam_t,
                                                    orthographic_scale=fixed_orthographic_scale,
                                                    lights_rgb_settings=lights_rgb_settings,
                                                    verts_features=vertex_var_colours)['rgb_images'].cpu().detach().numpy()[0]

            # Reposed body visualisation
            reposed_body_vis_rgb = body_vis_renderer(vertices=pred_reposed_vertices_flipped_mean,
                                                     textures=plain_texture,
                                                     cam_t=fixed_cam_t,
                                                     orthographic_scale=fixed_orthographic_scale,
                                                     lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]
            reposed_body_vis_rgb_rot90 = body_vis_renderer(vertices=pred_reposed_vertices_rot90_mean,
                                                           textures=plain_texture,
                                                           cam_t=fixed_cam_t,
                                                           orthographic_scale=fixed_orthographic_scale,
                                                           lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

            # Combine all visualisations
            combined_vis_rows = 2
            combined_vis_cols = 4
            combined_vis_fig = np.zeros((combined_vis_rows * visualise_wh, combined_vis_cols * visualise_wh, 3),
                                        dtype=body_vis_rgb.dtype)
            # Cropped input image
            combined_vis_fig[:visualise_wh, :visualise_wh] = cropped_for_proxy_rgb.cpu(
            ).detach().numpy()[0].transpose(1, 2, 0)

            # Proxy representation + 2D joints scatter + 2D joints confidences
            proxy_rep_input = proxy_rep_input[0].sum(
                dim=0).cpu().detach().numpy()
            proxy_rep_input = np.stack(
                [proxy_rep_input]*3, axis=-1)  # single-channel to RGB
            proxy_rep_input = cv2.resize(
                proxy_rep_input, (visualise_wh, visualise_wh))
            for joint_num in range(cropped_for_proxy['joints2D'].shape[1]):
                hor_coord = cropped_for_proxy['joints2D'][0, joint_num, 0].item(
                ) * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                ver_coord = cropped_for_proxy['joints2D'][0, joint_num, 1].item(
                ) * visualise_wh / pose_shape_cfg.DATA.PROXY_REP_SIZE
                cv2.circle(proxy_rep_input,
                           (int(hor_coord), int(ver_coord)),
                           radius=3,
                           color=(255, 0, 0),
                           thickness=-1)
                cv2.putText(proxy_rep_input,
                            str(joint_num),
                            (int(hor_coord + 4), int(ver_coord + 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
                cv2.putText(proxy_rep_input,
                            str(joint_num) + " {:.2f}".format(
                                hrnet_output['joints2Dconfs'][joint_num].item()),
                            (10, 16 * (joint_num + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), lineType=2)
            combined_vis_fig[visualise_wh:2*visualise_wh,
                             :visualise_wh] = proxy_rep_input

            # Posed 3D body
            combined_vis_fig[:visualise_wh,
                             visualise_wh:2*visualise_wh] = body_vis_rgb
            combined_vis_fig[visualise_wh:2*visualise_wh,
                             visualise_wh:2*visualise_wh] = body_vis_rgb_rot90
            combined_vis_fig[:visualise_wh, 2*visualise_wh:3 *
                             visualise_wh] = body_vis_rgb_rot180
            combined_vis_fig[visualise_wh:2*visualise_wh, 2 *
                             visualise_wh:3*visualise_wh] = body_vis_rgb_rot270

            # T-pose 3D body
            combined_vis_fig[:visualise_wh, 3*visualise_wh:4 *
                             visualise_wh] = reposed_body_vis_rgb
            combined_vis_fig[visualise_wh:2*visualise_wh, 3 *
                             visualise_wh:4*visualise_wh] = reposed_body_vis_rgb_rot90
            vis_save_path = os.path.join(save_dir, image_fname)
            cv2.imwrite(vis_save_path, combined_vis_fig[:, :, ::-1] * 255)

            if visualise_uncropped:
                # Uncropped visualisation by projecting 3D body onto original image
                rgb_to_uncrop = body_vis_output['rgb_images'].permute(
                    0, 3, 1, 2).contiguous().cpu().detach().numpy()
                iuv_to_uncrop = body_vis_output['iuv_images'].permute(
                    0, 3, 1, 2).contiguous().cpu().detach().numpy()
                bbox_centres = hrnet_output['bbox_centre'][None].cpu(
                ).detach().numpy()
                bbox_whs = torch.max(hrnet_output['bbox_height'], hrnet_output['bbox_width'])[
                    None].cpu().detach().numpy()
                bbox_whs *= pose_shape_cfg.DATA.BBOX_SCALE_FACTOR
                uncropped_for_visualise = batch_crop_opencv_affine(output_wh=(visualise_wh, visualise_wh),
                                                                   num_to_crop=1,
                                                                   rgb=rgb_to_uncrop,
                                                                   iuv=iuv_to_uncrop,
                                                                   bbox_centres=bbox_centres,
                                                                   bbox_whs=bbox_whs,
                                                                   uncrop=True,
                                                                   uncrop_wh=(orig_image.shape[1], orig_image.shape[0]))
                uncropped_rgb = uncropped_for_visualise['rgb'][0].transpose(
                    1, 2, 0) * 255
                uncropped_seg = uncropped_for_visualise['iuv'][0, 0, :, :]
                # Body pixels are > 0
                background_pixels = uncropped_seg[:, :, None] == 0
                uncropped_rgb_with_background = uncropped_rgb * (np.logical_not(background_pixels)) + \
                    orig_image * background_pixels

                uncropped_vis_save_path = os.path.splitext(vis_save_path)[
                    0] + '_uncrop.png'
                cv2.imwrite(uncropped_vis_save_path,
                            uncropped_rgb_with_background[:, :, ::-1])

            if visualise_samples:
                samples_rows = 3
                samples_cols = 6
                samples_fig = np.zeros((samples_rows * visualise_wh, samples_cols * visualise_wh, 3),
                                       dtype=body_vis_rgb.dtype)
                for i in range(num_samples + 1):
                    body_vis_output_sample = body_vis_renderer(vertices=pred_vertices_samples[[i]],
                                                               textures=plain_texture,
                                                               cam_t=cam_t,
                                                               orthographic_scale=orthographic_scale,
                                                               lights_rgb_settings=lights_rgb_settings)
                    body_vis_rgb_sample = batch_add_rgb_background(backgrounds=cropped_for_proxy_rgb,
                                                                   rgb=body_vis_output_sample['rgb_images'].permute(
                                                                       0, 3, 1, 2).contiguous(),
                                                                   seg=body_vis_output_sample['iuv_images'][:, :, :, 0].round())
                    body_vis_rgb_sample = body_vis_rgb_sample.cpu().detach().numpy()[
                        0].transpose(1, 2, 0)

                    body_vis_rgb_rot90_sample = body_vis_renderer(vertices=pred_vertices_rot90_samples[[i]],
                                                                  textures=plain_texture,
                                                                  cam_t=fixed_cam_t,
                                                                  orthographic_scale=fixed_orthographic_scale,
                                                                  lights_rgb_settings=lights_rgb_settings)['rgb_images'].cpu().detach().numpy()[0]

                    row = (2 * i) // samples_cols
                    col = (2 * i) % samples_cols
                    samples_fig[row*visualise_wh:(row+1)*visualise_wh, col*visualise_wh:(
                        col+1)*visualise_wh] = body_vis_rgb_sample

                    row = (2 * i + 1) // samples_cols
                    col = (2 * i + 1) % samples_cols
                    samples_fig[row * visualise_wh:(row + 1) * visualise_wh, col * visualise_wh:(
                        col + 1) * visualise_wh] = body_vis_rgb_rot90_sample

                    samples_fig_save_path = os.path.splitext(
                        vis_save_path)[0] + '_samples.png'
                    cv2.imwrite(samples_fig_save_path,
                                samples_fig[:, :, ::-1] * 255)


def writeToCSV(data):
    str = 'w'
    if(os.path.exists('./csv_files/all_data.csv')):
        str = 'a'
    with open('./csv_files/all_data.csv', str) as f:
        # create the csv writer
        writer = csv.writer(f)
        if(str == 'w'):
            writer.writerow(['name', 'Maximum uncertainty', 'Average Uncertainty', 'Number of Joints', 'Camera Scale', 'Camera X translation', 'Camera Y translation', 'Global X rotation', 'Global Y rotation', 'Global Z rotation', 'R0X', 'R0Y', 'R0Z', 'R1X', 'R1Y', 'R1Z', 'R2X', 'R2Y', 'R2Z', 'R3X', 'R3Y', 'R3Z', 'R4X', 'R4Y', 'R4Z', 'R5X', 'R5Y', 'R5Z', 'R6X', 'R6Y', 'R6Z', 'R7X', 'R7Y', 'R7Z', 'R8X', 'R8Y', 'R8Z', 'R9X', 'R9Y', 'R9Z', 'R10X', 'R10Y', 'R10Z', 'R11X', 'R11Y', 'R11Z', 'R12X', 'R12Y', 'R12Z',
                            'R13X', 'R13Y', 'R13Z', 'R14X', 'R14Y', 'R14Z', 'R15X', 'R15Y', 'R15Z', 'R16X', 'R16Y', 'R16Z', 'R17X', 'R17Y', 'R17Z', 'R18X', 'R18Y', 'R18Z', 'R19X', 'R19Y', 'R19Z', 'R20X', 'R20Y', 'R20Z', 'R21X', 'R21Y', 'R21Z', 'R22X', 'R22Y', 'R22Z', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16'])
        # write a row to the csv file
        writer.writerow(data)


def calculatePredS(pred_pose_S):
    sDat = []
    for x in range(0, 23):
        # u = pred_pose_U[x]
        #  v =pred_pose_V[x]
        #  f = pred_pose_F[x]
        # Calculate euclidean distance of F = USV
        #uDat[x] = math.sqrt(float(u[0,x])**2+float(u[1,x])**2+float(u[2,x])**2)
        sDat.append(math.sqrt(float(
            pred_pose_S[0, x, 0])**2+float(pred_pose_S[0, x, 1])**2+float(pred_pose_S[0, x, 2])**2))
        #  vDat[x] = math.sqrt(float(v[0,x])**2+float(v[1,x])**2+float(v[2,x])**2)
        #fDat[x] = math.sqrt(float(f[0,x])**2+float(f[1,x])**2+float(f[2,x])**2)
    return sDat


def appendREuclideanToDTA(DTA, dat):
    DTA.append(dat[0])
    DTA.append(dat[1])
    DTA.append(dat[2])
    DTA.append(dat[3])
    DTA.append(dat[4])
    DTA.append(dat[5])
    DTA.append(dat[6])
    DTA.append(dat[7])
    DTA.append(dat[8])
    DTA.append(dat[9])
    DTA.append(dat[10])
    DTA.append(dat[11])
    DTA.append(dat[12])
    DTA.append(dat[13])
    DTA.append(dat[14])
    DTA.append(dat[15])
    DTA.append(dat[16])
    DTA.append(dat[17])
    DTA.append(dat[18])
    DTA.append(dat[19])
    DTA.append(dat[20])
    DTA.append(dat[21])
    DTA.append(dat[22])
    return DTA


def appendFJointsToDTA(DTA, fDat):

    DTA.append(fDat[0][0])
    DTA.append(fDat[0][1])
    DTA.append(fDat[0][2])

    DTA.append(fDat[1][0])
    DTA.append(fDat[1][1])
    DTA.append(fDat[1][2])

    DTA.append(fDat[2][0])
    DTA.append(fDat[2][1])
    DTA.append(fDat[2][2])

    DTA.append(fDat[3][0])
    DTA.append(fDat[3][1])
    DTA.append(fDat[3][2])

    DTA.append(fDat[4][0])
    DTA.append(fDat[4][1])
    DTA.append(fDat[4][2])

    DTA.append(fDat[5][0])
    DTA.append(fDat[5][1])
    DTA.append(fDat[5][2])

    DTA.append(fDat[6][0])
    DTA.append(fDat[6][1])
    DTA.append(fDat[6][2])

    DTA.append(fDat[7][0])
    DTA.append(fDat[7][1])
    DTA.append(fDat[7][2])

    DTA.append(fDat[8][0])
    DTA.append(fDat[8][1])
    DTA.append(fDat[8][2])

    DTA.append(fDat[9][0])
    DTA.append(fDat[9][1])
    DTA.append(fDat[9][2])

    DTA.append(fDat[10][0])
    DTA.append(fDat[10][1])
    DTA.append(fDat[10][2])

    DTA.append(fDat[11][0])
    DTA.append(fDat[11][1])
    DTA.append(fDat[11][2])

    DTA.append(fDat[12][0])
    DTA.append(fDat[12][1])
    DTA.append(fDat[12][2])

    DTA.append(fDat[13][0])
    DTA.append(fDat[13][1])
    DTA.append(fDat[13][2])

    DTA.append(fDat[14][0])
    DTA.append(fDat[14][1])
    DTA.append(fDat[14][2])

    DTA.append(fDat[15][0])
    DTA.append(fDat[15][1])
    DTA.append(fDat[15][2])

    DTA.append(fDat[16][0])
    DTA.append(fDat[16][1])
    DTA.append(fDat[16][2])

    DTA.append(fDat[17][0])
    DTA.append(fDat[17][1])
    DTA.append(fDat[17][2])

    DTA.append(fDat[18][0])
    DTA.append(fDat[18][1])
    DTA.append(fDat[18][2])

    DTA.append(fDat[19][0])
    DTA.append(fDat[19][1])
    DTA.append(fDat[19][2])

    DTA.append(fDat[20][0])
    DTA.append(fDat[20][1])
    DTA.append(fDat[20][2])

    DTA.append(fDat[21][0])
    DTA.append(fDat[21][1])
    DTA.append(fDat[21][2])

    DTA.append(fDat[22][0])
    DTA.append(fDat[22][1])
    DTA.append(fDat[22][2])

    return DTA


def AppendMultipleToDTA(DTA, sDat):
    DTA.append(sDat[0].item())
    DTA.append(sDat[1].item())
    DTA.append(sDat[2].item())
    DTA.append(sDat[3].item())
    DTA.append(sDat[4].item())
    DTA.append(sDat[5].item())
    DTA.append(sDat[6].item())
    DTA.append(sDat[7].item())
    DTA.append(sDat[8].item())
    DTA.append(sDat[9].item())
    DTA.append(sDat[10].item())
    DTA.append(sDat[11].item())
    DTA.append(sDat[12].item())
    DTA.append(sDat[13].item())
    DTA.append(sDat[14].item())
    DTA.append(sDat[15].item())
    DTA.append(sDat[16].item())

    return DTA
