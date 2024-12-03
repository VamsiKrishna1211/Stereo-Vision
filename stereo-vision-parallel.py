# %%
# %env EGL_PLATFORM surfaceless

import os
# os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "true"
# os.environ["WEBRTC_IP"] = "127.0.0.1"
# os.environ["WEBRTC_PORT"] = "8889"

import stereo
from stereo.utils import common_utils
from easydict import EasyDict
from stereo.modeling import models
from stereo.datasets.dataset_template import build_transform_by_cfg
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
# import torch_tensorrt

import cv2
import numpy as np

import open3d as o3d
# o3d.visualization.webrtc_server.enable_webrtc()
from open3d.visualization import Visualizer

import asyncio
import nest_asyncio
# nest_asyncio.apply()


device = 'cpu' if not torch.backends.mps.is_available() else 'mps'
print(device)

focal_length_left_h = 1404.9707644369653
focal_length_left_v = 1403.9142834191978
cx_R = 959.6977300305203  
cy_R = 541.5754355100794
camera_matrix_left = np.array([[focal_length_left_h, 0, cx_R], [0, focal_length_left_v, cy_R], [0, 0, 1]])
# 0.053564863912631905 0.05216935306303675 -0.00025046500414719426 -0.0007604197477040687 -0.20467060036288837
dist_coeffs_left = np.array([0.053564863912631905, 0.05216935306303675, -0.00025046500414719426, -0.0007604197477040687, -0.20467060036288837])
newcameramtx_left, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_left, dist_coeffs_left, (1920,1080), 1, (1920,1080))

focal_length_right_h = 1403.625781226055
focal_length_right_v = 1403.6775759719974
cx_L = 970.6823606246131  
cy_L = 537.652774517532 

camera_matrix_right = np.array([[focal_length_right_h, 0, cx_L], [0, focal_length_right_v, cy_L], [0, 0, 1]])
# 0.042743552637767517 0.003474288976375031 -0.00032838506356857537 -4.26553021947547e-05 -0.10961391062739367
dist_coeffs_right = np.array([0.042743552637767517, 0.003474288976375031, -0.00032838506356857537, -4.26553021947547e-05, -0.10961391062739367])
newcameramtx_right, roi = cv2.getOptimalNewCameraMatrix(camera_matrix_right, dist_coeffs_right, (1920,1080), 1, (1920,1080))

# Rotation and Translation Matrices
R = np.array([
    [0.9997389194630696, 0.020656440024216148, 0.009767517409253334],
    [-0.020840842660613458, 0.9995990020830661, 0.019170141150603396],
    [-0.009367613784185385, -0.01936869949336983, 0.9997685238553602]
])

T = np.array([-10.065795209719223, 0.0137315547971429, -0.2450403146577965])

# baseline = 0.193 # mm meters
# baseline = baseline/1000
baseline = 10.1
# baseline = 0.07162429074481386


# %%
def create_model(cfg_file, ):
    yaml_config = common_utils.config_loader(cfg_file)
    cfgs = EasyDict(yaml_config)
    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING
    
    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING


    transform = build_transform_by_cfg(transform_config)
    model = models.lightstereo.lightstereo.LightStereo(cfgs.MODEL)
    model_weights = torch.load("/Users/vamsikrishna/Data/projects/Robot-Vision-Project/stereo-depth-mapping/models/LightStereo-S-SceneFlow.ckpt", map_location=device)

    model.load_state_dict(model_weights['model_state'])


    # model = models.psmnet.psmnet.PSMNet(cfgs.MODEL)

    model = torch.compile(model, backend="cudagraphs")
    model.to(device)

    return model, transform, cfgs

# %%
def get_depth_map(cfgs, model: torch.nn.Module, transform, left_image, right_image): # All RGB Images.
    # left_image = Image.open(left_image)
    # right_image = Image.open(right_image)

    # left_img = np.array(Image.open(left_image_path).convert('RGB'), dtype=np.float32)
    # right_img = np.array(Image.open(right_image_path).convert('RGB'), dtype=np.float32)
    sample = {
    'left': left_image,
    'right': right_image
    }

    sample = transform(sample)
    sample['left'] = sample['left'].unsqueeze(0)
    sample['right'] = sample['right'].unsqueeze(0)

    for k, v in sample.items():
        sample[k] = v.to(device) if torch.is_tensor(v) else v
        print(sample[k].device)

    # left_image_out = transform(left_image)
    # right_image_out = transform(right_image)

    # left_image = left_image.to(device)
    # right_image = right_image.to(device)

    print("Model Running")
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, enabled=cfgs.OPTIMIZATION.AMP):
            torch.compiler.cudagraph_mark_step_begin()
            start = time.time()
            model_pred = model(sample)
            disp_pred = model_pred['disp_pred'].squeeze(1)
            print("Model run done, but extracting data")

            disparity = disp_pred.cpu().squeeze().numpy()

            print("Time taken: ", time.time() - start)
    # print("Model Run Done")

        # GY Camera



    # camera_intrinsic_L = o3d.camera.PinholeCameraIntrinsic(
    # width=left_image.shape[0],  # Replace with your image width
    # height=left_image.shape[1], # Replace with your image height
    # fx=focal_length_gy_h,   # Replace with your camera's focal length in x
    # fy=focal_length_gy_v,   # Replace with your camera's focal length in y
    # cx=cx_L,   # Replace with your camera's principal point in x
    # cy=cy_L    # Replace with your camera's principal point in y
    # )

    # window_size = 4
    # min_disp = 0
    # ndisp = 350
    # max_disp = ndisp

    # num_disp = ((max_disp - min_disp)//16)*16

    # start = time.time()

    # left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
    # right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

    # stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    #                              numDisparities = num_disp,
    #                                 blockSize = 11,
    #                                 P1 = 8*3*window_size**2,
    #                                 P2 = 64*3*window_size**2,
    #                                 disp12MaxDiff = 1,
    #                                 uniquenessRatio = 15,
    #                                 speckleWindowSize = 100,
    #                                 speckleRange = 32,
    #                                 preFilterCap = 10,
    #                                 mode = cv2.StereoSGBM_MODE_SGBM_3WAY
    #                            )
    # disparity = stereo.compute(left_image, right_image).astype(np.float32)
    # disparity = (disparity/16 - min_disp) / max_disp
    # end = time.time()
    # print("Time taken for disparity: ", end - start)
    # disparity = np.clip(disparity, np.finfo(float).eps, 255)

    # camera_intrinsic_R = o3d.camera.PinholeCameraIntrinsic(
    #     width=right_image.shape[0],  # Replace with your image width
    #     height=right_image.shape[1], # Replace with your image height
    #     fx=focal_length_gg_h,   # Replace with your camera's focal length in x
    #     fy=focal_length_gg_v,   # Replace with your camera's focal length in y
    #     cx=cx_R,   # Replace with your camera's principal point in x
    #     cy=cy_R    # Replace with your camera's principal point in y
    # )

    plt.imshow(disparity, cmap='grey')
    plt.show()
    depth = (focal_length_left_v * baseline) / (disparity)

    print("returning depth")

    return depth

# %%
# async def process_and_display_streams(left_frame_queue, right_frame_queue):
#     model, transform, cfgs = create_model("/home/vamsik1211/Data/personal-projects/Robot-Vision-Project/stereo-depth-mapping/OpenStereo/cfgs/lightstereo/lightstereo_lx_sceneflow.yaml")
#     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
#     pcd = o3d.geometry.PointCloud()
    
#     vis = Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)


#     while True:
#         if not left_frame_queue.empty() and not right_frame_queue.empty():
#             left_frame = left_frame_queue.get()
#             right_frame = right_frame_queue.get()

#             left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
#             right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

#             depth_map, camera_intrinsic_mat = await get_depth_map(cfgs, model, transform, left_frame, right_frame)

#             o3d_img = o3d.geometry.Image(left_frame.astype(np.uint8))
#             o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))
#             rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img, o3d_depth, convert_rgb_to_intensity=False)
#             pcd_temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic_mat)
#             pcd.points = pcd_temp.points

#             # o3d.visualization.draw_geometries([pcd])
#             vis.update_geometry(pcd)
#             vis.poll_events()
#             vis.update_renderer()

#             # plt.imshow(depth_map)
#             # plt

#     # depth_map = np.array(depth_map

# async def process_and_display_streams(left_frame_queue, right_frame_queue):
#     model, transform, cfgs = create_model("/home/vamsik1211/Data/personal-projects/Robot-Vision-Project/stereo-depth-mapping/OpenStereo/cfgs/lightstereo/lightstereo_lx_sceneflow.yaml")
#     o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
#     pcd = o3d.geometry.PointCloud()
    
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window()
#     vis.add_geometry(pcd)

#     def update_geometry(vis):
#         print("Waiting for frames")
#         if not left_frame_queue.empty() and not right_frame_queue.empty():
#             print("Reading frames")

#             left_frame = left_frame_queue.get()
#             right_frame = right_frame_queue.get()

#             left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
#             right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

#             print("Got frames")
#             print("Starting inference")
#             depth_map, camera_intrinsic_mat = asyncio.run(get_depth_map(cfgs, model, transform, left_frame, right_frame))

#             o3d_img = o3d.geometry.Image(left_frame.astype(np.uint8))
#             o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))
#             rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img, o3d_depth, convert_rgb_to_intensity=False)
#             pcd_temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic_mat)
#             pcd.points = pcd_temp.points

#             vis.update_geometry(pcd)
#             vis.poll_events()
#             vis.update_renderer()

#     vis.register_key_callback(ord(" "), update_geometry)

#     while True:
#         # print("Looping....")
#         vis.poll_events()
#         vis.update_renderer()
#         await asyncio.sleep(0.01)


def process_and_display_streams(rtsp_url_1, rtsp_url_2):
    model, transform, cfgs = create_model("/Users/vamsikrishna/Data/projects/Robot-Vision-Project/stereo-depth-mapping/OpenStereo/cfgs/lightstereo/lightstereo_s_sceneflow.yaml")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    
    # pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)

    cap_1 = cv2.VideoCapture(rtsp_url_1)
    cap_2 = cv2.VideoCapture(rtsp_url_2)

    # Stereo Rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        (1920, 1080), R, T,
        alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, R1, P1,
        (1920, 1080), cv2.CV_32FC1
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, R2, P2,
        (1920, 1080), cv2.CV_32FC1
    )

    while True:
        ret_1, left_frame = cap_1.read()
        ret_2, right_frame = cap_2.read()

        # ret_1, left_frame = cap_2.read()
        # ret_2, right_frame = cap_1.read()

        if not ret_1 or not ret_2:
            print(f"No frame on one or both streams")
            break

        # left_frame = cv2.undistort(left_frame, camera_matrix_left, dist_coeffs_left)
        # left_frame = cv2.rotate(left_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # left_frame = cv2.flip(left_frame, 1)

        # right_frame = cv2.undistort(right_frame, camera_matrix_right, dist_coeffs_right)
        # right_frame = cv2.rotate(right_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # right_frame = cv2.flip(right_frame, 1)

        left_frame = cv2.remap(left_frame, map1x, map1y, cv2.INTER_LINEAR)
        right_frame = cv2.remap(right_frame, map2x, map2y, cv2.INTER_LINEAR)


        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("Left Frame", left_frame)
        cv2.imshow("Right Frame", right_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # plt.imshow(left_frame)
        # plt.show()

        # plt.imshow(right_frame)
        # plt.show()

        depth_map = get_depth_map(cfgs, model, transform, left_frame, right_frame)
        camera_intrinsic_L = o3d.camera.PinholeCameraIntrinsic(
            width=depth_map.shape[0],  
            height=depth_map.shape[1],
            fx=focal_length_left_v,   
            fy=focal_length_left_h,  
            cx=cy_L,   
            cy=cx_L    
        )

        o3d_img = o3d.geometry.Image(left_frame.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img, o3d_depth, convert_rgb_to_intensity=False)

        print("Creating Point Cloud")
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic_L)
        o3d.visualization.draw_geometries([pcd])
        # pcd.non_blocking = True
        # pcd.points = pcd_temp.points 

        # vis.update_geometry(pcd)
        # vis.poll_events()
        # vis.update_renderer()

        # ... (optional: display the frames using cv2.imshow)

    cap_1.release()
    cap_2.release()

if __name__ == "__main__":
    # rtsp_url_2 = 'rtsp://172.16.33.186:8554/cam'
    # rtsp_url_1 = 'rtsp://172.16.33.188:8554/cam'
    rtsp_url_1 = 'rtsp://172.16.33.186:8554/cam'
    rtsp_url_2 = 'rtsp://172.16.33.188:8554/cam'
    process_and_display_streams(rtsp_url_1, rtsp_url_2)

# %%
# async def capture_stream(rtsp_url, frame_queue):
#     cap = cv2.VideoCapture(rtsp_url)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"No Frame on {rtsp_url}")
#             break
#         # print(f"read frame on {rtsp_url}")

#         frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         frame = cv2.flip(frame, 1)

#         cv2.imshow(f'VIDEO {rtsp_url}', frame)
#         cv2.waitKey(1)

#         await frame_queue.put(frame)
#         await asyncio.sleep(0.0001)  # Small sleep to simulate async behavior
#     cap.release()

# async def main():
#     rtsp_url_1 = 'rtsp://172.16.33.154:8554/cam'
#     rtsp_url_2 = 'rtsp://172.16.33.188:8554/cam'
    
#     frame_queue_1 = asyncio.Queue()
#     frame_queue_2 = asyncio.Queue()
    
#     task1 = asyncio.create_task(capture_stream(rtsp_url_1, frame_queue_1))
#     task2 = asyncio.create_task(capture_stream(rtsp_url_2, frame_queue_2))
#     task3 = asyncio.create_task(process_and_display_streams(frame_queue_1, frame_queue_2))
    
#     await asyncio.gather(task1, task2, task3)

# To run the asyncio event loop

# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
# pcd = o3d.geometry.PointCloud()

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)

# asyncio.run(main())

# %%
# import cv2

# vcap = cv2.VideoCapture("rtsp://172.16.33.188:8554/cam")
# while(1):
#     ret, frame = vcap.read()
#     if not ret:
#         print("Error reading frame")
#         break
#     print(frame.shape)
#     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#     cv2.imshow('VIDEO', frame)
#     cv2.waitKey(1)

# %%


# %%



