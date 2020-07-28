"""Examples of using pyrender for viewing and offscreen rendering.
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyglet
pyglet.options['shadow_window'] = False

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as SciRot
import h5py
import imageio
import json

from pyrender import PerspectiveCamera, IntrinsicsCamera, \
                     DirectionalLight, SpotLight, PointLight,\
                     Mesh, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

#=============================================================================
# Geometric setup
#=============================================================================
fov = np.deg2rad(6.0)

# portion_of_fov_for_this_scene = (10 / 300) / fov
portion_of_fov_for_this_scene = 1.0

# print(portion_of_fov_for_this_scene)
img_width = 4608
img_height = 3456
baseline_depth_ratio = 2.0 / 300.0
back_dist_depth_ratio = 2.0 / 300.0
xy_angle_perturb_level = 0.5       # noise level for perturbing the right camera
z_angle_perturb_level = 3

fx = img_width / 2.0 / np.tan(fov / 2.0)
fy = fx
cx = img_width / 2.0
cy = img_height / 2.0

out_dir = './data/elephant'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#==============================================================================
# Mesh creation
#==============================================================================

# trimesh
mesh = trimesh.load('./models/elephant/elephant.obj')
mesh = mesh.geometry[list(mesh.geometry.keys())[0]]
# mesh.show()

if isinstance(mesh, trimesh.Scene):
    scene_mesh = mesh
elif isinstance(mesh, trimesh.Trimesh):
    scene_mesh = trimesh.Scene(mesh)
else:
    print('Unknown type: {}'.format(type(mesh)))
    exit(-1)

# show mesh and its bounding box; press w to view in wire-frame mode
# (scene_mesh + scene_mesh.bounding_box_oriented).show()

#==============================================================================
# Scene creation
#==============================================================================
scene = Scene.from_trimesh_scene(scene_mesh, ambient_light=np.array([0.02, 0.02, 0.02]), bg_color=np.array([0, 0, 0]))
# scene = Scene.from_trimesh_scene(scene_mesh, ambient_light=np.array([0.02, 0.02, 0.02]), bg_color=np.array([79, 155, 217]))
diag_size = 1.2 * scene.scale

#==============================================================================
# compute camera poses
#==============================================================================
def compute_dist(diag_size, fov, portion_of_fov_for_this_scene=1.0):
    diag_size /= portion_of_fov_for_this_scene
    dist = diag_size / (2.0 * np.tan(fov))
    dist *= 1.1

    return dist

######### important: pyrender's 'camera pose' is the inverse of the projection matrix
#########  we are faimiliar with
######### for camera coordinate frame, opencv convention: x right, y down, z point to scene
######### pyrender use opengl convention: x right, y up, z point away from the scene

def opencv_to_opengl(R, tvec):
    # opencv to opengl: rotate along x axis by 180 degrees
    rot_mat = [[1., 0., 0.],
               [0., -1., 0.],
               [0., 0., -1]];

    R = np.dot(rot_mat, R)
    tvec = np.dot(rot_mat, tvec)

    return R, tvec

def opengl_to_opencv(R, tvec):
    # opengl to opencv: rotate along x axis by 180 degrees
    rot_mat = [[1., 0., 0.],
               [0., -1., 0.],
               [0., 0., -1]];

    R = np.dot(rot_mat, R)
    tvec = np.dot(rot_mat, tvec)

    return R, tvec

def to_pyrender_pose(R, tvec):
    proj_mat = np.eye(4)
    proj_mat[:3, :3] = R
    proj_mat[:3, 3:4] = tvec

    return np.linalg.inv(proj_mat)


def from_pyrender_pose(pose):
    proj_mat = np.linalg.inv(pose)

    R = proj_mat[:3, :3]
    tvec = proj_mat[:3, 3:4]
    return R, tvec

def compute_initial_camera_pose(scene, dist):
    centroid = scene.centroid

    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([
        [0.0, -s2, s2],
        [1.0, 0.0, 0.0],
        [0.0, s2, s2]
    ])

    displace_vec = dist * np.array([1.0, 0.0, 1.0]) * s2
    cp[:3, 3] = displace_vec + centroid # camera center

    to_centroid_dist = np.linalg.norm(displace_vec)

    # double check whether centroid-cam_center line is orthogonal to x-y plane
    R, tvec = from_pyrender_pose(cp)
    cam_center = np.dot(R.T, -tvec)
    line_vec = cam_center - centroid.reshape((3, 1))
    line_vec = np.dot(R, line_vec)
    assert(np.isclose(line_vec[0, 0], 0) and np.isclose(line_vec[1, 0], 0) and line_vec[2, 0] > 0)

    return cp, to_centroid_dist


def rotate_along_z(cam_pose, adjust):
    R, tvec = from_pyrender_pose(cam_pose)
    cam_center = np.dot(R.T, -tvec)

    R_adjust = SciRot.from_euler('zxy', np.array([adjust, 0.0, 0.0]), degrees=True).as_dcm()

    R_new = np.dot(R_adjust, R)
    tvec_new = -np.dot(R_new, cam_center)

    return to_pyrender_pose(R_new, tvec_new)

def rotate_along_y(cam_pose, adjust):
    R, tvec = from_pyrender_pose(cam_pose)
    cam_center = np.dot(R.T, -tvec)

    R_adjust = SciRot.from_euler('yzx', np.array([adjust, 0.0, 0.0]), degrees=True).as_dcm()

    R_new = np.dot(R_adjust, R)
    tvec_new = -np.dot(R_new, cam_center)

    return to_pyrender_pose(R_new, tvec_new)


def rotate_along_x(cam_pose, adjust):
    R, tvec = from_pyrender_pose(cam_pose)
    cam_center = np.dot(R.T, -tvec)

    R_adjust = SciRot.from_euler('xyz', np.array([adjust, 0.0, 0.0]), degrees=True).as_dcm()

    R_new = np.dot(R_adjust, R)
    tvec_new = -np.dot(R_new, cam_center)

    return to_pyrender_pose(R_new, tvec_new)


def move_to_up(cam_pose, distance):
    R, tvec = from_pyrender_pose(cam_pose)

    rel_R = np.eye(3)
    rel_tvec = np.array([0, -distance, 0]).reshape((3, 1))

    R_new = np.dot(rel_R, R)
    tvec_new = np.dot(rel_R, tvec) + rel_tvec

    return to_pyrender_pose(R_new, tvec_new)


def move_to_right(cam_pose, baseline):
    R, tvec = from_pyrender_pose(cam_pose)

    rel_R = np.eye(3)
    rel_tvec = np.array([-baseline, 0, 0]).reshape((3, 1))

    R_new = np.dot(rel_R, R)
    tvec_new = np.dot(rel_R, tvec) + rel_tvec

    return to_pyrender_pose(R_new, tvec_new)


def move_to_back(cam_pose, back_dist):
    R, tvec = from_pyrender_pose(cam_pose)

    rel_R = np.eye(3)
    rel_tvec = np.array([0, 0, -back_dist]).reshape((3, 1))

    R_new = np.dot(rel_R, R)
    tvec_new = np.dot(rel_R, tvec) + rel_tvec

    return to_pyrender_pose(R_new, tvec_new)

# keep camera center fixed
def perturb_rotation(cam_pose, xy_perturb_level, z_perturb_level):
    R, tvec = from_pyrender_pose(cam_pose)
    cam_center = np.dot(R.T, -tvec)

    sci_rot = SciRot.from_dcm(R)
    angles = sci_rot.as_euler('zyx', degrees=True)
    adjust = (np.random.uniform(0, 1, 3) - 0.5) * 2.0 * xy_perturb_level
    adjust[0] = (np.random.uniform(0, 1, 1) - 0.5) * 2.0 * z_perturb_level
    print('perturbing by ZYX angles (degrees): {:.4f}, {:.4f}, {:.4f}'.format(adjust[0], adjust[1], adjust[2]))

    angles += adjust
    R_new = SciRot.from_euler('zyx', angles, degrees=True).as_dcm()
    tvec_new = -np.dot(R_new, cam_center)

    return to_pyrender_pose(R_new, tvec_new)


def adjust_zyx_angles(cam_pose, adjust):
    print('perturbing by ZYX angles (degrees): {:.4f}, {:.4f}, {:.4f}'.format(adjust[0], adjust[1], adjust[2]))

    R, tvec = from_pyrender_pose(cam_pose)
    cam_center = np.dot(R.T, -tvec)

    R_adjust = SciRot.from_euler('zyx', np.array(adjust), degrees=True).as_dcm()

    R_new = np.dot(R_adjust, R)
    tvec_new = -np.dot(R_new, cam_center)

    return to_pyrender_pose(R_new, tvec_new)


def get_pyrender_cam(intrinsics, cam_pose, scene):
    R, tvec = from_pyrender_pose(cam_pose)

    import itertools
    bbx = scene.bounds
    abs_z_values = []
    for (x, y, z) in itertools.product(list(bbx[:, 0]), list(bbx[:, 1]), list(bbx[:, 2])):
        tmp = np.array([x, y, z]).reshape((3, 1))
        tmp = np.dot(R, tmp) + tvec
        abs_z_values.append(np.abs(tmp[2, 0]))

    znear = np.min(abs_z_values) * 0.9
    zfar = np.max(abs_z_values) * 1.1
    print('znear, zfar: {},{}'.format(znear, zfar))

    fx, fy, cx, cy = intrinsics
    cam = IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)

    return cam

# ground-truth poses
dist = compute_dist(diag_size, fov, portion_of_fov_for_this_scene)
cam_pose, to_centroid_dist = compute_initial_camera_pose(scene, dist)
# adjust initial pose
cam_pose = rotate_along_z(cam_pose, 90)

print('diag_size: {}, to_centroid_dist: {}'.format(diag_size, to_centroid_dist))
# exit(-1)

baseline = to_centroid_dist * baseline_depth_ratio
cam_pose_right = move_to_right(cam_pose, baseline)

back_dist = to_centroid_dist * back_dist_depth_ratio
cam_pose_back = move_to_back(cam_pose, back_dist)

#==============================================================================
# Light creation
#==============================================================================

direc_l = DirectionalLight(color=np.ones(3), intensity=25)     # parallel light
# direc_l_node = scene.add(direc_l, pose=cam_pose)

cp = np.eye(4)
s2 = 1.0 / np.sqrt(3.0)
cp[:3, :3] = np.array([
    [0.0, -s2, np.sqrt(1 - s2*s2)],
    [1.0, 0.0, 0.0],
    [0.0, np.sqrt(1 - s2*s2), s2]
])
direc_l_node = scene.add(direc_l, pose=cp)

#==============================================================================
# Using the viewer with the default view angle
#==============================================================================

# v = Viewer(scene, viewport_size=(640*2, 480*2), shadows=True)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================

# cam_node = scene.add(cam, pose=cam_pose)
# v = Viewer(scene, viewport_size=(640*2, 480*2), shadows=True)

#==============================================================================
# Rendering offscreen from the cameras
#==============================================================================

print('\nrendering left image...')
# render left image
# cam_pose = perturb_rotation(cam_pose, xy_angle_perturb_level, z_angle_perturb_level)
# cam_pose = adjust_zyx_angles(cam_pose, [-1.244, 0.316, -0.025])
cam = get_pyrender_cam((fx, fy, cx, cy), cam_pose, scene)

print('fx, fy, cx, cy: {}, {}, {}, {}'.format(fx, fy, cx, cy))
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
np.savetxt(os.path.join(out_dir, 'left_K.txt'), K, delimiter=',')
R, tvec = from_pyrender_pose(cam_pose)
R, tvec = opengl_to_opencv(R, tvec)
np.savetxt(os.path.join(out_dir, 'left_Rt.txt'), np.hstack((R, tvec)), delimiter=',')

cam_node = scene.add(cam, pose=cam_pose)
flags = RenderFlags.VERTEX_NORMALS | RenderFlags.SHADOWS_DIRECTIONAL
r = OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
color, depth = r.render(scene, flags=flags)
# color, depth = r.render(scene)
r.delete()
scene.remove_node(cam_node)     # remove camera from the scene

imageio.imwrite(os.path.join(out_dir, 'color.png'), color)

h5f = h5py.File(os.path.join(out_dir, 'depth.h5'), 'w')
h5f.create_dataset('data', data=depth)
h5f.close()

min_val, max_val = np.percentile(depth[depth>0], (1, 99))
depth_tmp = np.clip(depth, min_val, max_val)
depth_tmp = (depth_tmp - min_val) / (max_val - min_val)
imageio.imwrite(os.path.join(out_dir, 'depth.png'), np.uint8(depth_tmp*255.0))

mean_depth = np.mean(depth[depth > 0])

# write disparity map
# depth_mask = depth > 0
# disparity = fx * baseline / depth
# h5f = h5py.File(os.path.join(out_dir, 'disparity.h5'), 'w')
# h5f.create_dataset('data', data=disparity)
# h5f.close()
# min_disp = np.min(disparity[depth_mask])
# max_disp = np.max(disparity[depth_mask])
# print('max disp: {}'.format(max_disp))
# disparity = (disparity - min_disp) / (max_disp - min_disp)
# disparity[np.isinf(disparity)] = 0
# imageio.imwrite(os.path.join(out_dir, 'disparity.png'), np.uint8(disparity*255.0))

# render right image
print('\nrendering right image...')

# cam_pose_right = perturb_rotation(cam_pose_right, xy_angle_perturb_level, z_angle_perturb_level)
cam_pose_right = adjust_zyx_angles(cam_pose_right, [1.508, -0.312, -0.103])
cam = get_pyrender_cam((fx, fy, cx, cy), cam_pose_right, scene)

print('fx, fy, cx, cy: {}, {}, {}, {}'.format(fx, fy, cx, cy))
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
np.savetxt(os.path.join(out_dir, 'right_K.txt'), K, delimiter=',')
R, tvec = from_pyrender_pose(cam_pose_right)
R, tvec = opengl_to_opencv(R, tvec)
np.savetxt(os.path.join(out_dir, 'right_Rt.txt'), np.hstack((R, tvec)), delimiter=',')

cam_node = scene.add(cam, pose=cam_pose_right)
r = OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
color, depth = r.render(scene, flags=flags)
# color, depth = r.render(scene)
r.delete()
scene.remove_node(cam_node)

imageio.imwrite(os.path.join(out_dir, 'color_right.png'), color)

# render back image
print('\nrendering back image...')

# cam_pose_back = perturb_rotation(cam_pose_back, xy_angle_perturb_level, z_angle_perturb_level)
cam_pose_back = adjust_zyx_angles(cam_pose_back, [1.065, -0.447, -0.280])
cam = get_pyrender_cam((fx, fy, cx, cy), cam_pose_back, scene)

print('fx, fy, cx, cy: {}, {}, {}, {}'.format(fx, fy, cx, cy))
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
np.savetxt(os.path.join(out_dir, 'back_K.txt'), K, delimiter=',')
R, tvec = from_pyrender_pose(cam_pose_back)
R, tvec = opengl_to_opencv(R, tvec)
np.savetxt(os.path.join(out_dir, 'back_Rt.txt'), np.hstack((R, tvec)), delimiter=',')

cam_node = scene.add(cam, pose=cam_pose_back)
r = OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
color, depth = r.render(scene, flags=flags)
# color, depth = r.render(scene)
r.delete()
scene.remove_node(cam_node)

imageio.imwrite(os.path.join(out_dir, 'color_back.png'), color)

#================================================================
# write some meta information
#================================================================
info_dict = dict([('baseline', float(baseline)),
                  ('back_dist', float(back_dist)),
                  ('mean_depth', float(mean_depth)),
                  ('fov', np.rad2deg(fov)), 
                  ('fx', float(fx)),
                  ('fy', float(fy)),
                  ('diag_size', float(diag_size)),
                  ('to_centroid_dist', float(to_centroid_dist))])

with open(os.path.join(out_dir, 'info_dict.json'), 'w') as fp:
    json.dump(info_dict, fp, indent=2)
