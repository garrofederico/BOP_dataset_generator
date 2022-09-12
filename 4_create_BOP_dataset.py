import os
import sys

sys.path.append(os.path.abspath('.'))
import random
import json
from utils import load_bop_meshes, draw_pose_axis
import shutil
import csv
import math
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import trimesh
import yaml
import torch


# choose a backend for pyrender, check: https://pyrender.readthedocs.io/en/latest/examples/offscreen.html
# os.environ['PYOPENGL_PLATFORM'] = 'egl'





def convert_list_to_bop_format(images_poses_list, outdir, cfg):
    # Configs:
    mesh_path = cfg["MESH_PATH"]
    n_keypoints = cfg["N_KEYPOINTS"]
    mesh = np.array(json.load(open(mesh_path))["MESH"], dtype=np.float32)
    mesh = np.reshape(mesh, (n_keypoints, 3, 1))
    K = np.array(json.load(open(os.path.abspath(cfg["INTRINSICS_PATH"])))['K'], dtype=np.float64)
    dist = np.array(json.load(open(os.path.abspath(cfg["INTRINSICS_PATH"])))['dist'], dtype=np.float32)

    outdir = outdir + '000000/'
    out_maskvisib_dir = outdir + 'mask_visib/'
    out_rgb_dir = outdir + 'rgb/'
    #
    os.makedirs(out_maskvisib_dir, exist_ok=True)
    os.makedirs(out_rgb_dir, exist_ok=True)
    #
    out_gt_file = outdir + 'scene_gt.json'
    out_gt_info_file = outdir + 'scene_gt_info.json'
    out_gt_camera_file = outdir + 'scene_camera.json'
    #
    gt_dict = {}
    gt_info_dict = {}
    gt_camera_dict = {}
    #
    # loop for processing the batch of photos
    outIdx = 1
    #
    a = os.path.splitext(outdir)
    text_name = os.path.dirname(os.path.dirname(outdir))
    text_fp = os.path.join(os.path.dirname(text_name), os.path.basename(text_name) + '.txt')


    with open(text_fp, 'w') as w:
        for fpath, pose_in_image in images_poses_list:
            R = pose_in_image[0]
            T = pose_in_image[1]
            print(fpath)
            cvImg = cv2.imread(fpath)

            height, width, _ = cvImg.shape
            #
            dirPath, fname = fpath.rsplit('/', 1)
            ext = os.path.splitext(fpath)[1]
            baseName = ("%06d" % outIdx)
            #
            if not cfg["DEBUG"]:
                # txt file
                shutil.copyfile(fpath, out_rgb_dir + baseName + ext)
                fp = os.path.normpath(out_rgb_dir + baseName + ext)
                fp_list = fp.split(os.sep)

                image_fp = os.path.join(fp_list[-4], fp_list[-3], fp_list[-2], fp_list[-1])
                w.write(image_fp + '\n')
            # solamente para comparar lo de abajo
            # K3 = get_camera_K(width, height, 100)

            # retrieve points from the XML annotations file
            idx = int(fname[0:5]) - 1  # use filename as index
            # if True:
            #     points_annotated = []
            #     t = root[2 + idx:][0]
            #     for p in t[:4]:
            #         x, y = p.attrib['points'].split(',')
            #         x = float(x)
            #         y = float(y)
            #         points_annotated.append([x, y])
            #
            #     # TODO: Crear R y T a partir de los puntos
            #     points_annotated = np.array(points_annotated, dtype=np.float32)
            #     points_annotated = np.reshape(points_annotated, (4, 2, 1))
            #     _, rvec, tvec, _ = cv2.solvePnPRansac(mesh[:n_keypoints], points_annotated, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            #     R = cv2.Rodrigues(rvec)[0]
            #     T = tvec  # (parece)

            # TODO: crear la mascara aca (bbox a partir de los puntos)
            points_annotated = cv2.projectPoints(mesh, R, T, K, dist)[0]
            points_annotated = np.array(points_annotated, dtype=np.int32)
            points_annotated = np.squeeze(points_annotated)

            # create bounding box mask
            square_mask = np.zeros((height, width), dtype=np.uint8)
            mesh_mask = [np.min(points_annotated[:, 0]), np.min(points_annotated[:, 1]),
                         np.max(points_annotated[:, 0]), np.max(points_annotated[:, 1])]
            mesh_mask = np.array(mesh_mask, dtype=np.int32)
            square_mask[mesh_mask[1]:mesh_mask[3], mesh_mask[0]:mesh_mask[2]] = 255
            # print(currentpose)
            # print(K)
            # rImg, depth = render_objects([mesh], [0], [currentpose], K, width, height)
            #
            # print(K)
            # print(R)
            # print(T)

            # create bounding object mask
            assert cfg["OBJECT"] in {'repere', 'haltere'}, "OBJECT in config file has to be 'repere' or 'haltere' "
            shape_mask = np.zeros((height, width), dtype=np.uint8)
            if cfg["OBJECT"] == "repere":
                for pIdx in range(len(points_annotated[:, 0])):
                    shape_mask = cv2.circle(shape_mask, (int(points_annotated[pIdx, 0]), int(points_annotated[pIdx, 1])), 1, 255, -1)

                shape_mask = cv2.line(shape_mask, (points_annotated[0, 0], points_annotated[0, 1]), (points_annotated[3, 0], points_annotated[3, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)
                shape_mask = cv2.line(shape_mask, (points_annotated[1, 0], points_annotated[1, 1]), (points_annotated[3, 0], points_annotated[3, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)
                shape_mask = cv2.line(shape_mask, (points_annotated[2, 0], points_annotated[2, 1]), (points_annotated[3, 0], points_annotated[3, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)
            elif cfg["OBJECT"] == "haltere":
                shape_mask = cv2.circle(shape_mask, (int(points_annotated[1, 0]), int(points_annotated[1, 1])), 3 * cfg["MASK_LINE_THICKNESS"], 255, -1)
                shape_mask = cv2.circle(shape_mask, (int(points_annotated[3, 0]), int(points_annotated[3, 1])), 3 * cfg["MASK_LINE_THICKNESS"], 255, -1)

                shape_mask = cv2.line(shape_mask, (points_annotated[0, 0], points_annotated[0, 1]), (points_annotated[4, 0], points_annotated[4, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)
                shape_mask = cv2.line(shape_mask, (points_annotated[1, 0], points_annotated[1, 1]), (points_annotated[4, 0], points_annotated[4, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)
                shape_mask = cv2.line(shape_mask, (points_annotated[2, 0], points_annotated[2, 1]), (points_annotated[4, 0], points_annotated[4, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)
                shape_mask = cv2.line(shape_mask, (points_annotated[3, 0], points_annotated[3, 1]), (points_annotated[4, 0], points_annotated[4, 1]), 255,
                                      thickness=cfg["MASK_LINE_THICKNESS"], lineType=cv2.LINE_AA)

            if cfg["DEBUG"]:
                cv2.imshow("image", cvImg)
                # cv2.imshow("render", rImg)
                cv2.imshow("mask_render", square_mask)
                cv2.imshow("mask_reproj", shape_mask)
                cv2.waitKey(0)
            #
            outmaskName = baseName + "_000000.png"
            if not cfg["DEBUG"]:
                cv2.imwrite(out_maskvisib_dir + outmaskName, shape_mask)
            #
            pose = {}
            pose['cam_R_m2c'] = list(R.reshape(-1))
            pose['cam_t_m2c'] = list(T.reshape(-1))
            pose['obj_id'] = 1
            gt_dict[str(outIdx)] = [pose]  # only one object
            #
            cam = {}
            cam['cam_K'] = list(K.reshape(-1))
            cam['depth_scale'] = 0
            gt_camera_dict[str(outIdx)] = cam
            #
            info = {}
            ys, xs = np.where(square_mask == 255)
            xmin = xmax = ymin = ymax = 0
            if len(ys) > 0:
                xmin = int(xs.min())
                xmax = int(xs.max())
                ymin = int(ys.min())
                ymax = int(ys.max())
            info['bbox_visib'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            gt_info_dict[str(outIdx)] = [info]
            #
            outIdx += 1

    if not cfg["DEBUG"]:
        with open(out_gt_file, 'w') as outfile:
            json.dump(gt_dict, outfile, indent=2)
        with open(out_gt_camera_file, 'w') as outfile:
            json.dump(gt_camera_dict, outfile, indent=2)
        with open(out_gt_info_file, 'w') as outfile:
            json.dump(gt_info_dict, outfile, indent=2)

    return


def get_poses_from_annotations(annotations, mesh, n_keypoints, K, dist):
    # Retrieve annotated points from Cvat xml file
    # Returns "format"
    tree = ET.parse(annotations)
    root = tree.getroot()
    poses_from_annotations = []
    for c in root[2:]:
        points = []
        for point in c[:n_keypoints]:
            x, y = point.attrib['points'].split(',')
            x = float(x)
            y = float(y)
            points.append([x, y])
        # TODO: read "ROTATED_ANNOTATIONS" from cfg file
        #if cfg["SETUP"]["ROTATED_ANNOTATIONS"]:
        #points = [points[1], points[2], points[0], points[3]]
        points = np.array(points, dtype=np.float32)
        points = np.reshape(points, (n_keypoints, 2, 1))
        _, rvec, tvec, _ = cv2.solvePnPRansac(mesh[:n_keypoints], points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rvec)
        poses_from_annotations.append((rmat, tvec))
    return poses_from_annotations

def convert_to_bop_format(cfg):

    out_path = cfg["OUTPUT_PATH"]
    images_poses_list = join_poses_and_images_list(cfg)

    if cfg["FOR_INFERENCE"]:
        # images_poses_list.sort()
        test_list = images_poses_list
        convert_list_to_bop_format(test_list, out_path + 'testing/', cfg)
    else:
        # TODO: Make splits from cfg
        random.seed(42)
        random.shuffle(images_poses_list)
        trainSpl = int(len(images_poses_list) * 0.8 + 0.5)
        testSpl = int(len(images_poses_list) * 0.9 + 0.5)
        train_list = images_poses_list[:trainSpl]
        test_list = images_poses_list[trainSpl:testSpl]
        val_list = images_poses_list[testSpl:]

        convert_list_to_bop_format(test_list, out_path + 'testing/', cfg)
        convert_list_to_bop_format(train_list, out_path + 'training/', cfg)
        convert_list_to_bop_format(val_list, out_path + 'validation/', cfg)

    # # train_list = imglist
    # trainSpl = int(len(imglist) * 0.8 + 0.5)
    # testSpl = int(len(imglist) * 0.9 + 0.5)
    #
    # # train_list = imglist[:trainSpl]
    # # test_list = imglist[trainSpl:testSpl]
    # # val_list = imglist[testSpl:]
    #
    # train_list = imglist[1800:1900]
    # test_list = imglist
    # val_list = imglist[1700:1800]

    # train_list.sort()
    # test_list.sort()
    # val_list.sort()
    #
    # convert_list_to_bop_format(test_list, out_path + 'testing/', mesh, keypoints, cfg)
    # convert_list_to_bop_format(train_list, out_path + 'training/', mesh, keypoints, cfg)
    # convert_list_to_bop_format(val_list, out_path + 'validation/', mesh, keypoints, cfg)


def join_poses_and_images_list(cfg):
    # beginning of function
    n_keypoints = cfg["N_KEYPOINTS"]
    mesh_path = cfg["MESH_PATH"]
    mesh = np.array(json.load(open(mesh_path))["MESH"], dtype=np.float32)
    mesh = np.reshape(mesh, (n_keypoints, 3, 1))
    K = np.array(json.load(open(os.path.abspath(cfg["INTRINSICS_PATH"])))['K'], dtype=np.float64)
    dist = np.array(json.load(open(os.path.abspath(cfg["INTRINSICS_PATH"])))['dist'], dtype=np.float32)

    images_poses_list = []
    if cfg["FOR_INFERENCE"]:
        img_list = [cfg["INPUT_PATH"] + f
                    for f in os.listdir(cfg["INPUT_PATH"])
                    if f.endswith('.png') or f.endswith('.jpg')
                    ]
        img_list.sort()
        poses_list = [(np.zeros((3,3)), np.zeros(3))
                      for i in range(len(img_list))]
        images_poses_list = list(zip(img_list, poses_list))
        return images_poses_list
    for seqIdx, (input_image_set, annotations) in enumerate(zip(cfg["INPUT_PATH"], cfg["ANNOTATIONS_PATH"])):
        # get list of poses
        poses_list = get_poses_from_annotations(annotations, mesh, 4, K, dist) # TODO: replace N of keypoints with 4
        # get list of images
        img_list = [input_image_set + f
                    for f in os.listdir(input_image_set)
                    if f.endswith('.png') or f.endswith('.jpg')
                    ]
        img_list.sort()
        # join two lists
        images_and_poses = list(zip(img_list, poses_list))
        images_poses_list.extend(images_and_poses)
    return images_poses_list


def generate_3d_bounding_box(cfg):
    meshes, objID_2_clsID = load_bop_meshes(cfg["MODEL_PATH"])
    bbox3ds = []
    for mf in meshes:
        ms = trimesh.load(mf)
        bb = ms.bounding_box_oriented.vertices
        # bb = ms.bounding_box.vertices
        bbox3ds.append(bb.tolist())
    with open(cfg["OUTPUT_PATH"] + 'bbox.json', 'w') as outfile:
        json.dump(bbox3ds, outfile, indent=4)


def calculate_mesh_diameter_and_write(cfg):
    objects = [f for f in os.listdir(cfg["MODEL_PATH"]) if f.endswith('.ply')]
    objects.sort()
    meshes = []
    for objname in objects:
        mp = cfg["MODEL_PATH"] + objname
        mesh = trimesh.load(mp)
        meshes.append(mesh)
        # print(mp + '...')
        vol = mesh.bounding_sphere.volume
        diameter = ((vol * 3 / (4 * np.pi)) ** (1 / 3)) * 2
        with open(cfg["OUTPUT_PATH"] + "readme.txt", 'w') as w:
            w.write("Config file parameters used for the creation of the dataset: \n\n")
            for key, value in cfg.items():
                w.write(f"{key}: {value} \n")
            w.write(f"\nMesh diameter (to input in the .yml config file):{diameter:.2f}")
        print("Mesh diameter (to input in the .yml config file):", "%.2f" % diameter)


def copy_models_to_dataset(cfg):
    # make "models" folder
    os.makedirs(cfg["OUTPUT_PATH"] + "models", exist_ok=True)
    # load meshes
    meshFiles = [f for f in os.listdir(cfg["MODEL_PATH"]) if f.endswith('.ply')]
    meshFiles.sort()
    for mFile in meshFiles:
        shutil.copy(cfg["MODEL_PATH"] + '/' + mFile, cfg["OUTPUT_PATH"] + "models/")


def get_config_file(config_file):
    assert os.path.exists(config_file)
    with open(config_file, "r") as stream:
        try:
            file = yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            print(exc)
    file_name = os.path.basename(config_file)
    config_name = os.path.splitext(file_name)[0] + '/'
    file["OUTPUT_PATH"] += config_name

    return file


def main():
    config_file = 'CONFIGS/BOP_dataset_haltere_dataset_2_vids.yaml'

    cfg = get_config_file(config_file)

    copy_models_to_dataset(cfg)

    generate_3d_bounding_box(cfg)

    convert_to_bop_format(cfg)
    # for seqIdx, (input_image_set, annotations) in enumerate(zip(cfg["INPUT_PATH"], cfg["ANNOTATIONS_PATH"])):
    #     print(seqIdx, input_image_set, annotations)

    calculate_mesh_diameter_and_write(cfg)

    # dir_path = "/Users/federico/PycharmProjects/physic_overlay_codebase/DATA/6_2/GH010483/"
    # dirs = os.listdir(dir_path)
    # dirs.sort()

    # TODO: usar esta parte para crear dataset con varias carpetas
    # zip(dir_list, annotation_list) crear lista en yaml - - -
    seqIdx = 0
    # seqIdx = 350
    # seqIdx = 400
    # for dd in dirs:
    #     iPath = dir_path + dd + '/'
    #     oPath = out_path + ("seq_%06d" % seqIdx) + '/'
    #     seqIdx += 1
    #     if os.path.isdir(iPath):
    #         convert_to_bop_format(iPath, oPath, model_path, None)


if __name__ == "__main__":
    main()
