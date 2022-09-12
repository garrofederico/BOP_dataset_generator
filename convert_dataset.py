import os
import sys
sys.path.append(os.path.abspath('.'))
import random
import json
from utils import *
import shutil
import csv
import math
#import transforms3d
#import bs4 as bs

# choose a backend for pyrender, check: https://pyrender.readthedocs.io/en/latest/examples/offscreen.html
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_camera_K(width, height, fov):
    fov = fov * math.pi / 180
    # diagonal
    if True:
        fov = 2*math.atan(math.tan(fov/2)/math.sqrt(2))
    fx = width / (2 * math.tan(fov/2))
    fy = height / (2 * math.tan(fov/2))
    u = width / 2
    v = height / 2
    return np.array([fx,0,u,0,fy,v,0,0,1]).reshape(3,3)

def lookAt(eye, target, up):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -1 * np.dot(R, eye)
    # 
    camToWorld = np.concatenate((R, tvec.reshape(-1, 1)), axis=-1)
    camToWorld = np.concatenate((camToWorld, np.array([0,0,0,1]).reshape(1,4)), axis=0)
    return camToWorld

def convert_list_to_bop_format(filelist, outdir, mesh, keypoints):
    # 
    # debug = True
    debug = False
    # 
    outdir = outdir + '000000/'
    out_maskvisib_dir = outdir + 'mask_visib/'
    out_rgb_dir = outdir + 'rgb/'
    # 
    os.makedirs(out_maskvisib_dir, exist_ok = True)
    os.makedirs(out_rgb_dir, exist_ok = True)
    # 
    out_gt_file = outdir + 'scene_gt.json'
    out_gt_info_file = outdir + 'scene_gt_info.json'
    out_gt_camera_file = outdir + 'scene_camera.json'
    # 
    gt_dict = {}
    gt_info_dict = {}
    gt_camera_dict = {}
    # 

    outIdx = 1
    for fpath in filelist:
        print(fpath)
        cvImg = cv2.imread(fpath)
        height, width, _ = cvImg.shape
        # 
        dirPath, fname = fpath.rsplit('/', 1)
        ext = os.path.splitext(fpath)[1]

        # read pose from xml file
        xmlFile = fname.replace('.jpg', '.xml')
        xmlFile = dirPath + '/' + xmlFile
        # currentpose = readPoseFromXML(xmlFile, translationScaling=1000) # swisscube
        currentpose = readPoseFromXML(xmlFile, translationScaling=1) # vespa
        R = currentpose[:3,:3]
        T = currentpose[:3,3].reshape(-1,1)

        # print(currentpose)
        baseName = ("%06d" % outIdx)
        # 
        if not debug:
            shutil.copyfile(fpath, out_rgb_dir + baseName + ext)
        # 
        K = get_camera_K(width, height, 100)
        # 
        mask = np.zeros((height, width), dtype=np.uint8)

        # print(currentpose)
        # print(K)
        rImg, depth = render_objects([mesh], [0], [currentpose], K, width, height)
        mask[depth>0] = 255
        outmaskName = baseName + "_000000.png" 
        if not debug:
            cv2.imwrite(out_maskvisib_dir + outmaskName, mask)

        if debug:
            # cvImg = draw_pose_axis(cvImg, R, T, keypoints, K, 1)

            # add delta to pose
            if False:
                deltaTheta = np.pi/64
                deltaTrans = 100
                deltaR = transforms3d.euler.euler2mat(np.random.uniform(-deltaTheta, deltaTheta),
                                                        np.random.uniform(-deltaTheta, deltaTheta),
                                                        np.random.uniform(-deltaTheta, deltaTheta), axes='szyx')
                deltaT = np.array([np.random.uniform(-deltaTrans, deltaTrans),
                                    np.random.uniform(-deltaTrans, deltaTrans),
                                    np.random.uniform(-deltaTrans, deltaTrans)]).reshape(-1,1)
                R = np.matmul(deltaR, R)
                T = T + deltaT

            # cvImg = draw_pose_axis(cvImg, R, T, keypoints, K, 1)

            # GT surface reprojection
            # if False:
            if True:
                print(K)
                print(R)
                print(T)
                mask2 = np.zeros((height, width), dtype=np.uint8)
                vertex = mesh.vertices
                pts = np.matmul(K, np.matmul(R, vertex.transpose()) + T)
                xs = pts[0] / pts[2]
                ys = pts[1] / pts[2]
                for pIdx in range(len(xs)):
                    mask2 = cv2.circle(mask2, (int(xs[pIdx]), int(ys[pIdx])), 1, 255, -1)
        # 
        # debug
        if debug:
            cv2.imshow("image", cvImg)
            cv2.imshow("render", rImg)
            cv2.imshow("mask_render", mask)
            cv2.imshow("mask_reproj", mask2)
            cv2.waitKey(0)
        # 
        pose = {}
        pose['cam_R_m2c'] = list(R.reshape(-1))
        pose['cam_t_m2c'] = list(T.reshape(-1))
        pose['obj_id'] = 1
        gt_dict[str(outIdx)] = [pose] # only one object
        # 
        cam = {}
        cam['cam_K'] = list(K.reshape(-1))
        cam['depth_scale'] = 0
        gt_camera_dict[str(outIdx)] = cam
        # 
        info = {}
        ys, xs = np.where(mask == 255)
        xmin = xmax = ymin = ymax = 0
        if len(ys) > 0:
            xmin = int(xs.min())
            xmax = int(xs.max())
            ymin = int(ys.min())
            ymax = int(ys.max())
        info['bbox_visib'] = [xmin,ymin,xmax-xmin,ymax-ymin]
        gt_info_dict[str(outIdx)] = [info]
        # 
        outIdx += 1

    if not debug:
        with open(out_gt_file, 'w') as outfile:
            json.dump(gt_dict, outfile, indent=2)
        with open(out_gt_camera_file, 'w') as outfile:
            json.dump(gt_camera_dict, outfile, indent=2)
        with open(out_gt_info_file, 'w') as outfile:
            json.dump(gt_info_dict, outfile, indent=2)

    return

def convert_to_bop_format(data_path, out_path, model_path, keypoints):
    # 
    imglist = [data_path+f for f in os.listdir(data_path) if f.endswith('.png') or f.endswith('.jpg')]
    # random.shuffle(imglist)
    imglist.sort()

    # trainCnt = int(len(imglist) * 0.7 + 0.5)
    # train_list = imglist[:trainCnt]
    # test_list = imglist[trainCnt:]
    train_list = imglist

    meshes, objID_2_clsID = load_bop_meshes(model_path)

    convert_list_to_bop_format(train_list, out_path, meshes[0], keypoints)
    # convert_list_to_bop_format(val_list, out_path + 'valid/', meshes[0], keypoints)
    # convert_list_to_bop_format(test_list, out_path + 'test/', meshes[0], keypoints)

if __name__ == "__main__":
    # path = "/data/vespa_27_03_2020_0001/"
    # # path = "/data2/Sebastien/vespa_19_03_2020_0000/"
    # model_path = "/data/vespa_syn_0327/models/"
    # outpath = "/data/vespa_syn_0327/"
    # convert_to_bop_format(path, outpath, model_path, None)

    # path = "/data/swisscube_29_03_2020_0002/"
    # dir_path = "/home/yhu/data/vespa_17_08_2020_0001/"
    # dir_path = "/data/clearspace/09_22_2020_swisscube_seq_v1/"
    # out_path = "/data/clearspace/09_22_2020_swisscube_seq_v1_out/"
    dir_path = "/home/yhu/github/data_preparation/output_training/"
    out_path = "/home/yhu/github/data_preparation/output_training_bop/"
    dirs = os.listdir(dir_path)
    dirs.sort()
    # model_path = "/data/vespa/models/"
    # model_path = "/data/swisscube/models/"
    # model_path = "/data/swisscube_20200922_hu/models_w_antennas/"
    model_path = "/data/VESPA_1.0/models/"

    seqIdx = 0
    # seqIdx = 350
    # seqIdx = 400
    for dd in dirs:
        iPath = dir_path + dd + '/'
        oPath = out_path + ("seq_%06d" % seqIdx) + '/'
        seqIdx += 1
        if os.path.isdir(iPath):
            convert_to_bop_format(iPath, oPath, model_path, None)
