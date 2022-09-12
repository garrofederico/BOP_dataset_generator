'''
for SPEED competition
Yinlin Hu 2019.06
'''

import os
import random
import json
import numpy as np
import sys
sys.path.append(os.path.abspath('.'))
import cv2
from sklearn.cluster import KMeans
import trimesh

import matplotlib.pyplot as plt
# import transforms3d
#
# import math
# import neural_renderer as nr
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision

#import bs4 as bs
import matplotlib.pyplot as plt

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

def readPoseFromXML(xmlFile, translationScaling=1):
    with open(xmlFile, 'r') as f:
        xmlContent = f.read()
    soup = bs.BeautifulSoup(xmlContent, features="xml")
    targetNode = soup.find("shape", {"type": "instance"})
    translate = targetNode.transform.translate
    dx, dy, dz = float(translate["x"]), float(translate["y"]), float(translate["z"])
    tT = np.array([dx,dy,dz]).reshape(-1,1)
    if True:
        # new version after tracking datasets
        rot = np.fromstring(targetNode.transform.matrix['value'], sep=" ")
        tR = rot.reshape(4, 4)[:3,:3]
    else:
        rotates = targetNode.transform.findAll("rotate")
        assert(rotates[0]["x"] == "1" and rotates[1]["y"] == "1" and rotates[2]["z"] == "1")
        rx, ry, rz = float(rotates[0]["angle"]), float(rotates[1]["angle"]), float(rotates[2]["angle"])
        tR = transforms3d.euler.euler2mat(rx*math.pi/180, ry*math.pi/180, rz*math.pi/180, axes='sxyz')
    # 
    camToWorld = lookAt(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    # 
    tT = translationScaling * tT # scale
    currentpose = np.concatenate((tR, tT.reshape(-1, 1)), axis=-1)
    currentpose = np.concatenate((currentpose, np.array([0,0,0,1]).reshape(1,4)), axis=0)
    # 
    currentpose = np.matmul(np.linalg.inv(camToWorld), currentpose) # pose based on camera frame
    return currentpose

def ycb_read_meshes(meshpath):
    objects = [f for f in os.listdir(meshpath)]
    objects.sort()
    # objects = [objects[0]] # debug
    meshes = []
    for objname in objects:
        mp = meshpath + objname + '/textured_simple.obj'
        mesh = trimesh.load(mp)
        meshes.append(mesh)
        print(mp + '...')
    return meshes

def get_bop_meshes_diameters(meshpath):
    objects = [f for f in os.listdir(meshpath) if f.endswith('.ply')]
    objects.sort()
    meshes = []
    for objname in objects:
        mp = meshpath + objname
        mesh = trimesh.load(mp)
        meshes.append(mesh)
        # print(mp + '...')
        vol = mesh.bounding_sphere.volume
        diameter = ((vol*3/(4*np.pi))**(1/3))*2
        print("%.2f" % diameter)
    return meshes

def linemod_read_meshes(meshpath):
    target_objects = ['ape', 'can', 'cat', 'driller',
                  'duck', 'eggbox', 'glue', 'holepuncher']
    meshes = []
    for objname in target_objects:
        mp = meshpath + objname + '.ply'
        mesh = trimesh.load(mp)
        meshes.append(mesh)
        print(mp + '...')
    return meshes

def quaternion2rotation(quat):
    '''
    Do not use the quat2dcm() function in the SPEED utils.py, it is not rotation
    '''
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)

def quaternion2rotation_torch(quat):
    assert (quat.shape[1] == 4)
    # normalize first
    quat = quat / quat.norm(p=2, dim=1).view(-1, 1)

    a = quat[:, 0]
    b = quat[:, 1]
    c = quat[:, 2]
    d = quat[:, 3]

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return torch.stack((m0, m1, m2, m3, m4, m5, m6, m7, m8), dim=1).view(-1, 3, 3)

def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q
    
def get_anchors(labeldir, target_width, target_height, cluster_numbers=9):
    # collect all subdirectories, only depth 1
    dirs = [labeldir]
    for i in os.listdir(labeldir):
        if os.path.isdir(labeldir + i):
            dirs.append(labeldir + i + os.sep)

    files = []
    for d in dirs:
        for f in os.listdir(d):
            name = d + f
            if name.endswith('.txt'):
                files.append(name)

    files.sort()
    all_bboxs = []
    for i in files:
        print(i)
        annot = np.loadtxt(i)
        if len(annot.shape) < 2:
            annot = annot.reshape(1,-1)
        for a in annot:
            bb = a[1:5]
            all_bboxs.append([bb[2] * target_width, bb[3] * target_height])
    # clustering
    all_bboxs = np.array(all_bboxs).reshape(-1, 2)
    kms = KMeans(n_clusters=cluster_numbers, random_state=0).fit(all_bboxs)
    return kms.cluster_centers_

def generate_turbulent_matrix(offset, angle, scale, width, height):
    dw = int(width * offset)
    dh = int(height * offset)
    pleft = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    shiftM = np.array([[1.0, 0.0, -pleft], [0.0, 1.0, -ptop], [0.0, 0.0, 1.0]])  # translation

    # random rotation and scaling
    cx = width / 2 # fix the rotation center to the image center
    cy = height / 2
    ang = random.uniform(-angle, angle)
    sfactor = random.uniform(-scale, +scale) + 1
    tmp = cv2.getRotationMatrix2D((cx, cy), ang, sfactor)  # rotation with scaling
    rsM = np.concatenate((tmp, [[0, 0, 1]]), axis=0)

    # combination
    M = np.matmul(rsM, shiftM)

    return M

def collect_images(path, outname):
    imgList = []
    for root, dirs, files in os.walk(path):
        for fName in files:
            fullPath = os.path.join(root, fName)
            if fullPath.endswith('.jpg') or fullPath.endswith('.png'):
                imgList.append(fullPath)
    imgList.sort()
    # write sets
    allf = open(outname, 'w')
    for i in imgList:
        allf.write(i +'\n')

def get_pose_statistics(labeldir, clsNum):
    # collect all subdirectories, only depth 1
    dirs = [labeldir]
    for i in os.listdir(labeldir):
        if os.path.isdir(labeldir + i):
            dirs.append(labeldir + i + os.sep)

    files = []
    for d in dirs:
        for f in os.listdir(d):
            name = d + f
            if name.endswith('.txt'):
                files.append(name)

    files.sort()
    all_quat = [[] for x in range(clsNum)]
    all_tran = [[] for x in range(clsNum)]
    for i in files:
        print(i)
        annot = np.loadtxt(i)
        if len(annot.shape) < 2:
            annot = annot.reshape(1,-1)
        for a in annot:
            clsId = int(a[0])
            bb = a[1:5]
            quat = a[5:9]
            tran = a[9:12]
            R = quaternion2rotation(quat)
            ai, aj, ak = transforms3d.euler.mat2euler(R, axes='szyx')
            # R = transforms3d.euler.euler2mat(ai, aj, ak, axes='szyx')
            all_quat[clsId].append([ai,aj,ak])
            all_tran[clsId].append(list(tran))
    return all_quat, all_tran

def get_class_weights(fileList, labelDir, nC):
    with open(fileList, 'r') as file:
        imglist = file.readlines()
        imglist.sort()

    class_freq = np.zeros(nC)
    for imgPath in imglist:
        imgPath = imgPath.rstrip()
        print(imgPath)
        # 
        temp, fName = os.path.split(imgPath)
        dirName = temp[temp.rfind(os.sep) + 1:]
        labelName = fName.replace(os.path.splitext(fName)[-1], '.txt')
        labelAuxName = fName.replace(os.path.splitext(fName)[-1], '.npz')
        candiPath = labelDir + labelName
        if os.path.exists(candiPath):# first search in the current directory
            labePath = candiPath
            labelAuxPath = labelDir + labelAuxName
        else:
            invalidPath = candiPath
            candiPath = labelDir + dirName + os.sep + labelName # search deeper
            if os.path.exists(candiPath):
                labePath = candiPath
                labelAuxPath = labelDir + dirName + os.sep + labelAuxName
            else:
                # giveup
                labePath = invalidPath
                labelAuxPath = invalidPath
        # 
        try:
            with open(labePath, 'r') as f:
                label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            classIdx = label[:, 0]
            # 
            annot = np.load(labelAuxPath)
            labels_m = annot['labelmask']
            for i in range(len(classIdx)):
                id = int(classIdx[i])
                class_freq[0] += (labels_m == 0).sum()
                class_freq[id+1] += (labels_m == (i+1)).sum()
        except:
            print('Loading failed: %s' % imgPath)

    # print(class_freq)
    weights = list(np.median(class_freq) / class_freq)
    str = ''
    for w in weights:
        str += ('%.3f,' % w)
    print(str[:-1])
    
def render_objects(meshes, ids, poses, K, w, h):
    '''
    '''
    assert(K[0][1] == 0 and K[1][0] == 0 and K[2][0] ==0 and K[2][1] == 0 and K[2][2] == 1)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    objCnt = len(ids)
    assert(len(poses) == objCnt)
    
    # set background with 0 alpha, important for RGBA rendering
    scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 0.0]), ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    # pyrender.Viewer(scene, use_raymond_lighting=True)
    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera = pyrender.IntrinsicsCamera(fx=fx,fy=fy,cx=cx,cy=cy,znear=0.05,zfar=100000)
    camera_pose = np.eye(4)
    # reverse the direction of Y and Z, check: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
    camera_pose[1][1] = -1
    camera_pose[2][2] = -1
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=4.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    # light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    # light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)
    for i in range(objCnt):
        clsId = int(ids[i])
        mesh = pyrender.Mesh.from_trimesh(meshes[clsId])

        H = np.zeros((4,4))
        H[0:3] = poses[i][0:3]
        H[3][3] = 1.0
        scene.add(mesh, pose=H)

    # pyrender.Viewer(scene, use_raymond_lighting=True)

    r = pyrender.OffscreenRenderer(w, h)
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    # flags = pyrender.RenderFlags.OFFSCREEN
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    color, depth = r.render(scene, flags=flags)
    # color, depth = r.render(scene)
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR) # RGB to BGR (for OpenCV)
    color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA) # RGBA to BGRA (for OpenCV)
    # # 
    if False:
        plt.figure()
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(color)
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        # plt.imshow(depth)
        plt.show()
    # # 
    # r.delete()
    # color = None
    # 
    return color, depth

def remap_pose(srcK, srcR, srcT, pt3d, dstK, transM):
    ptCnt = len(pt3d)
    pts = np.matmul(transM, np.matmul(srcK, np.matmul(srcR, pt3d.transpose()) + srcT))
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)
    xy2d = np.concatenate((xs.reshape(-1,1),ys.reshape(-1,1)), axis=1)
    # retval, rot, trans, inliers = cv2.solvePnPRansac(pt3d, xy2d, dstK, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=5.0)
    retval, rot, trans = cv2.solvePnP(pt3d.reshape(ptCnt,1,3), xy2d.reshape(ptCnt,1,2), dstK, None, flags=cv2.SOLVEPNP_EPNP)
    if retval:
        newR = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        newT = trans.reshape(-1, 1)

        newPts = np.matmul(dstK, np.matmul(newR, pt3d.transpose()) + newT)
        newXs = newPts[0] / (newPts[2] + 1e-8)
        newYs = newPts[1] / (newPts[2] + 1e-8)
        newXy2d = np.concatenate((newXs.reshape(-1,1),newYs.reshape(-1,1)), axis=1)
        diff_in_pix = np.linalg.norm(xy2d - newXy2d, axis=1).mean()

        # print('---')
        # print(srcK)
        # print(dstK)
        # print(srcR)
        # print(srcT.reshape(-1))
        # print(newR)
        # print(newT.reshape(-1))
        # print("diff in %f pixels" % diff_in_pix)
        
        return newR, newT, diff_in_pix
    else:
        print('Error in pose remapping!')
        assert(0)
        return srcR, srcT, -1

def generate_masks(meshes, ids, poses, K, w, h):
    _, mixed_depth = render_objects(meshes, ids, poses, K, w, h)
    objCnt = len(ids)
    assert(len(poses) == objCnt)
    ins_mask = np.zeros((h,w), dtype=np.uint32)
    vert_mask = np.zeros((h,w,3), dtype=np.float32)
    gridx, gridy = np.meshgrid(np.arange(w), np.arange(h))
    for i in range(objCnt):
        if objCnt == 1:
            # skip the duplicated rendering if with only one object
            current_depth = mixed_depth
            im = (current_depth != 0)
        else:
            _, current_depth = render_objects(meshes, [ids[i]], [poses[i]], K, w, h)
            im = np.logical_and(current_depth != 0, np.abs(mixed_depth  - current_depth) < 1e-8)
        ins_mask[im] = i+1
        xy1 = np.concatenate((gridx[im].reshape(1,-1),gridy[im].reshape(1,-1),np.ones_like(gridx[im]).reshape(1,-1)), axis=0)
        xyn = np.matmul(np.linalg.inv(K), xy1)
        # get 3D points in Camera coordinates
        # ref: https://math.stackexchange.com/questions/1650877/how-to-find-a-point-which-lies-at-distance-d-on-3d-line-given-a-position-vector
        xyzc = (current_depth[im].reshape(1, -1) * xyn) / (np.linalg.norm(xyn, axis=0).reshape(1, -1))
        # change to object coordinates
        vert = np.matmul(poses[i][:, :3].T, xyzc-poses[i][:, 3].reshape(-1, 1))
        vert_mask[im] = vert.T
        # debug
        if False:
            print(ids[i])
            # mesh = pyrender.Mesh.from_points(vert.T)
            # mesh = pyrender.Mesh.from_trimesh(meshes[ids[i]])
            # mesh = pyrender.Mesh.from_points(meshes[ids[i]].vertices)
            # mesh0 = pyrender.Mesh.from_points(np.array([[0,0,0]]), colors = np.array([[0,0,255]]))
            bs = meshes[ids[i]].bounding_sphere
            c = bs.centroid
            r = np.power((3 * bs.volume)/(4*np.pi), 1/3)
            mesh_sphere = trimesh.primitives.Sphere(radius=r, center=c, subdivisions=3)
        if False:
        # if True:
            # add color for sphere
            rgb = mesh_sphere.vertices / np.expand_dims(np.linalg.norm(mesh_sphere.vertices, axis=1), axis=1)
            rgb = (rgb+1)*255 / 2  # map [-1,1] to [0,255]
            mesh_sphere.visual.vertex_colors = rgb

            mesh_sphere = pyrender.Mesh.from_trimesh(mesh_sphere)
            scene = pyrender.Scene()
            # scene.add(mesh)
            # scene.add(mesh0)
            scene.add(mesh_sphere)
            pyrender.Viewer(scene)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(vert_mask)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # # plt.imshow(depth, cmap=plt.cm.gray_r)
    # plt.imshow(ins_mask)
    # plt.show()

    # # debug
    # tmpImg = np.ones((h,w,3), dtype=np.uint8) * 255
    # validMask = (ins_mask > 0)
    # verts = vert_mask[validMask]
    # rgbs = verts / np.expand_dims(np.linalg.norm(verts, axis=1), axis=1)
    # rgbs = (rgbs+1)*255 / 2  # map [-1,1] to [0,255]
    # tmpImg[validMask] = rgbs

    # cv2.imshow('vertmap', tmpImg)
    # cv2.waitKey(0)

    if False:
    # if True:
        tmpImg = cv2.normalize(np.uint8(ins_mask), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("maskImg", tmpImg)
        tmpImg = vert_mask / np.expand_dims(np.linalg.norm(vert_mask, axis=2), axis=2)
        tmpImg = cv2.normalize(tmpImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("vertImg", tmpImg)
        # 
        checkMask = np.zeros((h,w), dtype=np.uint32)
        for i in range(objCnt):
            im = (ins_mask == i+1)
            rep = np.matmul(K, np.matmul(poses[i][:, :3], vert_mask[im].T) + poses[i][:, 3].reshape(-1, 1))
            xs = np.int32(rep[0]/rep[2])
            ys = np.int32(rep[1]/rep[2])
            checkMask[ys,xs] = 255
        tmpImg = cv2.normalize(np.uint8(checkMask), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow("check Mask", tmpImg)
        cv2.waitKey(0)

    return ins_mask, vert_mask

def add_pose_contour(mesh, K, R, T, color, cvImg):
    # 
    h, w, _ = cvImg.shape
    currentpose = np.concatenate((R, T.reshape(-1, 1)), axis=-1)
    _, depth = render_objects([mesh], [0], [currentpose], K, w, h)
    validMap = (depth>0).astype(np.uint8)
    # 
    # find contour
    contours, _ = cv2.findContours(validMap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # cvImg = cv2.drawContours(cvImg, contours, -1, (255, 255, 255), 2, cv2.LINE_AA) # border
    cvImg = cv2.drawContours(cvImg, contours, -1, color, 4)
    return cvImg

def load_bop_meshes(model_path):
    # load meshes
    meshFiles = [f for f in os.listdir(model_path) if f.endswith('.ply')]
    meshFiles.sort()
    meshes = []
    objID_2_clsID = {}
    for mFile in meshFiles:
        print('loading %s' % (model_path + mFile))
        objId = int(os.path.splitext(mFile)[0][4:])
        objID_2_clsID[str(objId)] = len(meshes)
        meshes.append(trimesh.load(model_path + mFile))
    # 
    return meshes, objID_2_clsID

def draw_bounding_box(cvImg, R, T, bbox, intrinsics, color, thickness):
    rep = np.matmul(intrinsics, np.matmul(R, bbox.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    bbox_lines = [0, 1, 0, 2, 0, 4, 5, 1, 5, 4, 6, 2, 6, 4, 3, 2, 3, 1, 7, 3, 7, 5, 7, 6]
    for i in range(12):
        id1 = bbox_lines[2*i]
        id2 = bbox_lines[2*i+1]
        cvImg = cv2.line(cvImg, (x[id1],y[id1]), (x[id2],y[id2]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_pose_axis(cvImg, R, T, intrinsics, thickness, box3d=None):
    if box3d:
        radius = np.linalg.norm(bbox, axis=1).mean()
    else:
        radius = 0.1
    aPts = np.array([[0,0,0],[0,0,radius],[0,radius,0],[radius,0,0]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[1],y[1]), (0,0,255), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[2],y[2]), (0,255,0), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[3],y[3]), (255,0,0), thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg
    
def dataset_statistics(list_file, model_path):
    # load file list
    with open(list_file, 'r') as f:
        filenames = f.readlines()

    meshes, objID_2_clsID = load_bop_meshes(model_path)
    colors = plt.get_cmap('tab20', len(meshes)).colors * 255
    
    debug = False
    # debug = True

    all_Ts = []

    # read GT
    dir_annots = {}
    for img_path in filenames:
        img_path = img_path.strip()
        print(img_path)
        # 
        if debug:
            cvImg = cv2.imread(img_path)
            h, w, _ = cvImg.shape
            contourImg = np.copy(cvImg)

        # 
        gt_dir, tmp, imgName = img_path.rsplit('/', 2)
        assert(tmp == 'rgb')
        imgBaseName, _ = os.path.splitext(imgName)
        im_id = int(imgBaseName)
        # 
        camera_file = gt_dir + '/scene_camera.json'
        gt_file = gt_dir + "/scene_gt.json"
        gt_info_file = gt_dir + "/scene_gt_info.json"
        gt_mask_visib = gt_dir + "/mask_visib/"

        if gt_dir in dir_annots:
            gt_json, gt_info_json, cam_json = dir_annots[gt_dir]
        else:
            gt_json = json.load(open(gt_file))
            gt_info_json = json.load(open(gt_info_file))
            cam_json = json.load(open(camera_file))
            dir_annots[gt_dir] = [gt_json, gt_info_json, cam_json]

        annot_camera = cam_json[str(im_id)]
        annot_poses = gt_json[str(im_id)]
        annot_infos = gt_info_json[str(im_id)]

        objCnt = len(annot_poses)
        K = np.array(annot_camera['cam_K']).reshape(3,3)

        for i in range(objCnt):
            # 
            R = np.array(annot_poses[i]['cam_R_m2c']).reshape(3,3)
            T = np.array(annot_poses[i]['cam_t_m2c']).reshape(3,1)
            obj_id = annot_poses[i]['obj_id']
            cls_id = objID_2_clsID[str(obj_id)]
            # 
            all_Ts.append(float(T[0]))
            all_Ts.append(float(T[1]))
            all_Ts.append(float(T[2]))

            if debug:
                mask_vis_file = gt_mask_visib + ("%06d_%06d.png" %(im_id, i))
                mask_vis = cv2.imread(mask_vis_file, cv2.IMREAD_UNCHANGED)
                bbox = annot_infos[i]['bbox_visib']
                contourImg = cv2.rectangle(contourImg, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255))
                # cv2.imshow(str(i), mask_vis)
                # if True:
                if False:
                    contourImg = add_pose_contour(meshes[cls_id], K, R, T, colors[cls_id], contourImg)
                else:
                    # GT surface reprojection
                    vertex = meshes[cls_id].vertices
                    pts = np.matmul(K, np.matmul(R, vertex.transpose()) + T)
                    xs = pts[0] / pts[2]
                    ys = pts[1] / pts[2]
                    for pIdx in range(len(xs)):
                        contourImg = cv2.circle(contourImg, (int(xs[pIdx]), int(ys[pIdx])), 3, colors[cls_id], -1)
        if debug:
            cv2.imshow("pose", contourImg)
            cv2.waitKey(0)
    all_Ts = np.array(all_Ts).reshape(-1, 3)
    mean_T = all_Ts.mean(axis=0)
    std_T = all_Ts.std(axis=0)
    return mean_T, std_T

def get_single_bop_annotation(img_path):
    # add attributes to function, for fast loading
    if not hasattr(get_single_bop_annotation, "dir_annots"):
        get_single_bop_annotation.dir_annots = {}
    # 
    img_path = img_path.strip()
    # 
    gt_dir, tmp, imgName = img_path.rsplit('/', 2)
    assert(tmp == 'rgb')
    imgBaseName, _ = os.path.splitext(imgName)
    im_id = int(imgBaseName)
    # 
    camera_file = gt_dir + '/scene_camera.json'
    gt_file = gt_dir + "/scene_gt.json"
    gt_info_file = gt_dir + "/scene_gt_info.json"
    gt_mask_visib = gt_dir + "/mask_visib/"

    if gt_dir in get_single_bop_annotation.dir_annots:
        gt_json, gt_info_json, cam_json = get_single_bop_annotation.dir_annots[gt_dir]
    else:
        gt_json = json.load(open(gt_file))
        gt_info_json = json.load(open(gt_info_file))
        cam_json = json.load(open(camera_file))
        get_single_bop_annotation.dir_annots[gt_dir] = [gt_json, gt_info_json, cam_json]

    annot_camera = cam_json[str(im_id)]
    annot_poses = gt_json[str(im_id)]
    annot_infos = gt_info_json[str(im_id)]

    objCnt = len(annot_poses)
    K = np.array(annot_camera['cam_K']).reshape(3,3)

    items = []
    for i in range(objCnt):
        mask_vis_file = gt_mask_visib + ("%06d_%06d.png" %(im_id, i))
        mask_vis = cv2.imread(mask_vis_file, cv2.IMREAD_UNCHANGED)
        # 
        bbox = annot_infos[i]['bbox_visib']
        # bbox = annot_infos[i]['bbox_obj']
        # contourImg = cv2.rectangle(contourImg, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255))
        # cv2.imshow(str(i), mask_vis)
        # 
        R = np.array(annot_poses[i]['cam_R_m2c']).reshape(3,3)
        T = np.array(annot_poses[i]['cam_t_m2c']).reshape(3,1)
        obj_id = annot_poses[i]['obj_id']
        # cls_id = objID_2_clsID[str(obj_id)]
        items.append([obj_id, R, T, bbox, mask_vis])
    
    return K, items

def show_annotations(list_file, model_path):
    # load file list
    dataDir = os.path.split(list_file)[0]
    with open(list_file, 'r') as f:
        filenames = f.readlines()
    filenames = [dataDir+'/'+x.strip() for x in filenames]

    meshes, objID_2_clsID = load_bop_meshes(model_path)
    colors = plt.get_cmap('tab20', len(meshes)).colors * 255
    
    # read GT
    for img_path in filenames:
        K, items = get_single_bop_annotation(img_path)
        # 
        img_path = img_path.strip()
        print(img_path)
        # 
        cvImg = cv2.imread(img_path)
        h, w, _ = cvImg.shape
        contourImg = np.copy(cvImg)
        #
        if False:
            sketch_gray, sketch_color = cv2.pencilSketch(cvImg, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            cv2.imshow("gray", sketch_gray)
            cv2.imshow("color", sketch_gray)
            # cv2.waitKey(0)
            # 

        for itm in items:
            obj_id, R, T, bbox, mask_vis = itm
            print(T)
            # if float(T[2]) < 2.7:
            #     continue
            # 
            contourImg = cv2.rectangle(contourImg, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255), 5)
            # cv2.imshow(str(i), mask_vis)
            cls_id = objID_2_clsID[str(obj_id)]
            if True:
            # if False:
                contourImg = add_pose_contour(meshes[cls_id], K, R, T, colors[cls_id], contourImg)
                # contourImg = draw_pose_axis(contourImg, R, T, K, 10)
                pass
            else:
                # GT surface reprojection
                vertex = meshes[cls_id].vertices
                pts = np.matmul(K, np.matmul(R, vertex.transpose()) + T)
                xs = pts[0] / pts[2]
                ys = pts[1] / pts[2]
                for pIdx in range(len(xs)):
                    contourImg = cv2.circle(contourImg, (int(xs[pIdx]), int(ys[pIdx])), 1, colors[cls_id], -1)

            cv2.imshow("pose", contourImg)
            cv2.waitKey(0)

def show_depth_distribution_(filenameList, scaling_factor = 1):
    # meshes, objID_2_clsID = load_bop_meshes(model_path)
    # colors = plt.get_cmap('tab20', len(meshes)).colors * 255
    # read GT
    # get depth statistics
    all_depths = []

    for img_path in filenameList:
        K, items = get_single_bop_annotation(img_path.strip())
        for itm in items:
            obj_id, R, T, bbox, mask_vis = itm
            depth = float(T[2])
            print(depth)
            all_depths.append(depth)
    plt.hist((np.array(all_depths) * scaling_factor + 0.5).astype(int), bins=10)
    plt.show()

def show_depth_distribution(list_file, scaling_factor = 1):
    # load file list
    dataDir = os.path.split(list_file)[0]
    with open(list_file, 'r') as f:
        filenames = f.readlines()
    filenames = [dataDir+'/'+f for f in filenames]
    show_depth_distribution_(filenames, scaling_factor)

def resample_directories_according_to_depth(rootDir, model_filepath, outDir, min_depth, max_depth):
    subDirs = os.listdir(rootDir)
    subDirs = [rootDir+d+os.sep for d in subDirs if os.path.isdir(rootDir+d)]
    subDirs.sort()

    filenames = []
    mesh = trimesh.load(model_filepath)

    allPathsRaw = []
    allDepthsRaw = []
    for dir in subDirs:
        files = os.listdir(dir)
        files = [dir + f for f in files if f.endswith('.jpg')]
        files.sort()
        # 
        paths = []
        depths = []
        for img_path in files:
            print(img_path)
            # read pose from xml file
            xmlFile = img_path.replace('.jpg', '.xml')
            # currentpose = readPoseFromXML(xmlFile, translationScaling=1000) # swisscube
            currentpose = readPoseFromXML(xmlFile, translationScaling=1) # vespa
            R = currentpose[:3,:3]
            T = currentpose[:3,3].reshape(-1,1)

            # tmpImg = cv2.imread(img_path)
            # height, width, _ = tmpImg.shape
            height = 1024
            width = 1024
            K = get_camera_K(width, height, 100)

            # reprojection of the center
            rep = np.matmul(K, T)
            cx = (rep[0]/rep[2])[0]
            cy = (rep[1]/rep[2])[0]
            if cx<0 or cx>=width or cy<0 or cy>height:
                continue
            dpt = T[2][0]
            print(dpt)
            if dpt < min_depth or dpt > max_depth:
                continue

            paths.append(img_path)
            depths.append(dpt)

            # debug
            if False:
            # if True:
                print(T)
                tmpImg = cv2.imread(img_path)
                tmpImg = add_pose_contour(mesh, K, R, T, (0,0,255), tmpImg)
                tmpImg = cv2.circle(tmpImg, (int(cx),int(cy)), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow("xx",tmpImg)
                cv2.waitKey(0)
        
        allPathsRaw.append(paths)
        allDepthsRaw.append(depths)

    # remove requences too short
    allPaths = []
    allDepths = []
    for i in range(len(allPathsRaw)):
        if len(allPathsRaw[i]) >= 20:
            allPaths.append(allPathsRaw[i])
            allDepths.append(allDepthsRaw[i])

    #
    depthStore = []
    avgDepths = [0]*len(allDepths)
    for i in range(len(allDepths)):
        avgDepths[i] = np.array(allDepths[i]).mean()
        depthStore += allDepths[i]

    removeSeqNum = 10 #int(len(allDepths)/2) # remove half of them
    removeSeqFlag = [0]*len(avgDepths)
    binCnt = 10
    bins, ths = np.histogram(avgDepths, binCnt)
    for i in range(removeSeqNum):
        candiIdx = bins.argmax()
        minD = ths[candiIdx]
        maxD = ths[candiIdx+1]
        # remove a sequence
        for j in range(len(avgDepths)):
            if avgDepths[j]>=minD and avgDepths[j]<=maxD:
                avgDepths[j] = 0
                removeSeqFlag[j] = 1
                bins[candiIdx] -= 1
                break

    reservedFiles = []
    for i in range(len(removeSeqFlag)):
        if not removeSeqFlag[i]:
            reservedFiles += allPaths[i]

    # copy files
    for f in reservedFiles:
        imgf = f
        xmlf = f.replace('.jpg', '.xml')
        idx = f.rfind(os.sep, 0, f.rfind(os.sep))
        tgtf = outDir + f[idx:]
        tgtXmlf = outDir + xmlf[idx:]
        os.makedirs(tgtf[:tgtf.rfind(os.sep)], exist_ok=True)
        cmdStr = "cp %s %s" %(imgf, tgtf)
        print(cmdStr)
        os.system(cmdStr)
        cmdStr = "cp %s %s" %(xmlf, tgtXmlf)
        print(cmdStr)
        os.system(cmdStr)

def resample_trainset_according_to_depth(list_file, new_file, min_depth=-1, max_depth=-1):
    # load file list
    dataDir = os.path.split(list_file)[0]
    with open(list_file, 'r') as f:
        filenames = f.readlines()

    # meshes, objID_2_clsID = load_bop_meshes(model_path)
    # colors = plt.get_cmap('tab20', len(meshes)).colors * 255
    
    # read GT
    # get depth statistics
    all_depths = []
    for img_path in filenames:
        K, items = get_single_bop_annotation(dataDir+'/'+ img_path.strip())
        for itm in items:
            obj_id, R, T, bbox, mask_vis = itm
            if (mask_vis == 255).sum() < 10:
                continue
            depth = T[2]
            if min_depth > 0 and max_depth > 0:
                if depth >= min_depth and depth <= max_depth:
                    print(depth)
                    all_depths.append(depth)
            else:
                print(depth)
                all_depths.append(depth)
    all_depths = np.array(all_depths)
    plt.hist((all_depths + 0.5).astype(int), bins=100)
    plt.show()

    if min_depth <= 0 or max_depth <= 0:
        min_depth = all_depths.min()
        max_depth = all_depths.max()

    binCount = 100
    binStep = (max_depth - min_depth) / binCount
    histBins = ((all_depths - min_depth) / binStep + 0.5).astype(int)
    frequencies = np.histogram(histBins, bins=binCount)[0]

    max_freq = frequencies.max()

    # resampling
    new_file_list = []
    for img_path in filenames:
        K, items = get_single_bop_annotation(dataDir+'/'+ img_path.strip())
        for itm in items:
            obj_id, R, T, bbox, mask_vis = itm
            if (mask_vis == 255).sum() < 10:
                continue
            depth = T[2]
            if depth >= min_depth and depth <= max_depth:
                print(depth)
                binIdx = int((depth - min_depth) / binStep + 0.5)
                if binIdx == binCount:
                    binIdx -= 1
                repeat_count = int( max_freq / frequencies[binIdx] + 0.5)
                new_file_list += ([img_path] * repeat_count)

    with open(new_file, 'w') as f:
        for item in new_file_list:
            f.write(item)

def resample_valtestset_according_to_depth(list_file, new_file, min_depth=-1, max_depth=-1):
    # load file list
    dataDir = os.path.split(list_file)[0]
    with open(list_file, 'r') as f:
        filenames = f.readlines()

    # meshes, objID_2_clsID = load_bop_meshes(model_path)
    # colors = plt.get_cmap('tab20', len(meshes)).colors * 255
    
    # read GT
    # get depth statistics
    all_depths = []
    for img_path in filenames:
        K, items = get_single_bop_annotation(dataDir+'/'+ img_path.strip())
        for itm in items:
            obj_id, R, T, bbox, mask_vis = itm
            if (mask_vis == 255).sum() < 10:
                continue
            depth = T[2]
            if min_depth > 0 and max_depth > 0:
                if depth >= min_depth and depth <= max_depth:
                    print(depth)
                    all_depths.append(depth)
            else:
                print(depth)
                all_depths.append(depth)
    all_depths = np.array(all_depths)
    plt.hist((all_depths + 0.5).astype(int), bins=100)
    plt.show()

    if min_depth <= 0 or max_depth <= 0:
        min_depth = all_depths.min()
        max_depth = all_depths.max()

    binCount = 100
    binStep = (max_depth - min_depth) / binCount
    histBins = ((all_depths - min_depth) / binStep + 0.5).astype(int)
    frequencies = np.histogram(histBins, bins=binCount)[0]

    th_freq = frequencies.mean() - frequencies.std()

    # resampling
    new_file_list = []
    for img_path in filenames:
        K, items = get_single_bop_annotation(dataDir+'/'+ img_path.strip())
        for itm in items:
            obj_id, R, T, bbox, mask_vis = itm
            if (mask_vis == 255).sum() < 10:
                continue
            depth = T[2]
            if depth >= min_depth and depth <= max_depth:
                print(depth)
                binIdx = int((depth - min_depth) / binStep + 0.5)
                if binIdx == binCount:
                    binIdx -= 1
                reserv_prob = th_freq / frequencies[binIdx]
                if random.uniform(0, 1) <= reserv_prob:
                    new_file_list.append(img_path)

    with open(new_file, 'w') as f:
        for item in new_file_list:
            f.write(item)


def show_tensor_image(windowName, tensor):
    _, channel, _, _ = tensor.shape
    tmpImg = tensor.detach().cpu().numpy()[0].transpose((1, 2, 0))
    if channel == 3:
        tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_RGB2BGR)
    tmpImg = cv2.normalize(tmpImg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # tmpImg = cv2.resize(tmpImg, (w, h))
    cv2.imshow(windowName, tmpImg)

def rgb_to_gray(image):
    _, channel, height, width = image.shape
    assert(channel == 3)
    result = 0.2989 * image[:,0,...] + 0.5870 * image[:,1,...] + 0.1140 * image[:,2,...]
    return result.unsqueeze(1)

def img_diff(img1, img2):
    window_size1 = 5
    sigma1 = 1
    window_size2 = 5
    sigma2 = 3

    # img1 = rgb_to_gray(img1)
    # img2 = rgb_to_gray(img2)
    _, channel, height, width = img1.shape

    window1 = create_window(window_size1, channel, sigma1).type_as(img1)
    window2 = create_window(window_size2, channel, sigma2).type_as(img1)

    img1 = F.interpolate(img1, scale_factor=0.5, mode='bilinear')
    img2 = F.interpolate(img2, scale_factor=0.5, mode='bilinear')
    # show_tensor_image("im1", img1)
    # show_tensor_image("im2", img2)

    loss = 0
    layers = 1
    for s in range(layers):
        # mag1, theta1 = sobel_filter(img1)
        # mag2, theta2 = sobel_filter(img2)

        t1 = F.conv2d(img1, window1, padding = window_size1//2, groups = channel)
        t2 = F.conv2d(img1, window2, padding = window_size2//2, groups = channel)
        # img1 = torch.clamp(t1 / (t2 + 1e-5), max=2.0)

        t1 = F.conv2d(img2, window1, padding = window_size1//2, groups = channel)
        t2 = F.conv2d(img2, window2, padding = window_size2//2, groups = channel)
        # img2 = torch.clamp(t1 / (t2 + 1e-5), max=2.0)

        ssim_out = ssim(img1, img2)

        show_tensor_image("ssim", ssim_out)

        show_tensor_image("im1", img1)
        show_tensor_image("im2", img2)

        # show_tensor_image("mag1", mag1)
        # show_tensor_image("theta1", theta1)
        # show_tensor_image("mag2", mag2)
        # show_tensor_image("theta2", theta2)
        # 
        # loss = loss + (img1 - img2).abs().sum()
        loss = loss + ssim_out.abs().sum()
    
    # cv2.waitKey(0)
    return loss

def sobel_filter(x):
    batch, channel, h, w = x.shape
    # x = gaussian_smooth_tensor(x)
    x = x.mean(dim=1, keepdim=True)

    #Black and white input image x, 1x1xHxW
    a = torch.Tensor([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]).type_as(x)
    a = a.view((1,1,3,3))
    G_x = F.conv2d(x, a, padding = 1)
    b = torch.Tensor([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]).type_as(x)
    b = b.view((1,1,3,3))
    G_y = F.conv2d(x, b, padding = 1)
    # 
    theta = torch.atan2(G_y, G_x) # [-pi, pi]
    mag = torch.sqrt(G_x.pow(2)+G_y.pow(2))
    return mag, theta

def pose_refinement(list_file, model_path):
    # load file list
    with open(list_file, 'r') as f:
        filenames = f.readlines()
    root_path = os.path.split(list_file)[0] + os.sep

    meshes, objID_2_clsID = load_bop_meshes(model_path)
    colors = plt.get_cmap('tab20', len(meshes)).colors * 255
    
    # read GT
    for img_path in filenames:
        img_path = root_path + img_path.strip()
        K, items = get_single_bop_annotation(img_path)
        # 
        cvImg = cv2.imread(img_path)
        h, w, _ = cvImg.shape
        contourImg = np.copy(cvImg)
        # 
        renderWidth = int(w/2) # for efficiency
        # renderWidth = w
        for idx in range(len(items)):
            # if idx <= 3:
            #     continue
            obj_id, R, T, bbox, mask_vis = items[idx]
            cls_id = objID_2_clsID[str(obj_id)]
            # 
            # contourImg = add_pose_contour(meshes[cls_id], K, R, T, colors[cls_id], contourImg)
            # cv2.imshow("raw pose", contourImg)

            cvImg_n = cv2.resize(cvImg, (renderWidth, int(renderWidth*h/w)))
            cvImg_n = cv2.cvtColor(cvImg_n, cv2.COLOR_BGR2RGB)
            cvImg_n = cvImg_n.transpose(2, 0, 1)  # 3x416x416
            cvImg_n = np.ascontiguousarray(cvImg_n, dtype=np.float32)  # uint8 to float32
            cvImg_n /= 255.0  # 0 - 255 to 0.0 - 1.0
            cvImg_n = torch.from_numpy(cvImg_n).cuda().unsqueeze(0)

            # debug
            # print('---')
            # print(clsId)
            # print(R)
            # print(T)
            # print(K)
            mask_vis = cv2.resize(mask_vis, (renderWidth, int(renderWidth*h/w)), interpolation=cv2.INTER_NEAREST)
            mask_vis_tf = torch.from_numpy(mask_vis).repeat(1,3,1,1)

            ROI = (mask_vis == 255)
            nROI = (mask_vis == 0)

            # meshes[cls_id].show(viewer='gl')
            rdModel = Neural_Render_Module(meshes[cls_id], K, R, T, orig_size=w, image_size=renderWidth)
            rdModel.cuda()
            optimizer = torch.optim.Adam(rdModel.parameters(), lr=1e-1)
            for i in range(1000):
                optimizer.zero_grad()
                rImg = rdModel()
                rImg = rImg[:,:,:int(renderWidth*h/w),:] # cut out the desired part

                # set gray background first
                cvImg_n[mask_vis_tf == 0] = 0.5
                rImg[mask_vis_tf==0] = 0.5

                # if False:
                if True:
                    # show rendered image
                    tmpImg = rImg.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
                    tmpImg = (255*tmpImg).astype(np.uint8)
                    tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_RGB2BGR)
                    tmpImg = cv2.resize(tmpImg, (w, h))
                    cv2.imshow("render", tmpImg)

                    # show scaled image accordingly
                    tmpImg = cvImg_n.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
                    tmpImg = (255*tmpImg).astype(np.uint8)
                    tmpImg = cv2.cvtColor(tmpImg, cv2.COLOR_RGB2BGR)
                    tmpImg = cv2.resize(tmpImg, (w, h))
                    cv2.imshow("scaled image", tmpImg)

                loss = img_diff(cvImg_n, rImg)

                # ssimMap = ssim(cvImg_n, rImg)
                # # 
                # # show ssim map
                # tmpImg = ssimMap.detach().cpu().numpy()[0].transpose((1, 2, 0)) 
                # tmpImg = (255*tmpImg).clip(0,255).astype(np.uint8)
                # tmpImg = cv2.resize(tmpImg, (w, h))
                # cv2.imshow("ssim", tmpImg)
                # loss = torch.exp(-1 * ssimMap[0][0][ROI].mean())
                
                # cvImg_n_masked = cvImg_n.permute(0,2,3,1)[0]
                # cvImg_n_masked[nROI] = 0.5
                # cvImg_n_masked = cvImg_n_masked.permute(2,0,1).unsqueeze(0)
                # # 
                # iImgEdge = edge_filter(cvImg_n_masked)
                # iImgEdge[0][0][nROI] = 0
                # show_edge(iImgEdge.detach().cpu().numpy()[0], "iImgEdge", w, h)
                # rImgEdge = edge_filter(rImg)
                # rImgEdge[0][0][nROI] = 0
                # show_edge(rImgEdge.detach().cpu().numpy()[0], "rImgEdge", w, h)
                # show_edge((iImgEdge-rImgEdge).abs().detach().cpu().numpy()[0], "diffEdge", w, h)
                # loss = 0.0001 * (iImgEdge-rImgEdge).pow(2)[0][0][ROI].mean()

                # print(rdModel.R)
                # print(rdModel.T)
                print(loss)
                loss.backward()
                optimizer.step()


                # # show contour
                # contourImg = np.copy(cvImg)
                # R = rdModel.R[0].detach().cpu().numpy()
                # T = rdModel.T[0].detach().cpu().numpy().reshape(-1,1)
                # # GT surface reprojection
                # pts = np.matmul(K, np.matmul(R, vertex[clsId][:].T) + T)
                # xs = pts[0] / pts[2]
                # ys = pts[1] / pts[2]
                # maskImg = np.zeros((h,w), np.uint8)
                # # maskImg[ys.astype(np.int32).clip(0, h-1), xs.astype(np.int32).clip(0, w-1)] = 255
                # for pIdx in range(len(xs)):
                #     maskImg = cv2.circle(maskImg, (int(xs[pIdx]), int(ys[pIdx])), 2, 255, -1)
                # # cv2.imshow("mask", maskImg)
                # # cv2.waitKey(0)
                # # 
                # # fill the holes
                # maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel=np.ones((5,5), np.uint8))
                # # find contour
                # _, contours, _ = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                # contourImg = cv2.drawContours(contourImg, contours, -1, colors[clsId], 2, cv2.LINE_AA) # border
                # cv2.imshow("pose", contourImg)

                cv2.waitKey(10)

if __name__ == "__main__":
    # collect_images('/data/SUN2012pascalformat/JPEGImages/', 'backgrounds.txt')
    # collect_images('/data/synthetic_sphere/train_images/', '/data/synthetic_sphere/train.txt')
    # collect_images('/data/synthetic_sphere/test_images/', '/data/synthetic_sphere/test.txt')
    # 
    # collect_images('/data/LINEMOD_ICG_aux/synthetic_images/', '/data/LINEMOD_ICG_aux/syn.txt')
    # collect_images('/data/LINEMOD_ICG_aux/blend_images/', '/data/LINEMOD_ICG_aux/blend.txt')
    # collect_images('/data/YCB_Video_Dataset_aux/synthetic_images/', '/data/YCB_Video_Dataset_aux/syn.txt')
    
    # filelistPath = "/data/LINEMOD_OCC_aux/test.txt"
    # labelPath = "/data/LINEMOD_OCC_aux/labels/"
    # vertex = "/home/yhu/Github/yolo_pose/yolo/data/linemod_vertex.npy"
    
    # list_file = "/data/BOP/ycbv/train_real.txt"
    # list_file = "/data/BOP/ycbv/train_synt.txt"
    # model_path = "/data/BOP/ycbv/models_light/"
    # model_path = "/data/BOP/ycbv/models_eval/"
    # model_path = "/data/BOP/ycbv/models/"
    # 
    # list_file = "/data/BOP/lm/test.txt"
    # model_path = "/data/BOP/lm/models/"
    # 
    # list_file = "/data/BOP/lmo/test.txt"
    # model_path = "/data/BOP/lmo/models/"
    # 
    # list_file = "/data/BOP/tudl/test.txt"
    # list_file = "/data/BOP/tudl/train_real.txt"
    # model_path = "/data/BOP/tudl/models/"
    # 
    # list_file = "/data/BOP/tyol/test.txt"
    # model_path = "/data/BOP/tyol/models/"
    # 
    # list_file = "/data/BOP/tless/test_primesense.txt"
    # list_file = "/data/BOP/tless/train_primesense.txt"
    # list_file = "/data/BOP/tless/train_render_reconst.txt"
    # model_path = "/data/BOP/tless/models_cad/"
    # model_path = "/data/BOP/tless/models_reconst/"

    # 
    # list_file = "/data2/VESPA_Sebastien/train.txt"
    # model_path = "/data2/VESPA_Sebastien/models/"
    # list_file = "/home/yhu/data/yy/vespa_20200911_hu/train.txt"
    # list_file = "//home/yhu/tmpdata/output/09_21_2020_swisscube_seq_v1_out1.txt"

    # get_bop_meshes_diameters("/data/occ_linemod_custom/models/")
    
    # list_file = "/data/VESPA_1.0/training.txt"
    # model_path = "/data/swisscube_20200922_hu/models_w_antennas/"
    # list_file = "/data/vespa_20200918_hu/test_normalized.txt"
    # model_path = "/data/VESPA_1.0/models/"
    # model_path = "/data/swisscube_20200922_hu/models_w_antennas/"
    # show_annotations(list_file, model_path)
    # show_depth_distribution(list_file, scaling_factor=1000)

    # model_filepath = "/data/swisscube_20200922_hu/models_w_antennas/obj_000001.ply"
    # model_filepath = "/data/VESPA_1.0/models/obj_000001.ply"
    # resample_directories_according_to_depth("/home/yhu/github/clearspace-scripts/python/output_swisscube/training/", model_filepath, "output_training", 178.46, 1784.6)
    # resample_directories_according_to_depth("/data/output_vespa/testing/", model_filepath, "output_testing", 0.31, 3.1)

    # 
    # list_file = "/data/swisscube_20200922_hu/train_raw.txt"
    # new_file = "/data/swisscube_20200922_hu/train_new.txt"
    # resample_trainset_according_to_depth(list_file, new_file, 170, 1900)
    # show_depth_distribution(new_file)
    
    # list_file = "/data/swisscube_20200922_hu/test_normalized.txt"
    # new_file = "/data/swisscube_20200922_hu/valid_new.txt"
    # resample_valtestset_according_to_depth(list_file, new_file, 170, 1900)
    # show_depth_distribution(list_file)
    
    # list_file = "/data/vespa_20200918_hu/train_raw.txt"
    # new_file = "/data/vespa_20200918_hu/train_new.txt"
    # resample_trainset_according_to_depth(list_file, new_file, 0.27, 2.80)
    # show_depth_distribution(new_file, scaling_factor=1000)
    
    # list_file = "/data/vespa_20200918_hu/valid_raw.txt"
    # new_file = "/data/vespa_20200918_hu/valid_new.txt"
    # resample_valtestset_according_to_depth(list_file, new_file, 0.27, 2.80)
    # show_depth_distribution(new_file, scaling_factor=1000)
    # # 
    # list_file = "/data/speed_custom/test.txt"
    # new_file = "/data/speed_custom/train_new.txt"
    # resample_trainset_according_to_depth(list_file, new_file)
    # show_depth_distribution(list_file)

    list_file = "/data/ycbv/test.txt"
    model_path = "/data/ycbv/models_vc/"
    # list_file = "/data/Occ-LINEMOD/test.txt"
    # model_path = "/data/Occ-LINEMOD/models/"
    pose_refinement(list_file, model_path)
    # 
    
    # mesh_sphere = trimesh.primitives.Sphere(radius=1, subdivisions=3)
    # r = np.eye(3)
    # t = np.array([0, 0, 5]).reshape(-1, 1)
    # pose = np.concatenate((r, t), axis=1)
    # K = np.array([[800, 0.0, 320],

    #               [0.0, 800, 240],
    #               [0.0, 0.0, 1.0]])
    # generate_masks([mesh_sphere], [0], [pose], K, 640, 480)
