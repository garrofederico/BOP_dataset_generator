import os
import sys
sys.path.append(os.path.abspath('.'))
import random
import json
from utils import *
import shutil

import trimesh

if __name__ == "__main__":
    # outdir = '/data/vespa_syn_0327/'
    # meshfile = '/data/vespa_syn_0327/models/obj_000001.ply'
    # ms = trimesh.load(meshfile)
    # bbox3d = ms.bounding_box_oriented.vertices

    # with open(outdir + 'vespa_bbox.json', 'w') as outfile:
    #     json.dump(bbox3d[None].tolist(), outfile, indent=4)

    # outdir = '/home/yhu/data/yy/vespa_20200911_hu_out/'
    # meshfile = '/home/yhu/data/vespa/models/obj_000001.ply'
    # outdir = "/data/occ_linemod_custom/"
    # meshPath = "/data/occ_linemod_custom/models/"
    outdir = ""

    meshPath = "../MESH/"

    meshes, objID_2_clsID = load_bop_meshes(meshPath)
    bbox3ds = []
    for mf in meshes:
        ms = trimesh.load(mf)
        bb = ms.bounding_box_oriented.vertices
        # bb = ms.bounding_box.vertices
        bbox3ds.append(bb.tolist())

    with open(outdir + 'linemod_bbox.json', 'w') as outfile:
        json.dump(bbox3ds, outfile, indent=4)