import os
import trimesh
import numpy as np


meshpath = '../MESH/'
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
    print("Mesh diameter (to input in the .yml config file):","%.2f" % diameter)