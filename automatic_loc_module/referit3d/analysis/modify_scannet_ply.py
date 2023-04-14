import math
import os, sys, argparse
import inspect
import json
import numpy as np
from plyfile import PlyData, PlyElement
import random


def visualize(mesh_file, output_file):
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
    print("plydata = ", plydata)
    print("plydata = ", len(plydata['vertex']['red']))
    print("plydata = ", type(plydata['vertex']['x']))
    x = np.memmap('newfile.dat', dtype='float32', mode='w+', shape=(int(len(plydata['vertex']['x']) / 2),))
    idx = random.sample(list(range(0, len(plydata['vertex']['x']), 1)), int(len(plydata['vertex']['x']) / 2))
    vertex = []
    for i in idx:
        vertex.append((plydata['vertex']['x'][i], plydata['vertex']['y'][i], plydata['vertex']['z'][i],
                       plydata['vertex']['nx'][i], plydata['vertex']['ny'][i], plydata['vertex']['nz'][i],
                       plydata['vertex']['red'][i], plydata['vertex']['green'][i], plydata['vertex']['blue'][i],
                       plydata['vertex']['alpha'][i]))
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write("dw_sample_scene_0000.ply")


if __name__ == '__main__':
    #visualize(mesh_file="/home/eslam/eslam_pc/pc/scene0000_00_vh_clean.ply",
    #          output_file="/home/eslam/eslam_pc/pc/scene0000_00_vh_clean_dw_0.5.ply")
    visualize(mesh_file="/home/eslam/eslam_pc/pc/dw_sample_scene_0000.ply",
              output_file="/home/eslam/eslam_pc/pc/scene0000_00_vh_clean_dw_0.5.ply")
