# Author Jerry Xia
# McGill ID 260917329

import math
import igl
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="bunnylowres.obj")
parser.add_argument("--width", type=int, default=1440, help="Width of off screen framebuffer")
parser.add_argument("--height", type=int, default=720, help="Height of off screen framebuffer")
parser.add_argument("--px", type=int, default=10, help="Size of pixel in on screen framebuffer")
parser.add_argument("--test", type=int, default=3, help="run a numbered unit test")
args = parser.parse_args()

ti.init(arch=ti.cpu)  # can also use ti.gpu
px = args.px  # Size of pixel in on screen framebuffer
width, height = args.width // px, args.height // px  # Size of off screen framebuffer
pix = np.zeros((width, height, 3), dtype=np.float32)
depth = np.zeros((width, height, 1), dtype=np.float32)
pixti = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width * px, height * px))
V, _, N, T, _, TN = igl.read_obj(args.file)  # read mesh with normals


def compute_normals(V, T):
    N = np.zeros_like(V)
    for t in T:
        v1, v2, v3 = V[t[0]], V[t[1]], V[t[2]]
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)  # normalize the normal
        N[t[0]] += n
        N[t[1]] += n
        N[t[2]] += n
    N /= np.linalg.norm(N, axis=1, keepdims=True)  # normalize the normals
    return N


if N.size == 0:
    N = compute_normals(V, T)
    TN = T


@ti.kernel
# copy pixels from small framebuffer to large framebuffer
def copy_pixels():
    for i, j in pixels:
        if px < 2 or (tm.mod(i, px) != 0 and tm.mod(j, px) != 0):
            pixels[i, j] = pixti[i // px, j // px]


def draw_bounding_boxes(Vt, T, pixti, width, height):
    for t in T:
        v1, v2, v3 = Vt[t[0]], Vt[t[1]], Vt[t[2]]

        min_x = max(0, int(min(v1[0], v2[0], v3[0])))
        max_x = min(width, int(max(v1[0], v2[0], v3[0])))
        min_y = max(0, int(min(v1[1], v2[1], v3[1])))
        max_y = min(height, int(max(v1[1], v2[1], v3[1])))

        color = (random.random(), random.random(), random.random())
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                pixti[i, j] = color


def draw_triangles_barycentric(Vt, T, pixti, width, height):
    epsilon = 1e-8
    for t in T:
        v1, v2, v3 = Vt[t[0]], Vt[t[1]], Vt[t[2]]

        min_x = max(0, int(min(v1[0], v2[0], v3[0])))
        max_x = min(width, int(max(v1[0], v2[0], v3[0])))
        min_y = max(0, int(min(v1[1], v2[1], v3[1])))
        max_y = min(height, int(max(v1[1], v2[1], v3[1])))

        # setup
        q0, q1, q2 = (1, 0, 0), (0, 1, 0), (0, 0, 1)  # barycentric coordinates at the vertices
        det = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]) + epsilon
        cx = [(q1[i] - q0[i]) * (v3[1] - v1[1]) - (q2[i] - q0[i]) * (v2[1] - v1[1]) for i in range(3)] / det
        cy = [(q2[i] - q0[i]) * (v2[0] - v1[0]) - (q1[i] - q0[i]) * (v3[0] - v1[0]) for i in range(3)] / det
        qRow = [cx[i] * (min_x - v1[0]) + cy[i] * (min_y - v1[1]) + q0[i] for i in range(3)]

        # traversal
        for y in range(min_y, max_y + 1):
            qPix = list(qRow)  # copy qRow to qPix
            for x in range(min_x, max_x + 1):
                if all(q > 0 for q in qPix):  # if all barycentric coordinates are greater than 0
                    z = qPix[0] * v1[2] + qPix[1] * v2[2] + qPix[2] * v3[2]  # interpolate z
                    if z > depth[x, y]:
                        depth[x, y] = z
                        color = (qPix[0], qPix[1], qPix[2])  # use barycentric coordinates as color
                        pixti[x, y] = color
                for i in range(3):
                    qPix[i] += cx[i]  # increment qPix by cx for the next pixel in the row
            for i in range(3):
                qRow[i] += cy[i]  # increment qRow by cy for the next row


def draw_normalized(Vt, T, Nt, TN, pixti, width, height):
    epsilon = 1e-8
    for index in range(len(T)):
        t = T[index]
        v1, v2, v3 = Vt[t[0]], Vt[t[1]], Vt[t[2]]
        n1, n2, n3 = Nt[TN[index][0]], Nt[TN[index][0]], Nt[TN[index][0]]

        min_x = max(0, int(min(v1[0], v2[0], v3[0])))
        max_x = min(width, int(max(v1[0], v2[0], v3[0])))
        min_y = max(0, int(min(v1[1], v2[1], v3[1])))
        max_y = min(height, int(max(v1[1], v2[1], v3[1])))

        # setup
        q0, q1, q2 = (1, 0, 0), (0, 1, 0), (0, 0, 1)  # barycentric coordinates at the vertices
        det = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]) + epsilon
        cx = [(q1[i] - q0[i]) * (v3[1] - v1[1]) - (q2[i] - q0[i]) * (v2[1] - v1[1]) for i in range(3)] / det
        cy = [(q2[i] - q0[i]) * (v2[0] - v1[0]) - (q1[i] - q0[i]) * (v3[0] - v1[0]) for i in range(3)] / det
        qRow = [cx[i] * (min_x - v1[0]) + cy[i] * (min_y - v1[1]) + q0[i] for i in range(3)]

        # traversal
        for y in range(min_y, max_y + 1):
            qPix = list(qRow)  # copy qRow to qPix
            for x in range(min_x, max_x + 1):
                if all(q > 0 for q in qPix):  # if all barycentric coordinates are greater than 0
                    z = qPix[0] * v1[2] + qPix[1] * v2[2] + qPix[2] * v3[2]  # interpolate z
                    n = qPix[0] * n1[2] + qPix[1] * n2[2] + qPix[2] * n3[2]
                    if z > depth[x, y]:
                        depth[x, y] = z
                        grey = max(0, n)
                        pixti[x, y] = (grey, grey, grey)
                for i in range(3):
                    qPix[i] += cx[i]  # increment qPix by cx for the next pixel in the row
            for i in range(3):
                qRow[i] += cy[i]  # increment qRow by cy for the next row


gui = ti.GUI("Rasterizer", res=(width * px, height * px))
t = 0  # time step for time varying transformations
translate = np.array([width / 2, height / 2, 0])  # translate to center of window
scale = 200 / px * np.eye(3)  # scale to fit in the window

while gui.running:
    pix.fill(0)  # clear pixel buffer
    depth.fill(-math.inf)  # clear depth buffer
    # time varying transformation
    c, s = math.cos(1.2 * t), math.sin(1.2 * t)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = math.cos(t), math.sin(t)
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c, s = math.cos(1.8 * t), math.sin(1.8 * t)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    Vt = (scale @ Ry @ Rx @ Rz @ V.T).T
    Vt = Vt + translate
    Nt = (Ry @ Rx @ Rz @ N.T).T

    # draw!

    pixti.from_numpy(pix)
    if args.test == 1:
        draw_bounding_boxes(Vt, T, pixti, width, height)
    if args.test == 2:
        draw_triangles_barycentric(Vt, T, pixti, width, height)
    else:
        draw_normalized(Vt, T, Nt, TN, pixti, width, height)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += 0.001
