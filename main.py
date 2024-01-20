import math
import igl
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="bunny.obj")
parser.add_argument("--width", type=int, default=1440, help="Width of off screen framebuffer")
parser.add_argument("--height", type=int, default=720, help="Height of off screen framebuffer")
parser.add_argument("--px", type=int, default=10, help="Size of pixel in on screen framebuffer")
parser.add_argument("--test", type=int, default=1, help="run a numbered unit test")
args = parser.parse_args()

ti.init(arch=ti.cpu) # can also use ti.gpu
px = args.px # Size of pixel in on screen framebuffer
width, height = args.width//px, args.height//px  # Size of off screen framebuffer
pix = np.zeros((width, height, 3), dtype=np.float32)
depth = np.zeros((width, height, 1), dtype=np.float32)
pixti = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width*px, height*px))
V, _, N, T, _, TN = igl.read_obj(args.file) #read mesh with normals

@ti.kernel
# copy pixels from small framebuffer to large framebuffer
def copy_pixels():
    for i, j in pixels:
        if px<2 or (tm.mod(i,px)!=0 and tm.mod(j,px)!=0):
            pixels[i,j] = pixti[i//px,j//px]

def draw_bounding_boxes(Vt, T, pixti, width, height):
    for t in T:
        v1, v2, v3 = Vt[t[0]], Vt[t[1]], Vt[t[2]]

        min_x = max(0, int(min(v1[0], v2[0], v3[0])))
        max_x = min(width, int(max(v1[0], v2[0], v3[0])))
        min_y = max(0, int(min(v1[1], v2[1], v3[1])))
        max_y = min(height, int(max(v1[1], v2[1], v3[1])))

        color = (random.random(), random.random(), random.random())
        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                pixti[i, j] = color

gui = ti.GUI("Rasterizer", res=(width*px, height*px))
t = 0   # time step for time varying transformaitons
translate = np.array([ width/2,height/2,0 ]) # translate to center of window
scale = 200/px*np.eye(3) # scale to fit in the window

while gui.running:
    pix.fill(0) # clear pixel buffer
    depth.fill(-math.inf) # clear depth buffer
    #time varying transformation
    c,s = math.cos(1.2*t), math.sin(1.2*t)
    Ry = np.array([[c, 0, s], [0, 1, 0],[-s, 0, c]])
    c,s = math.cos(t), math.sin(t)
    Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c,s = math.cos(1.8*t), math.sin(1.8*t)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    Vt = (scale @ Ry @ Rx @ Rz @ V.T).T
    Vt = Vt + translate
    Nt = (Ry @ Rx @ Rz @ N.T).T

    # draw!

    pixti.from_numpy(pix)
    if args.test == 1:
        draw_bounding_boxes(Vt, T, pixti, width, height)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += 0.001