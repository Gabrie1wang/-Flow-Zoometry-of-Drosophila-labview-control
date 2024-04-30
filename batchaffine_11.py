#!/usr/bin/env python
# coding: utf-8

# In[2]:


#################################################################
#Set various parameters and exectue run all in jupyter notebook
#################################################################

#Directory with files
indir = r"R:\20220803_beads_backwards\h5_raw"
#Directory for files to go
outdir = r"R:\20220803_beads_backwards\h5_raw\1.1umvox_backwards_ds3x"
#type of file (pcoraw,dcimg,h5,tiff)
ftype = 'h5'
#magnification
mag = 5.5
#Light Sheet Angle
theta = 27
#Stage Translation Speed in um/frame
trans_speed = 1
#Stage Translation Direction
dir = 'Backward'
#Voxel size (x) in um
grid_x = 1.1
#Voxel size (y) in um
grid_y = 1.1
#Voxel size (z) in um
grid_z = 1.1
#Downsample factor
factor = 3

if ftype.lower() == 'pcoraw':
    ftype = 0
elif ftype.lower() == 'dcimg':
    ftype = 1
elif ftype.lower() == 'h5':
    ftype = 3
elif ftype.lower() == 'tiff':
    ftype = 2

if dir.lower() == 'forward':
    sgn = 1
else:
    sgn = 0
# print(sgn)

# import _hx_core # Avizo package
import numpy as np
import scipy
from scipy import ndimage
from scipy import signal
import cupy as cp
import cupyx
from cupyx.scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sys, os
from tkinter import filedialog
import tifffile
import dcimg
import h5py
# import io
# import imageio

def find_files(directory, pattern):
    """
    Find files with provided pattern in provided directory

    Args:
        directory: directory where to find the files
        pattern: pattern of interest in the file names

    Returns:
        Files of interest
    """
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.abspath(os.path.join(root, filename)).replace("\\","/"))
    return matches

def getTransformMatrix(pix, mag, theta, grid_obs_z, grid_x, grid_y, grid_z):
    #calculate the pixel size
    grid_obs_xy = pix/mag
    theta_rad = np.pi*theta/180
    #calculate transform matrix for Affine transformation
    Mnormalize = np.array([[grid_obs_xy,0,0],[0,grid_obs_xy,0],[0,0,grid_obs_z]])
    Mshear = np.array([[np.sin(theta_rad),0,0],[0,1,0],[np.cos(theta_rad),0,1]])
    Mscale = np.array([[1/grid_x,0,0],[0,1/grid_y,0],[0,0,1/grid_z]])
    #Maffine_inv is matrix that transforms points
    Maffine_inv = np.dot(Mscale, np.dot(Mshear, Mnormalize))
    #Maffine is matrix that transforms coordinates
    Maffine = np.linalg.inv(Maffine_inv)
    # Maffine = np.array([[grid_xy/grid_obs_xy/np.sin(theta_rad),0,grid_xy*np.cos(theta_rad)/grid_obs_z/np.sin(theta_rad)],\
    #                         [0,grid_xy/grid_obs_xy,0],\
    #                             [0,0,grid_z/grid_obs_z]])
    Maffine_inv = np.linalg.inv(Maffine)
    # print(Maffine_inv,Maffine)
    return Maffine_inv, Maffine


def getAffinedSize(Msize, Maffine): #calculate the image size after Affine transformation
    x0 = Msize[0]
    y0 = Msize[1]
    z0 = Msize[2]
    a = np.empty((3, 8))
    a[:,0] = np.dot(Maffine, np.array([0,0,0]))
    a[:,1] = np.dot(Maffine, np.array([x0,0,0]))
    a[:,2] = np.dot(Maffine, np.array([0,y0,0]))
    a[:,3] = np.dot(Maffine, np.array([0,0,z0]))
    a[:,4] = np.dot(Maffine, np.array([x0,y0,0]))
    a[:,5] = np.dot(Maffine, np.array([0,y0,z0]))
    a[:,6] = np.dot(Maffine, np.array([x0,0,z0]))
    a[:,7] = np.dot(Maffine, np.array([x0,y0,z0]))
    # print(a)
    # print(a[:,0])
    # print(a[0,:])
    xmax = max(a[0,:])
    ymax = max(a[1,:])
    zmax = max(a[2,:])
    xmin = min(a[0,:])
    ymin = min(a[1,:])
    zmin = min(a[2,:])
    return np.array([xmax,ymax,zmax]), np.array([xmin,ymin,zmin])

def Affine(fld, filetype, mag = 6.5, theta = 30, grid_x = 3, grid_y = 1, grid_z = 3, grid_obs_z = 10, sgn = 1):
    #experimental parameters
    pix = 6.5 # sCMOS pixel size in micrometer
    # filetype: 0: pcoraw, 1: dcimg, 2: tiff, 3: hdf5
    # mag: magnification
    # theta: light sheet angle in degree
    # grid_x, grid_y, grid_z: target grid size in micrometer
    # grid_obs_z: scanning step size in micrometer
    # sgn:  0: image moves from right to left, 1: image moves from left to right
 
    if filetype == 0:
    #input .pcoraw file
        FITC = tifffile.imread(fld)
        #axis order is changed to (x,y,z) by transpose
        #axis order is (z,y,x) when loaded
        FITC = cp.asarray(FITC, dtype='float16')
        FITC = cp.transpose(FITC)

    if filetype == 2:
    #input .tiff file which has been converted from DCIMG
        FITC = tifffile.imread(fld)
        #axis order is changed to (x,y,z) by transpose
        #axis order is (z,y,x) when loaded
        FITC = cp.asarray(FITC, dtype='float16')
        FITC = cp.transpose(FITC)
        #FITC = cp.rot90(FITC, -1, (0,1)) #this will rotate altered tiff files

    if filetype == 3:
    #input .h5 HDF5 file which has been converted from DCIMG
        hf = h5py.File(fld)                    
        key = list(hf.keys())[0]
        object2 = hf[key]
        object = object2[:,:,:]
        FITC = object
        #axis order is changed to (x,y,z) by transpose
        #axis order is (z,y,x) when loaded
        FITC = cp.asarray(FITC, dtype='float16')
        FITC = cp.transpose(FITC)
        FITC = cp.rot90(FITC, -1, (0,1)) #this will rotate altered h5 files    
    
    else:
    #input .dcimg file
        dcimgFile = dcimg.DCIMGFile(fld)
        FITC = dcimgFile[:,:,:]

        dcimgFile = dcimg.DCIMGFile(fld)
        object = dcimgFile[:,:,:]
        object = np.transpose(object)

        FITC = object
        for i in range(object.shape[2]):
            FITC[:,:,i] = np.roll(object[:,:,i], 8*i, 0)

        #axis order is changed to (x,y,z) by transpose
        #axis order is (z,y,x) when loaded
        FITC = cp.asarray(FITC, dtype='float16')
        FITC = cp.rot90(FITC, -1, (0,1))

    

    #flip the image stack if the motion of stage is opposite
    if sgn == 1:
        FITC = cp.flip(FITC,axis=2)

    #calculate matrices for Affine transformation
    Maffine_inv, Maffine = getTransformMatrix(pix, mag, theta, grid_obs_z, grid_x, grid_y, grid_z)

    #calculate the size of the transformed image
    Vmax, Vmin = getAffinedSize(FITC.shape, Maffine_inv)
    # print(FITC.shape)
    Vsize = Vmax - Vmin
    Vsize = Vsize.astype('uint16')
    # print(Vsize)

    #do Affine transformation
    start = time.time()
    FITC_transformed = cupyx.scipy.ndimage.affine_transform(FITC, cp.asarray(Maffine), order=1,                                     output_shape=tuple(Vsize), offset = [Vmin[0],Vmin[1],Vmin[2]])
    elapsed_time = time.time() - start
    print ("elapsed_time (GPU):{0}".format(elapsed_time) + "[sec]")
    #print(FITC_transformed.shape)

    #Convert from float to uint16
    FITC_transformed_np = cp.asnumpy(FITC_transformed)
    FITC_transformed_np = FITC_transformed_np.astype('uint16')

    return FITC_transformed_np, Vsize, grid_x, grid_y, grid_z

def Downscale(img,Vsize,factor = 4):
        
        
        # mempool = cupy.get_default_memory_pool()
        # pinned_mempool = cupy.get_default_pinned_memory_pool()
        r = 1/factor
        output_shape = tuple((Vsize*r).astype(int))
        A = cp.asarray(np.eye(3)/r)
        FITC_transformed_np = ndimage.affine_transform(img, A, order=1, output_shape=output_shape)
        FITC_transformed_np = cp.asnumpy(FITC_transformed_np).astype('uint16')
        

        return FITC_transformed_np

def BatchAffine(indir,outdir,mag,theta,grid_x,grid_y,grid_z,trans_speed,sgn):
    import re
    import imageio
    from numba import cuda
    import cupy
    # factor = 4
    # import h5py
    #sets mempool for VRAM clearing
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    dirname = indir
    #Dump VRAM
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    # print(indir,outdir,mag,theta,grid_x,grid_y,grid_z,trans_speed,sgn)
    # if __name__ == '__main__':
    #set file extention
    if ftype == 0:
        pat_am = re.compile(r"\.pcoraw$")
    elif ftype == 1:
        pat_am = re.compile(r"\.dcimg$")
    elif ftype == 2:
        pat_am = re.compile(r"\.tif$")
    else:
        pat_am = re.compile(r"\.h5$")
    # print(pat_am)

    #start looping through all files in dirname
    for x in os.listdir(dirname):
        # print(x)
        mo = pat_am.search(x)
        if mo is None:
            continue
        # print(x)
        filename = x[:-3]
        # print(filename)
        #string for input/output files
        fp_input = f"{dirname}/{x}"
        fp_output = f"{outdir}/{filename}.h5"
        fp_output_resize = f"{outdir}/ds{factor}x_{filename}.h5"
        # fp_output_resize = f"{outdir}/Test_{x}"
        # Affine Transform
        print(fp_input)
        FITC_transformed_np, Vsize, grid_x_out, grid_y_out, grid_z_out = Affine(fp_input,ftype,mag,theta,grid_x,grid_y,grid_z,trans_speed,sgn)
        FITC_transformed_np = cp.asarray(FITC_transformed_np, dtype='float16')
        #Downscale 
        FITC_transformed_np = Downscale(FITC_transformed_np,Vsize,factor)
        # FITC_transformed_np = cp.asnumpy(FITC_transformed_np).astype('uint16')

        bounding_box = ((0.0, 0.0, 0.0), (grid_x_out * Vsize[0] * 1e-6, grid_y_out * Vsize[1] * 1e-6, grid_z_out * Vsize[2] * 1e-6))
        # Save new image
        hf = h5py.File(fp_output_resize,'w')
        hf.create_dataset('dataset_1',data = FITC_transformed_np)
        hf.create_dataset('bounding_box',data = bounding_box)
        hf.close()

        #Dump VRAM
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
starttime = time.time()
BatchAffine(indir,outdir,mag,theta,grid_x,grid_y,grid_z,trans_speed,sgn)
endtime = time.time()
print(endtime - starttime)

# fld = r"C:\Users\jgare\Documents\FINDS\TestOutput\Test_63148  7.dcimg-003.h5"
# hf = h5py.File(fld)                    
# key = list(hf.keys())[0]
# print(key)
# object2 = hf[key]
# object = object2[:,:,:]
# FITC = object
# #axis order is changed to (x,y,z) by transpose
# #axis order is (z,y,x) when loaded
# FITC = cp.asarray(FITC, dtype='float32')



# infile = f"{indir}/63148  7.dcimg.h5"
# print(infile[:-9])


# In[ ]:





# In[ ]:




