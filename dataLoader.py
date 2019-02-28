import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import os
import scipy.ndimage as ndimage


class BatchLoader(Dataset):
    def __init__(self, dataRoot, imSize = 256, isRandom=True, phase='TRAIN', rseed = None, shapeRange = [0, 500], cascade = None, isPoint=False):
        self.dataRoot = dataRoot
        self.imSize = imSize
        self.phase = phase.upper()

        shapeList = glob.glob(osp.join(dataRoot, 'Synth/Shape__*') )
        shapeList = sorted(shapeList)


        self.albedoList = []
        self.F0List = []
        for shape in shapeList:
            albedoNames = glob.glob(osp.join(shape, '*albedo.png') )
            self.albedoList = self.albedoList + albedoNames

        self.isPoint = isPoint
        if rseed is not None:
            random.seed(rseed)

        # BRDF parameter
        self.normalList = [x.replace('albedo', 'normal') for x in self.albedoList]
        self.roughList = [x.replace('albedo', 'rough') for x in self.albedoList]
        self.segList = [x.replace('albedo', 'seg') for x in self.albedoList]

        # Rendered Image
        self.imPList = [x.replace('albedo', 'imgPoint') for x in self.albedoList]
        self.imEList = [x.replace('albedo', 'imgEnv') for x in self.albedoList]

        # Geometry
        self.depthList = [x.replace('albedo', 'depth').replace('png', 'dat') for x in self.albedoList]

        # Environment Map
        self.SHList = []
        self.nameList = []
        for x in self.albedoList:
            suffix = '/'.join(x.split('/')[0:-1])
            fileName = x.split('/')[-1]
            fileName = fileName.split('_')
            self.SHList.append(osp.join(suffix, '_'.join(fileName[0:2]) + '.npy'  ) )
            self.nameList.append(osp.join(suffix, '_'.join(fileName[0:3]) ) )

        # Permute the image list
        self.count = len(self.albedoList)
        self.perm = list(range(self.count) )
        if isRandom:
            random.shuffle(self.perm)

        # Real Images
        #print(osp.join(osp.join(dataRoot,'Real/RealWordImages'), '*_mask.jpg'))
        self.realImageMaskNames = glob.glob(osp.join(osp.join(dataRoot,'Real/RealWorldImages'), '*_mask.jpg'))
        self.realImageNames = [x.replace('_mask', '') for x in self.realImageMaskNames]
        print (len(self.realImageNames),len(self.perm))
        self.permReal = np.random.randint(0,len(self.realImageNames),len(self.perm))

    def __len__(self):
        return len(self.perm)


    def __getitem__(self, ind):
        # Read segmentation
        seg = 0.5 * self.loadImage(self.segList[self.perm[ind] ] ) + 0.5
        seg = (seg[0, :, :] > 0.999999).astype(dtype = np.int)
        seg = ndimage.binary_erosion(seg, structure = np.ones( (2, 2) ) ).astype(dtype = np.float32 )
        seg = seg[np.newaxis, :, :]

        # Read albedo
        albedo = self.loadImage(self.albedoList[self.perm[ind] ] )
        albedo = albedo * seg

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[self.perm[ind] ] )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), 1e-5) )[np.newaxis, :]
        normal = normal * seg

        # Read roughness
        rough = self.loadImage(self.roughList[self.perm[ind] ] )[0:1, :, :]
        rough = (rough * seg)

        # Read rendered images
        imP = self.loadImage(self.imPList[self.perm[ind] ], isGama = True)
        imP = imP * seg
        imE = self.loadImage(self.imEList[self.perm[ind] ], isGama = True)
        imEbg = imE.copy()
        imE = imE * seg

        with open(self.depthList[self.perm[ind] ], 'rb') as f:
            byte = f.read()
            if len(byte) == 256 * 256 * 3 * 4:
                depth = np.array(struct.unpack(str(256*256*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([256, 256, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1] )
                depth = depth * seg
            elif len(byte) == 512 * 512 * 3 * 4:
                depth = np.array(struct.unpack(str(512*512*3)+'f', byte), dtype=np.float32)
                depth = depth.reshape([512, 512, 3])[:, :, 0:1]
                depth = depth.transpose([2, 0, 1] )
                depth = depth * seg


        if not os.path.isfile(self.SHList[self.perm[ind] ] ):
            #print('Fail to load {0}'.format(self.SHList[self.perm[ind] ] ) )
            SH = np.zeros([3, 9], dtype=np.float32)
        else:
            SH = np.load(self.SHList[self.perm[ind] ]).transpose([1, 0] )[:, 0:9]
            SH = SH.astype(np.float32)[::-1, :]
        name = self.nameList[self.perm[ind] ]

        # Scale the input
        scalePoint = 1.7
        imP = (imP + 1) * scalePoint - 1

        # Scale the Environment
        scaleEnv = 0.5
        imE = (imE + 1) * scaleEnv - 1
        imEbg = (imEbg + 1) * scaleEnv - 1
        SH = SH * scaleEnv

        imP = np.clip(imP, -1, 1)
        imE = np.clip(imE, -1, 1)

        imReal = self.loadImage(self.realImageNames[self.permReal[ind] ], isGama = True)
        segReal = 0.5 * self.loadImage(self.realImageMaskNames[self.permReal[ind] ] ) + 0.5
        #print (segReal.shape)
        segReal = (segReal[:, :, 0] > 0.999999).astype(dtype = np.int)
        segReal = ndimage.binary_erosion(segReal, structure = np.ones( (2, 2) ) ).astype(dtype = np.float32 )
        segReal = segReal[np.newaxis, :, :]


        batchDict = {'albedo': albedo,
                     'normal': normal,
                     'rough': rough,
                     'depth': depth,
                     'seg': seg,
                     'imP': imP,
                     'imE':  imE,
                     'imEbg': imEbg,
                     'SH': SH,
                     'name': name,
                     'albedoName': self.albedoList[self.perm[ind] ],
                     'realImage': imReal,
                     'realImageMask': segReal}



        return batchDict


    def loadImage(self, imName, isGama = False):
        if not os.path.isfile(imName):
            print('Fail to load {0}'.format(imName) )
            im = np.zeros([3, self.imSize, self.imSize], dtype=np.float32)
            return im

        im = Image.open(imName)
        im = self.imResize(im)
        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1])
        return im

    def imResize(self, im):
        w0, h0 = im.size
        assert( (w0 == h0) )
        im = im.resize((self.imSize, self.imSize), Image.ANTIALIAS)
        return im

    def loadNpy(self, name):
        data = np.load(name)
        return data
