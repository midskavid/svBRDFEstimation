import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import numpy as np
from torch.autograd import Variable
import argparse
import os
import models
import torchvision.utils as vutils
import torch.nn as nn
from PIL import Image
import glob
import os.path as osp
import scipy.ndimage as ndimage
import struct
import utils


parser = argparse.ArgumentParser()
# The locationi of testing set
parser.add_argument('--dataRoot', default='realImages', help='path to real image distorted by water')
parser.add_argument('--modelRootInit', default = None, help = 'the directory where the initialization trained model is save')
parser.add_argument('--modelRootsRefine', nargs='+', default=[None, None], help='the directory where the refine models are saved')
parser.add_argument('--modelRootGlob', default = None, help = 'the directory where the global illumination model is saved')
parser.add_argument('--epochIdInit', type=int, default = 14, help = 'the training epoch of the initial network')
parser.add_argument('--epochIdsRefine', nargs = '+', type=int, default = [7, 5], help='the training epoch of the refine network')
parser.add_argument('--epochIdGlob', type=int, default=17, help='the traing epoch of the global illuminationn prediction network')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic testing setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for testing')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0, 1], help='the gpus used for testing network')
# The testing weight
parser.add_argument('--cascadeLevel', type=int, default=2, help='cascade level')
# Refine input mode
parser.add_argument('--renderMode', type=int, default=0, help='Define the render type, \
        0 means render with direct lighting, 1 plus environment map, 2 plus global illumination')
parser.add_argument('--refineInputMode', type=int, default=0, help='Define the type of input for refinement, \
        0 means no feedback, 1 means error feedback, 2 means gradient feedback')
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.modelRootInit is None:
    opt.modelRootInit = 'check_initEnv_cascade0'

if len(opt.modelRootsRefine) != opt.cascadeLevel or opt.modelRootsRefine[0] is None:
    opt.modelRootsRefine = []
    for n in range(1, opt.cascadeLevel+1):
        root = 'check_initEnv'
        root += '_render{0}'.format(opt.renderMode)
        root += '_refine{0}'.format(opt.refineInputMode)
        root += '_cascade{0}'.format(n)
        opt.modelRootsRefine.append(root)

if opt.modelRootGlob is None:
    opt.modelRootGlob = 'check_globalIllumination'

if opt.experiment is None:
    opt.experiment = opt.dataRoot + '_results'
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )


opt.seed = 0
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
# initalize tensors
segBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imBgBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )

# Initial Network
encoderInit = nn.DataParallel(models.encoderInitial(), device_ids = opt.deviceIds)
albedoInit = nn.DataParallel(models.decoderInitial(mode=0), device_ids = opt.deviceIds)
normalInit = nn.DataParallel(models.decoderInitial(mode=1), device_ids = opt.deviceIds)
roughInit = nn.DataParallel(models.decoderInitial(mode=2), device_ids = opt.deviceIds)
depthInit = nn.DataParallel(models.decoderInitial(mode=3), device_ids = opt.deviceIds)
envInit = nn.DataParallel(models.envmapInitial(), device_ids = opt.deviceIds)

# Refine Network
encoderRefs, albedoRefs = [], []
normalRefs, roughRefs = [], []
depthRefs, envRefs = [], []
for n in range(0, opt.cascadeLevel):
    if opt.refineInputMode == 0:
        encoderRefs.append( nn.DataParallel(models.refineEncoder(), device_ids = opt.deviceIds ) )
    elif opt.refineInputMode == 1:
        encoderRefs.append( nn.DataParallel(models.refineEncoder_error(), device_ids = opt.deviceIds ) )
    else:
        raise ValueError('The value of opt refineInputMode is wrong')
    albedoRefs.append( nn.DataParallel(models.refineDecoder(mode=0), device_ids = opt.deviceIds ) )
    normalRefs.append( nn.DataParallel(models.refineDecoder(mode=1), device_ids = opt.deviceIds) )
    roughRefs.append( nn.DataParallel(models.refineDecoder(mode=2), device_ids = opt.deviceIds) )
    depthRefs.append( nn.DataParallel(models.refineDecoder(mode=3), device_ids=opt.deviceIds) )
    envRefs.append( nn.DataParallel(models.refineEnvDecoder(), device_ids = opt.deviceIds) )

renderLayer = models.renderingLayer(gpuId = opt.gpuId, isCuda = opt.cuda)

# Global illumination
globIllu1to2 = models.globalIllumination()
globIllu2to3 = models.globalIllumination()
#########################################

#########################################
# Load the trained model
encoderInit.load_state_dict(torch.load('{0}/encoderInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit), \
        map_location = lambda storage, loc:storage ) )
encoderInit = encoderInit.eval()
albedoInit.load_state_dict(torch.load('{0}/albedoInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit), \
        map_location = lambda storage, loc:storage ) )
albedoInit = albedoInit.eval()
normalInit.load_state_dict(torch.load('{0}/normalInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit), \
        map_location = lambda storage, loc:storage ) )
normalInit = normalInit.eval()
roughInit.load_state_dict(torch.load('{0}/roughInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit), \
        map_location = lambda storage, loc:storage ) )
roughInit = roughInit.eval()
depthInit.load_state_dict(torch.load('{0}/depthInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit), \
        map_location = lambda storage, loc:storage ) )
depthInit = depthInit.eval()
envInit.load_state_dict(torch.load('{0}/envInit_{1}.pth'.format(opt.modelRootInit, opt.epochIdInit), \
        map_location = lambda storage, loc:storage ) )
envInit = envInit.eval()

globIllu1to2.load_state_dict(torch.load('{0}/globIllu1to2_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob), \
        map_location = lambda storage, loc:storage ) )
globIllu1to2 = nn.DataParallel(globIllu1to2.eval(), device_ids = opt.deviceIds )
globIllu2to3.load_state_dict(torch.load('{0}/globIllu2to3_{1}.pth'.format(opt.modelRootGlob, opt.epochIdGlob), \
        map_location = lambda storage, loc:storage ) )
globIllu2to3 = nn.DataParallel(globIllu2to3.eval(), device_ids = opt.deviceIds )
for n in range(0, opt.cascadeLevel):
    encoderRefs[n].load_state_dict(torch.load('{0}/encoderRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n, opt.epochIdsRefine[n]), \
            map_location = lambda storage, loc:storage) )
    encoderRefs[n] = encoderRefs[n].eval()
    albedoRefs[n].load_state_dict(torch.load('{0}/albedoRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n, opt.epochIdsRefine[n]), \
            map_location = lambda storage, loc:storage) )
    albedoRefs[n] = albedoRefs[n].eval()
    normalRefs[n].load_state_dict(torch.load('{0}/normalRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n, opt.epochIdsRefine[n]), \
            map_location = lambda storage, loc:storage) )
    normalRefs[n] = normalRefs[n].eval()
    roughRefs[n].load_state_dict(torch.load('{0}/roughRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n, opt.epochIdsRefine[n]), \
            map_location = lambda storage, loc:storage) )
    roughRefs[n] = roughRefs[n].eval()
    depthRefs[n].load_state_dict(torch.load('{0}/depthRefs{1}_{2}.pth'.format(opt.modelRootsRefine[n], n, opt.epochIdsRefine[n]), \
            map_location = lambda storage, loc:storage) )
    depthRefs[n] = depthRefs[n].eval()
#########################################


##############  ######################
# Send things into GPU
if opt.cuda:
    segBatch = segBatch.cuda(opt.gpuId)
    imBatch = imBatch.cuda(opt.gpuId)
    imBgBatch = imBgBatch.cuda(opt.gpuId)

    encoderInit = encoderInit.cuda(opt.gpuId)
    albedoInit = albedoInit.cuda(opt.gpuId)
    normalInit = normalInit.cuda(opt.gpuId)
    roughInit = roughInit.cuda(opt.gpuId)
    depthInit = depthInit.cuda(opt.gpuId)
    envInit = envInit.cuda(opt.gpuId)
    globIllu1to2 = globIllu1to2.cuda(opt.gpuId)
    globIllu2to3 = globIllu2to3.cuda(opt.gpuId)
    for n in range(0, opt.cascadeLevel):
        encoderRefs[n] = encoderRefs[n].cuda(opt.gpuId)
        albedoRefs[n] = albedoRefs[n].cuda(opt.gpuId)
        normalRefs[n] = normalRefs[n].cuda(opt.gpuId)
        roughRefs[n] = roughRefs[n].cuda(opt.gpuId)
        depthRefs[n] = depthRefs[n].cuda(opt.gpuId)
        envRefs[n] = envRefs[n].cuda(opt.gpuId)
####################################


####################################

imgNames = glob.glob(osp.join(opt.dataRoot, '*input.png') )
j = 0
for imgName in imgNames:
    j += 1

    # Read the image with background
    imBg = Image.open(imgName )
    imBg = np.asarray(imBg ).astype(np.float32)
    imBg = (imBg / 255.0) ** (2.2)
    imBg = (2*imBg - 1).transpose([2, 0, 1] )[np.newaxis, :, :, :]

    # Read the segmentation mask
    segName = imgName.replace('input', 'mask')
    seg = Image.open(segName )
    seg = np.asarray(seg ).astype(np.float32) / 255.0
    if seg.shape[2] > 1:
        seg = seg[:, :, 0]
    seg = (seg > 0.999).astype(dtype = np.int)
    seg = ndimage.binary_erosion(seg, structure = np.ones( (4,4) ) ).astype(dtype=np.float32)

    seg = seg[np.newaxis, np.newaxis, :, :]

    im = imBg * seg

    # Load data from cpu to gpu
    segBatch.data.resize_(seg.shape )
    segBatch.data.copy_(torch.from_numpy(seg ) )

    imBatch.data.resize_(im.shape )
    imBatch.data.copy_( torch.from_numpy(im ) )
    imBgBatch.data.resize_(imBg.shape )
    imBgBatch.data.copy_( torch.from_numpy(imBg ) )

    albedoPreds = []
    normalPreds = []
    roughPreds = []
    depthPreds = []
    SHPreds = []
    globalIllu1s = []
    globalIllu2s = []
    globalIllu3s = []
    renderedEnvs = []

    batchSize = imBgBatch.size(0)
    assert(batchSize == 1)

    # Initial Prediction
    inputInit = torch.cat([imBatch, imBgBatch, segBatch], dim=1)
    x1, x2, x3, x4, x5, x = encoderInit(inputInit)
    albedoPred = albedoInit(x1, x2, x3, x4, x5, x)*segBatch.expand_as(imBatch)
    normalPred = normalInit(x1, x2, x3, x4, x5, x)*segBatch.expand_as(imBatch)
    roughPred = roughInit(x1, x2, x3, x4, x5, x)*segBatch
    depthPred = depthInit(x1, x2, x3, x4, x5, x)*segBatch
    SHPred = envInit(x)

    globalIllu1 = renderLayer.forward(albedoPred, normalPred,
            roughPred, depthPred, segBatch)
    renderedEnv = renderLayer.forwardEnv(albedoPred,
            normalPred, roughPred, SHPred, segBatch)
    inputGlob2 = torch.cat([globalIllu1, albedoPred, \
            normalPred, roughPred, depthPred, segBatch], dim=1)
    globalIllu2 = globIllu1to2(inputGlob2).detach()
    inputGlob3 = torch.cat([globalIllu2, albedoPred, \
            normalPred, roughPred, depthPred, segBatch], dim=1)
    globalIllu3 = globIllu2to3(inputGlob3).detach()
    globalIllu2 = 0.5*(globalIllu2 + 1)
    globalIllu3 = 0.5*(globalIllu3 + 1)


    albedoPreds.append(albedoPred)
    normalPreds.append(normalPred)
    roughPreds.append(roughPred)
    depthPreds.append(depthPred)
    SHPreds.append(SHPred)
    globalIllu1s.append(globalIllu1)
    globalIllu2s.append(globalIllu2)
    globalIllu3s.append(globalIllu3)
    renderedEnvs.append(renderedEnv)


    # Refine the BRDF reconstruction
    for n in range(0, opt.cascadeLevel):
        albedoPred = (albedoPreds[n] * segBatch.expand_as(albedoPred) ).detach()
        normalPred = (normalPreds[n] * segBatch.expand_as(normalPred) ).detach()
        roughPred = (roughPreds[n] * segBatch.expand_as(roughPred) ).detach()
        depthPred = (depthPreds[n] * segBatch.expand_as(depthPred) ).detach()
        SHPred = SHPreds[n].detach()

        globalIllu1 = renderLayer.forward(albedoPred, normalPred,
                roughPred, depthPred, segBatch)

        if opt.renderMode == 0:
            renderedImg = globalIllu1
        elif opt.renderMode == 1:
            renderedImg = renderedEnv + globalIllu1
            renderedImg = torch.clamp(renderedImg, 0, 1)
        elif opt.renderMode == 2:
            renderedImg = renderedEnv + globalIllu1 + \
                    globalIllu2 + globalIllu3
        else:
            raise ValueError("The renderMode should be 0, 1 or 2")

        if opt.refineInputMode == 0:
            inputRefine = torch.cat([albedoPred, normalPred, roughPred, depthPred, segBatch, \
                    imBatch, imBgBatch], dim=1)
        elif opt.refineInputMode == 1:
            error = (renderedImg - 0.5*(imBatch + 1) ) * segBatch.expand_as(imBatch)
            inputRefine = torch.cat( [albedoPred, normalPred, roughPred, depthPred, segBatch, \
                    imBatch, imBgBatch, error], dim=1)
        else:
            raise ValueError("The refine mode should be 0, 1 or 2" )

        x1, x3 = encoderRefs[n](inputRefine.detach() )
        albedoPred = albedoRefs[n](x1, x3) * segBatch.expand_as(imBatch)
        normalPred = normalRefs[n](x1, x3) * segBatch.expand_as(imBatch)
        roughPred = roughRefs[n](x1, x3) * segBatch
        depthPred = depthRefs[n](x1, x3) * segBatch
        SHPred = envRefs[n](x3, SHPred)
        globalIllu1 = renderLayer.forward(albedoPred, normalPred,
                roughPred, depthPred, segBatch)

        globalIllu2 = globIllu1to2(torch.cat([globalIllu1, albedoPred, \
                normalPred, roughPred, depthPred, segBatch], dim=1) ).detach()
        globalIllu3 = globIllu2to3(torch.cat([globalIllu2, albedoPred, \
                normalPred, roughPred, depthPred, segBatch], dim=1) ).detach()
        globalIllu2 = 0.5 * (globalIllu2 + 1) * segBatch.expand_as(imBatch )
        globalIllu3 = 0.5 * (globalIllu3 + 1) * segBatch.expand_as(imBatch )

        albedoPreds.append(albedoPred)
        normalPreds.append(normalPred)
        roughPreds.append(roughPred)
        depthPreds.append(depthPred)
        SHPreds.append(SHPred)
        globalIllu1s.append(globalIllu1)
        globalIllu2s.append(globalIllu2)
        globalIllu3s.append(globalIllu3)


    # Save the ground truth and the input
    vutils.save_image( ( (0.5*(imBatch + 1)*segBatch.expand_as(imBatch))**(1.0/2.2) ).data,
            '{0}/{1}_im.png'.format(opt.experiment, j) )
    vutils.save_image( ( (0.5*(imBgBatch + 1) )**(1.0/2.2) ).data,
            '{0}/{1}_imBg.png'.format(opt.experiment, j) )

    # Save the predicted results
    for n in range(0, len(albedoPreds) ):
        vutils.save_image( ( 0.5*(albedoPreds[n] + 1)*segBatch.expand_as(albedoPreds[n]) ).data,
                '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
    for n in range(0, len(normalPreds) ):
        vutils.save_image( ( 0.5*(normalPreds[n] + 1)*segBatch.expand_as(normalPreds[n]) ).data,
                '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
    for n in range(0, len(roughPreds) ):
        vutils.save_image( ( 0.5*(roughPreds[n] + 1)*segBatch.expand_as(roughPreds[n]) ).data,
                '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )
    for n in range(0, len(depthPreds) ):
        depthOut = 1 / torch.clamp(depthPreds[n], 1e-6, 10) * segBatch.expand_as(depthPreds[n])
        vutils.save_image( ( depthOut * segBatch.expand_as(depthPreds[n]) ).data,
                '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )

        with open('{0}/{1}_depthPred_{2}.dat'.format(opt.experiment, j, n), 'wb' ) as fileOut:
            depthRaw = depthPreds[n]
            depthRaw = (depthRaw.data * segBatch.expand_as(depthRaw ).data ).cpu().numpy().flatten().tolist()
            var = struct.pack('%df' % (opt.imageSize * opt.imageSize), *depthRaw )
            fileOut.write(var)

    for n in range(0, len(renderedEnvs) ):
        vutils.save_image( ( ( renderedEnvs[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                '{0}/{1}_imEPred.png'.format(opt.experiment, j, n) )
    for n in range(0, len(globalIllu1s) ):
        vutils.save_image( ( ( globalIllu1s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                '{0}/{1}_imP1Pred_{2}.png'.format(opt.experiment, j, n) )
    for n in range(0, len(globalIllu2s) ):
        vutils.save_image( ( ( globalIllu2s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                '{0}/{1}_imP2Pred_{2}.png'.format(opt.experiment, j, n) )
    for n in range(0, len(globalIllu3s) ):
        vutils.save_image( ( ( globalIllu3s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                '{0}/{1}_imP3Pred_{2}.png'.format(opt.experiment, j, n) )
    for n in range(0, len(SHPreds) ):
        utils.visualizeSH('{0}/{1}_predSH.png'.format(opt.experiment, j),
                SHPreds[n], [], 128, 256, 1, 1)
        SHPredNp = SHPreds[n].data.cpu().numpy()
        np.save('{0}/{1}_predSH.npy'.format(opt.experiment, j), SHPredNp)

