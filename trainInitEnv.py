import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

parser = argparse.ArgumentParser()
# The locationi of training set
#parser.add_argument('--dataRoot', default='DATA/', help='path to real image distorted by water')
parser.add_argument('--dataRoot', default='/datasets/home/13/113/ptayal/CSE291DA/svBRDFEstimation/DATA/', help='path to real image distorted by water')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=100, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=3, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')
parser.add_argument('--depthWeight', type=float, default=0.5, help='the weight for the depth component')
parser.add_argument('--globalIllu1', type=float, default=1.0, help='the weight of bounce 1')
parser.add_argument('--envWeight', type=float, default=0.01, help = 'the weight of training network for environmap prediction')
# Refine the network
parser.add_argument('--isRefine', action = 'store_true', help='whether to refine the network or not')
parser.add_argument('--epochId', type =int, default =-1, help='the training epoch for the model')
parser.add_argument('--modelRoot', default= None, help='the root to load the trained model')

parser.add_argument('--lamC', type=float, default=1.0, help='weight')
parser.add_argument('--lamZ', type=float, default=1.0, help='weight')
parser.add_argument('--lamTr', type=float, default=1.0, help='weight')
parser.add_argument('--lamId', type=float, default=8.0, help='weight')
parser.add_argument('--lamCyc', type=float, default=4.0, help='weight')
parser.add_argument('--lamTrc', type=float, default=1.0, help='weight')
parser.add_argument('--cascadeLevel', type=int, default=0, help='cascade level')

parser.add_argument('--loadModel', action='store_true', help='Load Saved Model')
parser.add_argument('--modelPath', default='check_initEnv/TrainedModel.pth', help='path to saved model')

parser.add_argument('--batchavgsize', type = int, default = 100, help = 'loss average after how many minibatches')
opt = parser.parse_args()
print(opt)

opt.gpuId = opt.deviceIds[0]

if opt.experiment is None:
    opt.experiment = 'check_initEnv'
os.system('mkdir {0}'.format(opt.experiment) )

os.system('cp *.py %s' % opt.experiment )

albeW, normW = opt.albedoWeight, opt.normalWeight
rougW, deptW = opt.roughWeight, opt.depthWeight
eW = opt.envWeight
g1W = opt.globalIllu1


lamC = opt.lamC
lamZ = opt.lamZ
lamTr = opt.lamTr
lamId = opt.lamId
lamCyc = opt.lamCyc
lamTrc = opt.lamTrc

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
# initalize tensors
albedoBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
normalBatch  = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
roughBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
segBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
depthBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize) )
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imBgBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
SHBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 9) )
imRealBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
imRealBgBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
segRealBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize) )
# Initial Network
encoderInit = nn.DataParallel(models.encoderInitial(), device_ids = opt.deviceIds)

# [kavidaya] In paper they mention sharing the weights between the two encoders.. I guess having two encoders is an overkill!
# encoderXInit = nn.DataParallel(models.encoderInitialXDA(), device_ids = opt.deviceIds)
# encoderYInit = nn.DataParallel(models.encoderInitialYDA(), device_ids = opt.deviceIds)
decoderXInit = nn.DataParallel(models.decoderInitial(mode=0), device_ids = opt.deviceIds)
decoderYInit = nn.DataParallel(models.decoderInitial(mode=0), device_ids = opt.deviceIds)

albedoInit = nn.DataParallel(models.decoderInitial(mode=0), device_ids = opt.deviceIds)
normalInit = nn.DataParallel(models.decoderInitial(mode=1), device_ids = opt.deviceIds)
roughInit = nn.DataParallel(models.decoderInitial(mode=2), device_ids = opt.deviceIds)
depthInit = nn.DataParallel(models.decoderInitial(mode=3), device_ids = opt.deviceIds)
envInit = nn.DataParallel(models.envmapInitial(), device_ids = opt.deviceIds)

# Discriminators

discriminatorLatentInit = nn.DataParallel(models.DiscriminatorLatent(), device_ids = opt.deviceIds)
discriminatorXInit = nn.DataParallel(models.DiscriminatorImg(), device_ids = opt.deviceIds)
discriminatorYInit = nn.DataParallel(models.DiscriminatorImg(), device_ids = opt.deviceIds)
# Refine Network
encoderRefs, albedoRefs = [], []
normalRefs, roughRefs = [], []
depthRefs = []

renderLayer = models.renderingLayer(gpuId = opt.gpuId, isCuda = opt.cuda)

scale = 1.0



##############  ######################
# Send things into GPU
if opt.cuda:
    albedoBatch = albedoBatch.cuda(opt.gpuId)
    normalBatch = normalBatch.cuda(opt.gpuId)
    roughBatch = roughBatch.cuda(opt.gpuId)
    depthBatch = depthBatch.cuda(opt.gpuId)
    segBatch = segBatch.cuda(opt.gpuId)
    imBatch = imBatch.cuda(opt.gpuId)
    imBgBatch = imBgBatch.cuda(opt.gpuId)
    imRealBatch = imRealBatch.cuda(opt.gpuId)
    imRealBgBatch = imRealBgBatch.cuda(opt.gpuId)
    segRealBatch = segRealBatch.cuda(opt.gpuId)
    SHBatch = SHBatch.cuda(opt.gpuId)

    # encoderInit = encoderInit.cuda(opt.gpuId)
    encoderInit = encoderInit.cuda(opt.gpuId)
    albedoInit = albedoInit.cuda(opt.gpuId)
    normalInit = normalInit.cuda(opt.gpuId)
    roughInit = roughInit.cuda(opt.gpuId)
    depthInit = depthInit.cuda(opt.gpuId)
    envInit = envInit.cuda(opt.gpuId)
    discriminatorLatentInit = discriminatorLatentInit.cuda(opt.gpuId)
    discriminatorXInit = discriminatorXInit.cuda(opt.gpuId)
    discriminatorYInit = discriminatorYInit.cuda(opt.gpuId)
####################################


####################################
# Initial Optimizer
# opEncoderInit = optim.Adam(encoderInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )

# [kavidaya] keeping the same stratergy for learning rate!!
# [kavidaya] since the skip connections are helpful in predicting albedo etc, keeping them intact...
# [kavidaya] It is only from decoders x and y that these connections are removed.
# [kavidaya] They would still work with the decoder h..

opEncoderInit = optim.Adam(encoderInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )
#opencoderInit = optim.Adam(encoderInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )
opDecoderXInit = optim.Adam(decoderXInit.parameters(), lr=2e-4 * scale, betas=(0.5, 0.999) )
opDecoderYInit = optim.Adam(decoderYInit.parameters(), lr=2e-4 * scale, betas=(0.5, 0.999) )

opAlbedoInit = optim.Adam(albedoInit.parameters(), lr=2e-4 * scale, betas=(0.5, 0.999) )
opNormalInit = optim.Adam(normalInit.parameters(), lr=2e-4 * scale, betas=(0.5, 0.999) )
opRoughInit = optim.Adam(roughInit.parameters(), lr=2e-4 * scale, betas=(0.5, 0.999) )
opDepthInit = optim.Adam(depthInit.parameters(), lr=2e-4 * scale, betas=(0.5, 0.999) )
opEnvInit = optim.Adam(envInit.parameters(), lr=2e-4, betas=(0.5, 0.999) )

# [kavidaya] Decide a learning rate for the discriminators..
opDiscriminatorLatentInit = optim.Adam(discriminatorLatentInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )
opDiscriminatorXInit = optim.Adam(discriminatorXInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )
opDiscriminatorYInit = optim.Adam(discriminatorYInit.parameters(), lr=1e-4 * scale, betas=(0.5, 0.999) )

#####################################



####################################
brdfDataset = dataLoader.BatchLoader(opt.dataRoot, imSize = opt.imageSize)
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 4, shuffle = False)

j = 0
# Stores j values for which we are generating graphs of losses
js = []
losses = ['totalErr', 'totalErrOrig','lossQz','lossQtrDisc','lossQid','lossQcyc','totalErrTrc']
# Loss values after the discriminator step
loss_trends_after_D = {k: [0] for k in losses}
# Loss values after the generator step
loss_trends_after_G = {k: [0] for k in losses}

albedoErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
globalIllu1ErrsNpList= np.ones( [1, 1+opt.cascadeLevel], dtype = np.float32 )
envErrsNpList = np.ones([1, 1+opt.cascadeLevel], dtype = np.float32)


lossMSE = torch.nn.MSELoss()
# lossCEntropy = torch.nn.CrossEntropyLoss()
lossDiscriminator = nn.BCELoss()

epoch = 0
if opt.loadModel : 
    print ('############Loading Model###########')
    checkpoint = torch.load(opt.modelPath)
    epoch = checkpoint['epoch']
    encoderInit.load_state_dict(checkpoint['encoderInit'])
    decoderXInit.load_state_dict(checkpoint['decoderXInit'])
    decoderYInit.load_state_dict(checkpoint['decoderYInit'])
    albedoInit.load_state_dict(checkpoint['albedoInit'])
    normalInit.load_state_dict(checkpoint['normalInit'])
    roughInit.load_state_dict(checkpoint['roughInit'])
    depthInit.load_state_dict(checkpoint['depthInit'])
    envInit.load_state_dict(checkpoint['envInit'])
    discriminatorLatentInit.load_state_dict(checkpoint['discriminatorLatentInit'])
    discriminatorXInit.load_state_dict(checkpoint['discriminatorXInit'])
    discriminatorYInit.load_state_dict(checkpoint['discriminatorYInit'])


    opEncoderInit.load_state_dict(checkpoint['opEncoderInit'])
    opDecoderXInit.load_state_dict(checkpoint['opDecoderXInit'])
    opDecoderYInit.load_state_dict(checkpoint['opDecoderYInit'])
    opAlbedoInit.load_state_dict(checkpoint['opAlbedoInit'])
    opNormalInit.load_state_dict(checkpoint['opNormalInit'])
    opRoughInit.load_state_dict(checkpoint['opRoughInit'])
    opDepthInit.load_state_dict(checkpoint['opDepthInit'])
    opEnvInit.load_state_dict(checkpoint['opEnvInit'])
    opDiscriminatorLatentInit.load_state_dict(checkpoint['opDiscriminatorLatentInit'])
    opDiscriminatorXInit.load_state_dict(checkpoint['opDiscriminatorXInit'])
    opDiscriminatorYInit.load_state_dict(checkpoint['opDiscriminatorYInit'])

    print ('############Model Loaded###########')


# [kavidaya] have to change the dataloader so that it gives batch/2 real and batch/2 fake...
# I guess just adding another key to the dictionary for realImages would suffice..
# This would reduce the amount of code we change below..

for epoch in list(range(opt.epochId+1, opt.nepoch)):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, dataBatch in enumerate(brdfLoader):
        j += 1
        # Load data from cpu to gpu
        albedo_cpu = dataBatch['albedo']
        albedoBatch.data.resize_(albedo_cpu.size() )
        albedoBatch.data.copy_(albedo_cpu )
        normal_cpu = dataBatch['normal']
        normalBatch.data.resize_(normal_cpu.size() )
        normalBatch.data.copy_(normal_cpu )
        rough_cpu = dataBatch['rough']
        roughBatch.data.resize_(rough_cpu.size() )
        roughBatch.data.copy_(rough_cpu )
        seg_cpu = dataBatch['seg']
        segBatch.data.resize_(seg_cpu.size() )
        segBatch.data.copy_(seg_cpu )
        segReal_cpu = dataBatch['realImageMask']
        segRealBatch.data.resize_(segReal_cpu.size() )
        segRealBatch.data.copy_(segReal_cpu )

        depth_cpu = dataBatch['depth']
        depthBatch.data.resize_(depth_cpu.size() )
        depthBatch.data.copy_(depth_cpu )

        # Load the image from cpu to gpu
        im_cpu = (dataBatch['imP'] + dataBatch['imE'] + 1) * seg_cpu.expand_as(normal_cpu)
        imBatch.data.resize_(im_cpu.shape )
        imBatch.data.copy_(im_cpu )
        imReal_cpu = dataBatch['realImage']
        imRealBatch.data.resize_(imReal_cpu.shape )
        imRealBatch.data.copy_(imReal_cpu )


        imBg_cpu = 0.5*(dataBatch['imP'] + 1) * seg_cpu.expand_as(normal_cpu ) \
                + 0.5*(dataBatch['imEbg'] + 1)
        imBg_cpu = 2*imBg_cpu - 1
        imBgBatch.data.resize_(imBg_cpu.size() )
        imBgBatch.data.copy_(imBg_cpu )

        # Load the spherical harmonics
        SH_cpu = dataBatch['SH']
        SHBatch.data.resize_(SH_cpu.size() )
        SHBatch.data.copy_(SH_cpu )
        nameBatch = dataBatch['name']


        # Clear the gradient in optimizer
        # opEncoderInit.zero_grad()


        # DISCRIMINATOR STEP....
        #######################################################
        opAlbedoInit.zero_grad()
        opNormalInit.zero_grad()
        opRoughInit.zero_grad()
        opDepthInit.zero_grad()
        opEnvInit.zero_grad()
        opEncoderInit.zero_grad()
        opDecoderXInit.zero_grad()
        opDecoderYInit.zero_grad()
        opDiscriminatorLatentInit.zero_grad()
        opDiscriminatorXInit.zero_grad()
        opDiscriminatorYInit.zero_grad()
        ########################################################
        # Build the cascade network architecture #
        albedoPreds = []
        normalPreds = []
        roughPreds = []
        depthPreds = []
        SHPreds = []
        globalIllu1s = []
        errors = []

        globalIllu1Gt = renderLayer.forward(albedoBatch, normalBatch,
                roughBatch, depthBatch, segBatch).detach()


        # Initial Prediction
        inputInit = torch.cat([imBatch, segBatch], dim=1)
        x1, x2, x3, x4, x5, xSynth = encoderInit(inputInit)
        albedoPred = albedoInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(albedoBatch )
        normalPred = normalInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(normalBatch )
        roughPred = roughInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(roughBatch )
        depthPred = depthInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(depthBatch )
        SHPred = envInit(xSynth)
        SHPreds.append(SHPred)
        globalIllu1 = renderLayer.forward(albedoPred, normalPred,
                roughPred, depthPred, segBatch)

        albedoPreds.append(albedoPred)
        normalPreds.append(normalPred)
        roughPreds.append(roughPred)
        depthPreds.append(depthPred)
        globalIllu1s.append(globalIllu1)

        ########################################################

        # Formulate Domain adaptation losses assuming batch/2 are synthetic and remaining real
        #print (imRealBatch.shape, segRealBatch.shape)
        inputRealInit = torch.cat([imRealBatch, segRealBatch], dim=1)
        y1, y2, y3, y4, y5, xReal = encoderInit(inputRealInit)
        # Loss 13333333 :
        #print ('#####',decoderXInit(xSynth).shape)
        idSynthetic = decoderXInit(x1, x2, x3, x4, x5, xSynth)
        idReal = decoderYInit(y1, y2, y3, y4, y5, xReal)
        lossQid = lossMSE(idSynthetic, inputInit[:,0:3,:,:]) + lossMSE(idReal, inputRealInit[:,0:3,:,:])# L2 norm between synthetic images + L2 norm between real images 

        # Loss 2 : 
        predLatent = torch.cat([xSynth, xReal])
        # take synthetic as 0s and real as 1s
        # May be have to send to GPU
        labels = torch.ones(len(predLatent)).cuda()
        #print ('#######################',len(predLatent))
        labels[0:len(predLatent)//2] = 0
        predLabels = discriminatorLatentInit(predLatent)
        # Discriminator is LSGAN
        lossQz = lossMSE(predLabels, labels)
        # lossQz = lossCEntropy(predLabels, labels) # cross entropy loss between predLabels and labels..

        # Loss 3 : 
        predTransX = discriminatorYInit(decoderYInit(x1, x2, x3, x4, x5, xSynth))
        predTransY = discriminatorXInit(decoderXInit(y1, y2, y3, y4, y5, xReal))
        lossQtr = lossDiscriminator(predTransX, torch.zeros(predTransX.size()).cuda()) + lossDiscriminator(predTransY, torch.zeros(predTransY.size()).cuda()) # cross entropy loss...

        # [kavidaya] Also, for training these descriminators, we would have to pass in the real images too...
        predActualX = discriminatorXInit(inputInit[:,0:3,:,:])
        predActualY = discriminatorYInit(inputRealInit[:,0:3,:,:])
        out = torch.zeros(torch.cat((predTransX, predActualX)).size()).cuda()
        out[0:len(predTransX)] = 1
        lossQtrDisc = lossDiscriminator(torch.cat((predActualX, predTransX)), out) + lossDiscriminator(torch.cat((predActualY, predTransY)), out) #

        # Loss 4 : 
        lossQcyc = lossMSE(decoderXInit(*encoderInit(torch.cat((decoderYInit(x1, x2, x3, x4, x5, xSynth), inputInit[:,3:4,:,:]),dim=1))),inputInit[:,0:3,:,:]) + lossMSE(decoderYInit(*encoderInit(torch.cat((decoderXInit(y1, y2, y3, y4, y5, xReal),inputRealInit[:,3:4,:,:]),dim=1))), inputRealInit[:,0:3,:,:])

        # Loss 5 :

        x1, x2, x3, x4, x5, xSynthTrc = encoderInit(torch.cat((decoderYInit(x1, x2, x3, x4, x5, xSynth),inputInit[:,3:4,:,:]),dim=1))
        albedoPredsTrc = albedoInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(albedoBatch )
        normalPredsTrc = normalInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(normalBatch )
        roughPredsTrc = roughInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(roughBatch )
        depthPredsTrc = depthInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(depthBatch )
        SHPredsTrc = envInit(xSynthTrc)
        globalIllu1sTrc = renderLayer.forward(albedoPredsTrc, normalPredsTrc,
                roughPredsTrc, depthPredsTrc, segBatch)

        ########################################################

        # Compute the error
        albedoErrs = []
        normalErrs = []
        roughErrs = []
        depthErrs = []
        globalIllu1Errs = []
        envErrs = []

        albedoErrsTrc = []
        normalErrsTrc = []
        roughErrsTrc = []
        depthErrsTrc = []
        globalIllu1ErrsTrc = []
        envErrsTrc = []


        pixelNum = (torch.sum(segBatch ).cpu().data).item()
        #print (pixelNum)
        for m in range(0, len(albedoPreds) ):
            albedoErrs.append( torch.sum( (albedoPreds[m] - albedoBatch)
                    * (albedoPreds[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )
            albedoErrsTrc.append( torch.sum( (albedoPredsTrc[m] - albedoBatch)
                    * (albedoPredsTrc[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )

        for m in range(0, len(normalPreds) ):
            normalErrs.append( torch.sum( (normalPreds[m] - normalBatch)
                    * (normalPreds[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )
            normalErrsTrc.append( torch.sum( (normalPredsTrc[m] - normalBatch)
                    * (normalPredsTrc[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )

        for m in range(0, len(roughPreds) ):
            roughErrs.append( torch.sum( (roughPreds[m] - roughBatch)
                    * (roughPreds[m] - roughBatch) * segBatch ) / pixelNum )
            roughErrsTrc.append( torch.sum( (roughPredsTrc[m] - roughBatch)
                    * (roughPredsTrc[m] - roughBatch) * segBatch ) / pixelNum )

        for m in range(0, len(depthPreds) ):
            depthErrs.append( torch.sum( (depthPreds[m] - depthBatch)
                    * (depthPreds[m] - depthBatch) * segBatch ) / pixelNum )
            depthErrsTrc.append( torch.sum( (depthPredsTrc[m] - depthBatch)
                    * (depthPredsTrc[m] - depthBatch) * segBatch ) / pixelNum )

        for m in range(0, len(globalIllu1s) ):
            globalIllu1Errs.append( torch.sum( (globalIllu1s[m] - globalIllu1Gt)
                    * (globalIllu1s[m] - globalIllu1Gt) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
            globalIllu1ErrsTrc.append( torch.sum( (globalIllu1sTrc[m] - globalIllu1Gt)
                    * (globalIllu1sTrc[m] - globalIllu1Gt) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )

        for m in range(0, len(SHPreds) ):
            envErrs.append( torch.mean( (SHPreds[m] - SHBatch) * (SHPreds[m] - SHBatch) ) )
            envErrsTrc.append( torch.mean( (SHPredsTrc[m] - SHBatch) * (SHPredsTrc[m] - SHBatch) ) )

        # Back propagate the gradients
        albedoErrSum = sum(albedoErrs)
        normalErrSum = sum(normalErrs)
        roughErrSum = sum(roughErrs)
        depthErrSum = sum(depthErrs)
        globalIllu1ErrSum = sum(globalIllu1Errs)
        envErrSum = sum(envErrs)

        albedoErrSumTrc = sum(albedoErrsTrc)
        normalErrSumTrc = sum(normalErrsTrc)
        roughErrSumTrc = sum(roughErrsTrc)
        depthErrSumTrc = sum(depthErrsTrc)
        globalIllu1ErrSumTrc = sum(globalIllu1ErrsTrc)
        envErrSumTrc = sum(envErrsTrc)

        totalErrOrig = albeW * albedoErrSum + normW * normalErrSum + rougW *roughErrSum \
                + deptW * depthErrSum + g1W * globalIllu1ErrSum + eW * envErrSum

        totalErrTrc = albeW * albedoErrSumTrc + normW * normalErrSumTrc + rougW *roughErrSumTrc \
                + deptW * depthErrSumTrc + g1W * globalIllu1ErrSumTrc + eW * envErrSumTrc


        ########################################################



        totalErr = lamC*totalErrOrig + lamZ*lossQz + lamTr*lossQtrDisc + lamId*lossQid + lamCyc*lossQcyc + lamTrc*totalErrTrc
        totalErr.backward(retain_graph=True)

        # Update the network parameter
        opDiscriminatorXInit.step()
        opDiscriminatorYInit.step()
        opDiscriminatorLatentInit.step()
        for losstype in losses:
            loss_trends_after_D[losstype][-1] += globals()[losstype].item()
        
        if j%opt.batchavgsize ==0:
            for losstype in losses:
                loss_trends_after_D[losstype][-1]/= opt.batchavgsize
                loss_trends_after_D[losstype].append(0)
                js.append(j)
                utils.writeNpErrToScreen(losstype+'D', [loss_trends_after_D[losstype][-2]], epoch, j)
        


        # Now GENERATOR STEP....
        #######################################################
        opAlbedoInit.zero_grad()
        opNormalInit.zero_grad()
        opRoughInit.zero_grad()
        opDepthInit.zero_grad()
        opEnvInit.zero_grad()
        opEncoderInit.zero_grad()
        opDecoderXInit.zero_grad()
        opDecoderYInit.zero_grad()
        opDiscriminatorLatentInit.zero_grad()
        opDiscriminatorXInit.zero_grad()
        opDiscriminatorYInit.zero_grad()
        ########################################################

        ########################################################
        # Build the cascade network architecture #
        albedoPreds = []
        normalPreds = []
        roughPreds = []
        depthPreds = []
        SHPreds = []
        globalIllu1s = []
        errors = []

        globalIllu1Gt = renderLayer.forward(albedoBatch, normalBatch,
                roughBatch, depthBatch, segBatch).detach()


        # Initial Prediction
        inputInit = torch.cat([imBatch, segBatch], dim=1)
        x1, x2, x3, x4, x5, xSynth = encoderInit(inputInit)
        albedoPred = albedoInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(albedoBatch )
        normalPred = normalInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(normalBatch )
        roughPred = roughInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(roughBatch )
        depthPred = depthInit(x1, x2, x3, x4, x5, xSynth) * segBatch.expand_as(depthBatch )
        SHPred = envInit(xSynth)
        SHPreds.append(SHPred)
        globalIllu1 = renderLayer.forward(albedoPred, normalPred,
                roughPred, depthPred, segBatch)

        albedoPreds.append(albedoPred)
        normalPreds.append(normalPred)
        roughPreds.append(roughPred)
        depthPreds.append(depthPred)
        globalIllu1s.append(globalIllu1)

        ########################################################

        # Formulate Domain adaptation losses assuming batch/2 are synthetic and remaining real
        inputRealInit = torch.cat([imRealBatch, segRealBatch], dim=1)
        y1, y2, y3, y4, y5, xReal = encoderInit(inputRealInit)
        # Loss 1 :
        idSynthetic = decoderXInit(x1, x2, x3, x4, x5, xSynth)
        idReal = decoderYInit(y1, y2, y3, y4, y5, xReal)
        lossQid = lossMSE(idSynthetic, inputInit[:,0:3,:,:]) + lossMSE(idReal, inputRealInit[:,0:3,:,:])# L2 norm between synthetic images + L2 norm between real images 

        # Loss 2 : 
        labels = torch.zeros(len(xReal)).cuda()
        predLabels = discriminatorLatentInit(xReal)
        lossQz = lossMSE(predLabels, labels)
        # lossQz = lossCEntropy(predLabels, labels) # cross entropy loss between predLabels and labels..

        # Loss 3 : 
        predTransX = discriminatorYInit(decoderYInit(x1, x2, x3, x4, x5, xSynth))
        predTransY = discriminatorXInit(decoderXInit(y1, y2, y3, y4, y5, xReal))
        lossQtr = lossDiscriminator(predTransX, torch.ones(predTransX.size()).cuda()) + lossDiscriminator(predTransY, torch.ones(predTransY.size()).cuda()) # cross entropy loss...

        # Loss 4 : 
        lossQcyc = lossMSE(decoderXInit(*encoderInit(torch.cat((decoderYInit(x1, x2, x3, x4, x5, xSynth), inputInit[:,3:4,:,:]),dim=1))),inputInit[:,0:3,:,:]) + lossMSE(decoderYInit(*encoderInit(torch.cat((decoderXInit(y1, y2, y3, y4, y5, xReal),inputRealInit[:,3:4,:,:]),dim=1))), inputRealInit[:,0:3,:,:])


        # Loss 5 :
        
        x1, x2, x3, x4, x5, xSynthTrc = encoderInit(torch.cat((decoderYInit(x1, x2, x3, x4, x5, xSynth),inputInit[:,3:4,:,:]),dim=1))
        albedoPredTrc = albedoInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(albedoBatch )
        normalPredTrc = normalInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(normalBatch )
        roughPredTrc = roughInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(roughBatch )
        depthPredTrc = depthInit(x1, x2, x3, x4, x5, xSynthTrc) * segBatch.expand_as(depthBatch )
        SHPredTrc = envInit(xSynthTrc)
        globalIllu1sTrc = renderLayer.forward(albedoPredTrc, normalPredTrc,
                roughPredTrc, depthPredTrc, segBatch)

        ########################################################

        # Compute the error
        albedoErrs = []
        normalErrs = []
        roughErrs = []
        depthErrs = []
        globalIllu1Errs = []
        envErrs = []

        albedoErrsTrc = []
        normalErrsTrc = []
        roughErrsTrc = []
        depthErrsTrc = []
        globalIllu1ErrsTrc = []
        envErrsTrc = []


        pixelNum = (torch.sum(segBatch ).cpu().data).item()
        for m in range(0, len(albedoPreds) ):
            albedoErrs.append( torch.sum( (albedoPreds[m] - albedoBatch)
                    * (albedoPreds[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )
            albedoErrsTrc.append( torch.sum( (albedoPredsTrc[m] - albedoBatch)
                    * (albedoPredsTrc[m] - albedoBatch) * segBatch.expand_as(albedoBatch) ) / pixelNum / 3.0 )

        for m in range(0, len(normalPreds) ):
            normalErrs.append( torch.sum( (normalPreds[m] - normalBatch)
                    * (normalPreds[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )
            normalErrsTrc.append( torch.sum( (normalPredsTrc[m] - normalBatch)
                    * (normalPredsTrc[m] - normalBatch) * segBatch.expand_as(normalBatch) ) / pixelNum / 3.0 )

        for m in range(0, len(roughPreds) ):
            roughErrs.append( torch.sum( (roughPreds[m] - roughBatch)
                    * (roughPreds[m] - roughBatch) * segBatch ) / pixelNum )
            roughErrsTrc.append( torch.sum( (roughPredsTrc[m] - roughBatch)
                    * (roughPredsTrc[m] - roughBatch) * segBatch ) / pixelNum )

        for m in range(0, len(depthPreds) ):
            depthErrs.append( torch.sum( (depthPreds[m] - depthBatch)
                    * (depthPreds[m] - depthBatch) * segBatch ) / pixelNum )
            depthErrsTrc.append( torch.sum( (depthPredsTrc[m] - depthBatch)
                    * (depthPredsTrc[m] - depthBatch) * segBatch ) / pixelNum )

        for m in range(0, len(globalIllu1s) ):
            globalIllu1Errs.append( torch.sum( (globalIllu1s[m] - globalIllu1Gt)
                    * (globalIllu1s[m] - globalIllu1Gt) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )
            globalIllu1ErrsTrc.append( torch.sum( (globalIllu1sTrc[m] - globalIllu1Gt)
                    * (globalIllu1sTrc[m] - globalIllu1Gt) * segBatch.expand_as(imBatch) ) / pixelNum / 3.0 )

        for m in range(0, len(SHPreds) ):
            envErrs.append( torch.mean( (SHPreds[m] - SHBatch) * (SHPreds[m] - SHBatch) ) )
            envErrsTrc.append( torch.mean( (SHPredsTrc[m] - SHBatch) * (SHPredsTrc[m] - SHBatch) ) )

        # Back propagate the gradients
        albedoErrSum = sum(albedoErrs)
        normalErrSum = sum(normalErrs)
        roughErrSum = sum(roughErrs)
        depthErrSum = sum(depthErrs)
        globalIllu1ErrSum = sum(globalIllu1Errs)
        envErrSum = sum(envErrs)

        albedoErrSumTrc = sum(albedoErrsTrc)
        normalErrSumTrc = sum(normalErrsTrc)
        roughErrSumTrc = sum(roughErrsTrc)
        depthErrSumTrc = sum(depthErrsTrc)
        globalIllu1ErrSumTrc = sum(globalIllu1ErrsTrc)
        envErrSumTrc = sum(envErrsTrc)

        totalErrOrig = albeW * albedoErrSum + normW * normalErrSum + rougW *roughErrSum \
                + deptW * depthErrSum + g1W * globalIllu1ErrSum + eW * envErrSum

        totalErrTrc = albeW * albedoErrSumTrc + normW * normalErrSumTrc + rougW *roughErrSumTrc \
                + deptW * depthErrSumTrc + g1W * globalIllu1ErrSumTrc + eW * envErrSumTrc


        ########################################################



        totalErr = lamC*totalErrOrig + lamZ*lossQz + lamTr*lossQtr + lamId*lossQid + lamCyc*lossQcyc + lamTrc*totalErrTrc
        totalErr.backward()


        opEncoderInit.step()
        opDecoderXInit.step()
        opDecoderYInit.step()
        opAlbedoInit.step()
        opNormalInit.step()
        opRoughInit.step()
        opDepthInit.step()
        opEnvInit.step()



        # Output training error
#         utils.writeErrToScreen('albedo', albedoErrs, epoch, j)
#         utils.writeErrToScreen('normal', normalErrs, epoch, j)
#         utils.writeErrToScreen('rough', roughErrs, epoch, j)
#         utils.writeErrToScreen('depth', depthErrs, epoch, j)
#         utils.writeErrToScreen('globalIllu1', globalIllu1Errs, epoch, j)
#         utils.writeErrToScreen('Env Error', envErrs, epoch, j)

#         utils.writeErrToFile('albedo', albedoErrs, trainingLog, epoch, j)
#         utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, j)
#         utils.writeErrToFile('rough', roughErrs, trainingLog, epoch, j)
#         utils.writeErrToFile('depth', depthErrs, trainingLog, epoch, j)
#         utils.writeErrToFile('globalIllu1', globalIllu1Errs, trainingLog, epoch, j)
#         utils.writeErrToFile('Env Error', envErrs, trainingLog, epoch, j)

        albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0)
        normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0)
        roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0)
        depthErrsNpList = np.concatenate( [depthErrsNpList, utils.turnErrorIntoNumpy(depthErrs)], axis=0)
        globalIllu1ErrsNpList = np.concatenate( [globalIllu1ErrsNpList, utils.turnErrorIntoNumpy(globalIllu1Errs)], axis=0)
        envErrsNpList = np.concatenate( [envErrsNpList, utils.turnErrorIntoNumpy(envErrs)], axis=0)

#         if j < 1000:
#             utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('envErrs_Accu:', np.mean(envErrsNpList[1:j+1, :], axis=0), epoch, j)

#             utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('envErrs_Accu:', np.mean(envErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j)
#         else:
#             utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[j-999:j+1, :], axis=0), epoch, j)
#             utils.writeNpErrToScreen('envErrs_Accu', np.mean(envErrsNpList[j-999:j+1, :], axis=0), epoch, j)

#             utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('depthAccu', np.mean(depthErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('globalIllu1Accu', np.mean(globalIllu1ErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)
#             utils.writeNpErrToFile('envErrs_Accu', np.mean(envErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j)

   
        for losstype in losses:
            loss_trends_after_G[losstype][-1] += globals()[losstype].item()
        
        if j%opt.batchavgsize ==0 :
            for losstype in losses:
                loss_trends_after_G[losstype][-1]/= opt.batchavgsize
                loss_trends_after_G[losstype].append(0)
                js.append(j)
                utils.writeNpErrToScreen(losstype+'G', [loss_trends_after_G[losstype][-2]], epoch, j)
        
        if j%1000 ==0:
            with open('losses_'+str(j)+'_'+str(epoch)+'.pickle', 'wb') as handle:
                pickle.dump([loss_trends_after_G, loss_trends_after_D, js], handle)
               

        if j == 1 or j == 1000 or j% 5000 == 0:
            



            # Generate forward pass on Real Images...
            inputRealInit = torch.cat([imRealBatch, segRealBatch], dim=1)
            x1, x2, x3, x4, x5, xReal = encoderInit(inputRealInit)
            albedoPredReal = albedoInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(albedoBatch ) #crude hack...
            normalPredReal = normalInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(normalBatch )
            roughPredReal = roughInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(roughBatch )
            depthPredReal = depthInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(depthBatch )
            SHPredReal = envInit(xReal)
            globalIllu1sReal = renderLayer.forward(albedoPredReal, normalPredReal,
                    roughPredReal, depthPredReal, segRealBatch)
            
            
            
            # Save the ground truth and the input
            vutils.save_image( (0.5*(albedoBatch + 1)*segBatch.expand_as(albedoBatch) ).data,
                    '{0}/{1}_albedoGt.png'.format(opt.experiment, j) )
            vutils.save_image( (0.5*(normalBatch + 1)*segBatch.expand_as(normalBatch) ).data,
                    '{0}/{1}_normalGt.png'.format(opt.experiment, j) )
            vutils.save_image( (0.5*(roughBatch + 1)*segBatch.expand_as(roughBatch) ).data,
                    '{0}/{1}_roughGt.png'.format(opt.experiment, j) )

            depthOut = 1 / torch.clamp(depthBatch, 1e-6, 10) * segBatch.expand_as(depthBatch)
            depthOut = (depthOut - 0.25) /0.8
            vutils.save_image( ( depthOut*segBatch.expand_as(depthBatch) ).data,
                    '{0}/{1}_depthGt.png'.format(opt.experiment, j) )

            vutils.save_image( ( (0.5*(imBatch + 1)*segBatch.expand_as(imBatch))**(1.0/2.2) ).data,
                    '{0}/{1}_im.png'.format(opt.experiment, j) )
            vutils.save_image( ( (0.5*(imBgBatch + 1) )**(1.0/2.2) ).data,
                    '{0}/{1}_imBg.png'.format(opt.experiment, j) )

            utils.visualizeSH('{0}/{1}_gtSH.png'.format(opt.experiment, j),
                    SHBatch, nameBatch, 128, 256, 2, 8)

            # Save the predicted results
            ###### Now save real...
            vutils.save_image( ( 0.5*(albedoPredReal + 1)*segRealBatch.expand_as(albedoPredReal) ).data,
                    '{0}/{1}_albedoPredReal_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(normalPredReal + 1)*segRealBatch.expand_as(normalPredReal) ).data,
                    '{0}/{1}_normalPredReal_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(roughPredReal + 1)*segRealBatch.expand_as(roughPredReal) ).data,
                    '{0}/{1}_roughPredReal_{2}.png'.format(opt.experiment, j, 0) )

            depthOut = 1 / torch.clamp(depthPredReal, 1e-6, 10) * segRealBatch.expand_as(depthPredReal)
            deepthOut = (depthPredReal - 0.25) /0.8
            vutils.save_image( ( depthOut * segRealBatch.expand_as(depthPredReal) ).data,
                    '{0}/{1}_depthPredReal_{2}.png'.format(opt.experiment, j, 0) )

            vutils.save_image( ( ( globalIllu1sReal * segRealBatch.expand_as(globalIllu1sReal) )**(1.0/2.2) ).data,
                    '{0}/{1}_imPredReal_{2}.png'.format(opt.experiment, j, 0) )
            
            x1, x2, x3, x4, x5, xReal = encoderInit(inputRealInit)
            albedoPredReal = albedoInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(albedoBatch ) #crude hack...
            normalPredReal = normalInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(normalBatch )
            roughPredReal = roughInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(roughBatch )
            depthPredReal = depthInit(x1, x2, x3, x4, x5, xReal) * segRealBatch.expand_as(depthBatch )
            SHPredReal = envInit(xReal)
            globalIllu1sReal = renderLayer.forward(albedoPredReal, normalPredReal,
                    roughPredReal, depthPredReal, segRealBatch)

            vutils.save_image( ( 0.5*(albedoPredReal + 1)*segRealBatch.expand_as(albedoPredReal) ).data,
                    '{0}/{1}_albedoPredRealXD_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(normalPredReal + 1)*segRealBatch.expand_as(normalPredReal) ).data,
                    '{0}/{1}_normalPredRealXD_{2}.png'.format(opt.experiment, j, 0) )
            vutils.save_image( ( 0.5*(roughPredReal + 1)*segRealBatch.expand_as(roughPredReal) ).data,
                    '{0}/{1}_roughPredRealXD_{2}.png'.format(opt.experiment, j, 0) )

            depthOut = 1 / torch.clamp(depthPredReal, 1e-6, 10) * segRealBatch.expand_as(depthPredReal)
            deepthOut = (depthPredReal - 0.25) /0.8
            vutils.save_image( ( depthOut * segRealBatch.expand_as(depthPredReal) ).data,
                    '{0}/{1}_depthPredRealXD_{2}.png'.format(opt.experiment, j, 0) )

            vutils.save_image( ( ( globalIllu1sReal * segRealBatch.expand_as(globalIllu1sReal) )**(1.0/2.2) ).data,
                    '{0}/{1}_imPredRealXD_{2}.png'.format(opt.experiment, j, 0) )
            
            vutils.save_image( ( (0.5*(imRealBatch + 1))**(1.0/2.2) ).data,
                    '{0}/{1}_imReal.png'.format(opt.experiment, j) )
            
            for n in range(0, opt.cascadeLevel + 1):
                vutils.save_image( ( 0.5*(albedoPreds[n] + 1)*segBatch.expand_as(albedoPreds[n]) ).data,
                        '{0}/{1}_albedoPred_{2}.png'.format(opt.experiment, j, n) )
                vutils.save_image( ( 0.5*(normalPreds[n] + 1)*segBatch.expand_as(normalPreds[n]) ).data,
                        '{0}/{1}_normalPred_{2}.png'.format(opt.experiment, j, n) )
                vutils.save_image( ( 0.5*(roughPreds[n] + 1)*segBatch.expand_as(roughPreds[n]) ).data,
                        '{0}/{1}_roughPred_{2}.png'.format(opt.experiment, j, n) )

                depthOut = 1 / torch.clamp(depthPreds[n], 1e-6, 10) * segBatch.expand_as(depthPreds[n])
                deepthOut = (depthPreds[n] - 0.25) /0.8
                vutils.save_image( ( depthOut * segBatch.expand_as(depthPreds[n]) ).data,
                        '{0}/{1}_depthPred_{2}.png'.format(opt.experiment, j, n) )

                vutils.save_image( ( ( globalIllu1s[n] * segBatch.expand_as(imBatch) )**(1.0/2.2) ).data,
                        '{0}/{1}_imPred_{2}.png'.format(opt.experiment, j, n) )                
                
                utils.visualizeSH('{0}/{1}_predSH.png'.format(opt.experiment, j),
                        SHPreds[m], nameBatch, 128, 256, 2, 8)

            # Save the model...

            torch.save({'epoch' : epoch,
            'encoderInit' : encoderInit.state_dict(),
            'decoderXInit' : decoderXInit.state_dict(),
            'decoderYInit' : decoderYInit.state_dict(),
            'albedoInit' : albedoInit.state_dict(),
            'normalInit' : normalInit.state_dict(),
            'roughInit' : roughInit.state_dict(),
            'depthInit' : depthInit.state_dict(),
            'envInit' : envInit.state_dict(),
            'discriminatorLatentInit' : discriminatorLatentInit.state_dict(),
            'discriminatorXInit' : discriminatorXInit.state_dict(),
            'discriminatorYInit' : discriminatorYInit.state_dict(),
            'opEncoderInit' : opEncoderInit.state_dict(),
            'opDecoderXInit' : opDecoderXInit.state_dict(),
            'opDecoderYInit' : opDecoderYInit.state_dict(),
            'opAlbedoInit' : opAlbedoInit.state_dict(),
            'opNormalInit' : opNormalInit.state_dict(),
            'opRoughInit' : opRoughInit.state_dict(),
            'opDepthInit' : opDepthInit.state_dict(),
            'opEnvInit' : opEnvInit.state_dict(),
            'opDiscriminatorLatentInit' : opDiscriminatorLatentInit.state_dict(),
            'opDiscriminatorXInit' : opDiscriminatorXInit.state_dict(),
            'opDiscriminatorYInit' : opDiscriminatorYInit.state_dict()}, opt.modelPath)

    trainingLog.close()

    # Update the training rate
    if (epoch + 1) % 2 == 0:
        # for param_group in opEncoderInit.param_groups:
        #     param_group['lr'] /= 2
        for param_group in opAlbedoInit.param_groups:
            param_group['lr'] /= 2
        for param_group in opNormalInit.param_groups:
            param_group['lr'] /= 2
        for param_group in opRoughInit.param_groups:
            param_group['lr'] /= 2
        for param_group in opDepthInit.param_groups:
            param_group['lr'] /= 2
        for param_group in opEnvInit.param_groups:
            param_group['lr'] /= 2

    # # Save the error record
    # np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    # np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    # np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList )
    # np.save('{0}/depthError_{1}.npy'.format(opt.experiment, epoch), depthErrsNpList )
    # np.save('{0}/globalIllu1_{1}.npy'.format(opt.experiment, epoch), globalIllu1ErrsNpList )
    # np.save('{0}/envErrs_{1}.npy'.format(opt.experiment, epoch), envErrsNpList )

    # # save the models
    # #torch.save(encoderInit.state_dict(), '{0}/encoderInit_{1}.pth'.format(opt.experiment, epoch) )
    # torch.save(albedoInit.state_dict(), '{0}/albedoInit_{1}.pth'.format(opt.experiment, epoch) )
    # torch.save(normalInit.state_dict(), '{0}/normalInit_{1}.pth'.format(opt.experiment, epoch) )
    # torch.save(roughInit.state_dict(), '{0}/roughInit_{1}.pth'.format(opt.experiment, epoch) )
    # torch.save(depthInit.state_dict(), '{0}/depthInit_{1}.pth'.format(opt.experiment, epoch) )
    # torch.save(envInit.state_dict(), '{0}/envInit_{1}.pth'.format(opt.experiment, epoch) )