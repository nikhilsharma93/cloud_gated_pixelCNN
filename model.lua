------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
-- This file defines the model, creates one, and returns it to be used by
-- train.lua and test.lua
------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nngraph'
require 'math'
require 'SpatialConvolution_masked'
require 'ReshapeCustom'
require 'BiasAddTable'
local nninit = require 'nninit'


local function initializeConv(moduleName, nip, nop, ...)
    return moduleName(nip, nop, ...):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1})
end

local function gatedActivationUnit(n_op)
    -- Define the gated activations
    local tanhBlock = nn.Sequential()
        tanhBlock:add(nn.Narrow(2, 1, n_op))
        tanhBlock:add(nn.Tanh())
        tanhBlock:add(nn.SpatialBatchNormalization(n_op))

    local sigmoidBlock = nn.Sequential()
        sigmoidBlock:add(nn.Narrow(2, n_op+1, n_op))
        sigmoidBlock:add(nn.Sigmoid())
        sigmoidBlock:add(nn.SpatialBatchNormalization(n_op))

    local parallel = nn.ConcatTable(2)
        parallel:add(tanhBlock)
        parallel:add(sigmoidBlock)

    local gatedActivation = nn.Sequential()
        gatedActivation:add(parallel)
        gatedActivation:add(nn.CMulTable())

    return gatedActivation
end

local function maskChannels(weights, n_ip, n_op, kW, noChannels, firstLayer, singleFeatMap)
    -- There are 2*n_op features that come out of the conv layer
    -- Mask them individually
    local singleFeatMap = singleFeatMap or false
    local n_op = n_op

    --If it is the output layer's masked convolution or the 1*1 convolution
    --on the horizontal stack, then there wil be just one set of feature maps,
    --since there is no split.
    if singleFeatMap == false then n_op = n_op/2 end

    local endLoop --if it is first channel, we have to include the last layer too
    if firstLayer == true then endLoop = noChannels else endLoop = noChannels-1 end

    for loopChannels = 1, endLoop do

        ---------------
        --Mask the first n_op features
        ---------------

        -- Select indices corresponding to op features
        local maskStartOp = 1 + (loopChannels-1)*(n_op/noChannels)
        local maskEndOp = loopChannels*(n_op/noChannels)

        -- Select the indices corresponding to ip channels
        if n_ip % 3 ~= 0 then print ('nip: ',n_ip); os.exit() end
        local maskStartIp
        if firstLayer == true then
            maskStartIp = (n_ip/noChannels)*(loopChannels-1)+1
        else
            maskStartIp = (n_ip/noChannels)*(loopChannels)+1
        end
        local maskEndIp = n_ip

        -- Select the position of the ith pixel in the kernel
        local pixelPos = kW
        weights[{ {maskStartOp,maskEndOp}, {maskStartIp,maskEndIp},
                  {}, {pixelPos} }] = 0


        ---------------
        -- If we are dealing with a single set of feature maps, just continue
        -- else, mask the next n_op features
        ---------------
        if singleFeatMap == true then ;
        else
            -- This can be done by just shifting the starting and ending positions
            -- of op features by n_op
            maskStartOp = maskStartOp + n_op
            maskEndOp = maskEndOp + n_op

            --Mask
            weights[{ {maskStartOp,maskEndOp}, {maskStartIp,maskEndIp},
                      {}, {pixelPos} }] = 0
        end
    end
end

local function fuse(n_op, factor, ...)
    local fuseConv
    local numArgs = select('#',...)
    if numArgs == 0 then
        fuseConv = initializeConv(nn.SpatialConvolution, factor*n_op, factor*n_op, 1, 1)
    else
        fuseConv = initializeConv(nn.SpatialConvolution_masked, factor*n_op, factor*n_op, 1, 1,
                                  1, 1, 0, 0, maskChannels, ...)
    end

    local parallel = nn.ParallelTable()
        parallel:add(nn.Identity())
        parallel:add(fuseConv)

    local fused = nn.Sequential()
        fused:add(parallel)
        fused:add(nn.CAddTable())
    return fused
end




local function gatedPixelUnit(n_ip, n_op, filtSize, noChannels, isFirstLayer,
                              embedddingSize)

    -- Define the previous vertical and horizontal stack
    local vStackIn = - nn.Identity() --make it a nngraph node
    local hStackIn = - nn.Identity() --make it a nngraph node
    local embedding = - nn.Identity() --make it a nngraph node


    local noChannels = noChannels --This helps during masking, when it is
                                  -- not the first layer

    -------------------------------------
    --Compute the output vertical stack
    -------------------------------------

    -- As hinted in the paper, the n*n masked convolution can be done by
    -- floor(n/2)*n kernel with appropriate padding and cropping
    local kernelW = filtSize
    local kernelH = math.floor(filtSize/2)
    local padW = math.floor(filtSize/2) -- so that width of op = width of ip
    local padH = math.floor(filtSize/2)
        -- To align with the masking scheme

    local vConv = initializeConv(nn.SpatialConvolution,
                                 n_ip, 2*n_op, kernelW, kernelH, 1,1, padW, padH)

    -- Note that the op of this layer has extra rows at the bottom, due to padding
    -- Crop those rows out
    local n_extraRows = math.floor(filtSize/2) + 1 --number of extra rows
    local vCropped = nn.SpatialZeroPadding(0,0,0, -n_extraRows)

    local vConvCropped = vStackIn
        - vConv
        - vCropped

    local verticalEmbedding = embedding
        - nn.Linear(embedddingSize, 2*n_op)

    local vStackOut = {vConvCropped, verticalEmbedding}
        - nn.BiasAddTable()
        - gatedActivationUnit(n_op)


    -------------------------------------
    --Compute the output horizontal stack
    -------------------------------------

    -- As hinted in the paper, the n*1 masked convolution can be done by
    -- ceil(n/2)*1 kernel with appropriate masking, padding, and cropping.
    -- If it is first layer, then the mask type is A
    -- otherwise it is of type B
    kernelW = math.ceil(filtSize/2)
    kernelH = 1
    padW = math.floor(filtSize/2)
    -- To align with the masking scheme
    padH = 0

    local hConv
    local hCropped

    if isFirstLayer == true then
        hConv = initializeConv(nn.SpatialConvolution_masked, n_ip, 2*n_op,
                               kernelW, kernelH, 1,1, padW, padH, maskChannels, noChannels, true)
    else
        hConv = initializeConv(nn.SpatialConvolution_masked, n_ip, 2*n_op,
                               kernelW, kernelH, 1,1, padW, padH, maskChannels, noChannels, false)
    end
    hCropped = nn.SpatialZeroPadding(0, -n_extraRows+1, 0, 0)

    -- Fuse the hStackIn and 1*1 convolved vStackOut
    local hConvCropped = hStackIn
        - hConv
        - hCropped

    local horizontalEmbedding = embedding
        - nn.Linear(embedddingSize, 2*n_op)

    local hFused = {hConvCropped, vConvCropped}
        - fuse(n_op, 2)

    local hStackOut = {hFused, horizontalEmbedding}
        - nn.BiasAddTable()
        - gatedActivationUnit(n_op)


    --If this is the first layer, there are no residual connections
    local hResidualOut
    if isFirstLayer == true then
        hResidualOut = hStackOut
    else
        --Add the residual connections
        hResidualOut = {hStackIn, hStackOut}
            - fuse(n_op, 1, noChannels, false, true)
    end

    return nn.gModule({vStackIn, hStackIn, embedding}, {vStackOut, hResidualOut, embedding})
end


local function createModel(noChannels, noFeatures, noLayers, quantLevels,
                           firstFiltSize, genFiltSize, noOutputFeatures, embedddingSize)

    local input = - nn.Identity()
    local embedding = - nn.Identity()

    local layers = nn.Sequential()
    for loopLayers = 1, noLayers do
        layers:add(gatedPixelUnit(noFeatures, noFeatures, genFiltSize, noChannels,
                                  false, embedddingSize))
    end

    local output = {input, nn.Copy()(input), embedding}
        - gatedPixelUnit(noChannels, noFeatures, firstFiltSize, noChannels,
                         true, embedddingSize)
        - layers

    --Now the output layer
    local outputLayer = nn.Sequential()
        outputLayer:add(nn.SelectTable(2)) --Select the horizontal stack out
        -- Followed by 2 layers of (ReLU + 1*1 conv of mask B)
        outputLayer:add(nn.ReLU())
        outputLayer:add(nn.SpatialBatchNormalization(noFeatures))
        outputLayer:add(initializeConv(nn.SpatialConvolution_masked, noFeatures,
                                        noOutputFeatures, 1,1, 1,1,0,0, maskChannels, noChannels, false, true))
        outputLayer:add(nn.ReLU())
        outputLayer:add(nn.SpatialBatchNormalization(noOutputFeatures))
        outputLayer:add(initializeConv(nn.SpatialConvolution_masked, noOutputFeatures,
                                        quantLevels*noChannels, 1,1,1,1,0,0, maskChannels, noChannels, false, true))



    local finalOutput = output
        - outputLayer
    -- The 4D output is BatchSize * (noChannels*quantLevels) * N * N

    local model = nn.gModule({input, embedding}, {finalOutput})
    return model
end


--Final LogSoftMax layer
local function addSoftMax(noChannels)
    local outputSoftMax = nn.Sequential()
        outputSoftMax:add(nn.ReshapeCustom(noChannels))
        outputSoftMax:add(nn.LogSoftMax())
    return outputSoftMax
end

-- Optional; to test model
local function testModel()
    local noChannels = 3
    local model = nn.Sequential()
        model:add(createModel(noChannels, 12, 5, 256, 7, 3, 15, 10))
        model:add(addSoftMax(noChannels))
    local inp = torch.rand(2,noChannels,32,32)
    local embedding = torch.Tensor(2,10):rand(2,10)
    local target = torch.Tensor(2*noChannels,32,32):random(255)
    local loss = nn.SpatialClassNLLCriterion()

    -- Test model forward
    local function testForward(model, inp, emb)
        return model:forward({inp,emb});
    end
    ok,op = pcall(testForward, model, inp, embedding)
    if ok then print ('Tested Model Forward') end


    -- Test Loss
    local function testLoss(output, target)
        loss:forward(output, target);
        return loss:backward(output, target);
    end
    ok,de = pcall(testLoss, op, target, true)
    if ok then print ('Tested Loss Function') end


    -- Test model forward
    local function testBackward(model, inp, emb, d)
        model:backward({inp,emb}, d);
    end
    ok = pcall(testBackward, model, inp, embedding, de)
    if ok then print ('Tested Model Backward') end


    -- Plot/Save the model to visualize
    --graph.dot((model.modules[1]).fg,'CNN','CNNVerify')
end

testModel()

print ('creating model...')

---------------------------------------------------------
-- Create a model and return it
---------------------------------------------------------
local noChannels = 3     --No of input channels
local noFeatures = 42*noChannels    --Hidden layer features
local noLayers = 12        --No of hidden layers
local quantLevels = 256    --8-bit image, 1 to 256 values
local firstFiltSize = 7  --filter size at input
local genFiltSize = 3    --filter size of hidden layer
local noOutputFeatures = 340*noChannels --No of features in the final output layer
local embedddingSize = 10  --length of one-hot encoding across different classes

model = nn.Sequential()
    model:add(createModel(noChannels, noFeatures, noLayers, quantLevels,
                          firstFiltSize, genFiltSize, noOutputFeatures, embedddingSize))
    model:add(addSoftMax(noChannels))

print (noChannels, noFeatures, noLayers, quantLevels,
                    firstFiltSize, genFiltSize, noOutputFeatures, embedddingSize)
-- return package:
return {
   model = model,
   loss = nn.SpatialClassNLLCriterion(),
}
