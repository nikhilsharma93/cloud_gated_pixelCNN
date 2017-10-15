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
local nninit = require 'nninit'


local function initializeConv(moduleName, ...)
    return moduleName(...)--:init('weight', nninit.xavier, {dist = 'normal', gain = 1.1})
end

local function gatedActivationUnit(n_op)
    -- Define the gated activations
    local tanhBlock = nn.Sequential()
        tanhBlock:add(nn.Narrow(2, 1, n_op))
        tanhBlock:add(nn.Tanh())

    local sigmoidBlock = nn.Sequential()
        sigmoidBlock:add(nn.Narrow(2, n_op+1, n_op))
        sigmoidBlock:add(nn.Sigmoid())

    local parallel = nn.ConcatTable(2)
        parallel:add(tanhBlock)
        parallel:add(sigmoidBlock)

    local gatedActivation = nn.Sequential()
        gatedActivation:add(parallel)
        gatedActivation:add(nn.CMulTable())

    return gatedActivation
end


local function fuse(n_op, factor)
    local fuseConv = initializeConv(nn.SpatialConvolutionMM, factor*n_op, factor*n_op, 1, 1)

    local parallel = nn.ParallelTable()
        parallel:add(nn.Identity())
        parallel:add(fuseConv)

    local fused = nn.Sequential()
        fused:add(parallel)
        fused:add(nn.CAddTable())
    return fused
end


local function maskChannels(weights, n_ip, n_op, kW, noChannels)
    -- There are 2*n_op features that come out of the conv layer
    -- Mask them individually
    local n_op = n_op/2
    for loopChannels = 1, noChannels-1 do

        ---------------
        --Mask the first n_op features
        ---------------

        -- Select indices corresponding to op features
        local maskStartOp = 1 + (loopChannels-1)*(n_op/noChannels)
        local maskEndOp = loopChannels*(n_op/noChannels)

        -- Select the indices corresponding to ip channels
        local maskStartIp = loopChannels+1
        local maskEndIp = noChannels

        -- Select the position of the ith pixel in the kernel
        local pixelPos = kW


        --Mask
        weights[{ {maskStartOp,maskEndOp}, {maskStartIp,maskEndIp},
                  {}, {pixelPos} }] = 0


        ---------------
        --Mask the next n_op features
        ---------------

        -- This can be done by just shifting the starting and ending positions
        -- of op features by n_op
        maskStartOp = maskStartOp + n_op
        maskEndOp = maskEndOp + n_op

        --Mask
        weights[{ {maskStartOp,maskEndOp}, {maskStartIp,maskEndIp},
                  {}, {pixelPos} }] = 0
    end
end


local function gatedPixelUnit(n_ip, n_op, filtSize, noChannels, isFirstLayer)

    -- Define the previous vertical and horizontal stack
    local vStackIn = - nn.Identity() --make it a nngraph node
    local hStackIn = - nn.Identity() --make it a nngraph node


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

    local vConv = initializeConv(nn.SpatialConvolutionMM,
                                 n_ip, 2*n_op, kernelW, kernelH, 1,1, padW, padH)

    -- Note that the op of this layer has extra rows at the bottom, due to padding
    -- Crop those rows out
    local n_extraRows = math.floor(filtSize/2) + 1 --number of extra rows
    local vCropped = nn.SpatialZeroPadding(0,0,0, -n_extraRows)

    local vConvCropped = vStackIn
        - vConv
        - vCropped

    local vStackOut = vConvCropped
        - gatedActivationUnit(n_op)



    -------------------------------------
    --Compute the output horizontal stack
    -------------------------------------

    -- As hinted in the paper, the n*1 masked convolution can be done by
    -- floor(n/2)*1 kernel with appropriate padding and cropping, if first layer,
    -- because current pixel should be masked
    -- ceil(n/2)*1 kernel with appropriate padding and cropping, if other layer
    if isFirstLayer == true then kernelW = math.floor(filtSize/2)
    else
        kernelW = math.ceil(filtSize/2)
    end
    kernelH = 1
    padW = math.floor(filtSize/2)
    -- To align with the masking scheme
    padH = 0

    local hConv
    local hCropped

    -- If first layer, then ith pixel is already masked by kernel width
    -- If not, then ith pixel can the ith pixels from previous channels only
    -- Hence, we have to use a channel mask
    if isFirstLayer == true then
        hConv = initializeConv(nn.SpatialConvolutionMM,
                               n_ip, 2*n_op, kernelW, kernelH, 1,1, padW, padH)
        hCropped = nn.SpatialZeroPadding(0, -n_extraRows, 0, 0)
    else
        hConv = initializeConv(nn.SpatialConvolution_masked, n_ip, 2*n_op,
                               kernelW, kernelH, 1,1, padW, padH, maskChannels, noChannels)
        hCropped = nn.SpatialZeroPadding(0, -n_extraRows+1, 0, 0)
    end


    -- Fuse the hStackIn and 1*1 convolved vStackOut
    local hConvCropped = hStackIn
        - hConv
        - hCropped

    local hStackOut = {hConvCropped, vConvCropped}
        - fuse(n_op, 2)
        - gatedActivationUnit(n_op)


    --If this is the first layer, there are no residual connections
    if isFirstLayer == true then
        return nn.gModule({vStackIn, hStackIn}, {vStackOut, hStackOut})
    else
        --Add the residual connections
        local hResidualOut = {hStackOut, hStackIn}
            - fuse(n_op, 1)
        return nn.gModule({vStackIn, hStackIn}, {vStackOut, hResidualOut})
    end
end


local function createModel(noChannels, noFeatures, noLayers, noClasses,
                           firstFiltSize, genFiltSize)

    local input = - nn.Identity()

    local layers = nn.Sequential()
    for loopLayers = 1, noLayers do
        layers:add(gatedPixelUnit(noFeatures, noFeatures, genFiltSize, noChannels,
                                  false))
    end

    local output = {input, nn.Identity()(input)}
        - gatedPixelUnit(noChannels, noFeatures, firstFiltSize, noChannels,
                         true)
        - layers

    --Now the output layer
    local outputLayer = nn.Sequential()
        outputLayer:add(nn.SelectTable(2)) --Select the horizontal stack out
        -- Followed by 2 layers of (ReLU + 1*1 conv of mask B)
        outputLayer:add(nn.ReLU())
        outputLayer:add(initializeConv(nn.SpatialConvolution_masked, noFeatures,
                                        noFeatures, 1,1, 1,1,0,0, maskChannels, noChannels))
        outputLayer:add(nn.ReLU())
        outputLayer:add(initializeConv(nn.SpatialConvolution_masked, noFeatures,
                                        noClasses*noChannels, 1,1,1,1,0,0, maskChannels, noChannels))


    --Final SoftMax / Sigmoid layer
    -- After the previous layer, the 4D output is
    -- BatchSize * (noChannels*noClasses) * N * N
    -- Break it into chunks of size "BatchSize*noChannels*N*N"
    -- LogSoftMax by itself will do the spatial log softmax on each chunk
    local function logSM(index)
        local model = nn.Sequential()
            model:add(nn.Narrow(2, (index-1)*noClasses+1, noClasses))
            model:add(nn.LogSoftMax())
        return model
    end

    local split = nn.ConcatTable()
    for loopSplit = 1, noChannels do
        split:add(logSM(loopSplit))
    end


    local finalOutput = output
        - outputLayer
        - split

    local model = nn.gModule({input}, {finalOutput})
    return model
end


function calcLoss(output, target, backward)

    -- Calculate and return E, dE
    local loss = nn.SpatialClassNLLCriterion()
    local output = output
    local target = target
    local E = {}
    local avgLoss = 0

    for loopChannels = 1, #output do
        local currentLoss = loss:forward(nn.SelectTable(loopChannels):forward(output),
                             nn.SelectTable(loopChannels):forward(target))
        E[loopChannels] = currentLoss
        avgLoss = avgLoss + currentLoss
    end
    avgLoss = avgLoss / #output

    if backward == nil then return E, avgLoss end

    --Else, backward is required
    local dE_dy = {}
    for loopChannels = 1, #output do
        dE_dy[loopChannels] = loss:backward(nn.SelectTable(loopChannels):forward(output),
                                       nn.SelectTable(loopChannels):forward(target))
    end

    return E, dE_dy, avgLoss
end


-- Optional; to test model
local function testModel()
    local noChannels = 3
    model = createModel(noChannels, 12, 5, 256, 7, 3)
    inp = torch.rand(2,noChannels,32,32)
    target={}
    for i=1,noChannels do
        target[i]=torch.Tensor(2,32,32):random(255)
    end


    -- Test model forward
    local function testForward(model, inp)
        return model:forward(inp)
    end
    ok, op = pcall(testForward, model, inp)
    if ok then print ('Tested Model Forward') end


    -- Test Loss
    ok, e, de = pcall(calcLoss, op, target, true)
    if ok then print ('Tested Loss Function') end


    -- Test model forward
    local function testBackward(model, inp, d)
        model:backward(inp, d);
    end
    ok = pcall(testBackward, model, inp, de)
    if ok then print ('Tested Model Backward') end


    -- Plot/Save the model to visualize
    graph.dot(model.fg,'CNN','CNNVerify')
end

testModel()


---------------------------------------------------------
-- Create a model and return it
---------------------------------------------------------
local noChannels = 3     --No of input channels
local noFeatures = 128    --Hidden layer features
local noLayers = 15      --No of hidden layers
local noClasses = 256    --8-bit image, 1 to 256 values
local firstFiltSize = 7  --filter size at input
local genFiltSize = 3    --filter size of hidden layer

model = createModel(noChannels, noFeatures, noLayers, noClasses,
                    firstFiltSize, genFiltSize)


-- return package:
return {
   model = model,
   loss = calcLoss,
}
