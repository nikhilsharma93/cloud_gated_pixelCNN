------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nngraph'
require 'math'
require 'SpatialConvolution_masked'

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
    local fuseConv = nn.SpatialConvolutionMM(factor*n_op, factor*n_op, 1, 1)

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
        -- so that 1st row of op is does not depend on ip,
        -- 2nd row of op depends on 1st row of ip,
        -- 3rd row depends on 1st and 2nd row of ip, and so on, according
        -- to kernel size

    local vConv = nn.SpatialConvolutionMM(n_ip, 2*n_op, kernelW, kernelH,
                                          1,1, padW, padH)

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
        -- so that 1st pixel in a given row does not depend on ip
        -- 2nd pixel in that row depends on the 1st
        -- 3rd pixel in that row depends on the 1st and 2nd, and so on
    padH = 0

    local hConv
    local hCropped

    -- If first layer, then ith pixel is already masked by kernel width
    -- If not, then ith pixel can the ith pixels from previous channels only
    -- Hence, we have to use a channel mask
    if isFirstLayer == true then
        hConv = nn.SpatialConvolutionMM(n_ip, 2*n_op, kernelW, kernelH,
                                        1,1, padW, padH)
        hCropped = nn.SpatialZeroPadding(0, -n_extraRows, 0, 0)
    else
        hConv = nn.SpatialConvolution_masked(n_ip, 2*n_op, kernelW, kernelH,
                                        1,1, padW, padH, maskChannels, noChannels)
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




function createModel(noChannels, noFeatures, noLayers, noClasses,
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

    local model = nn.gModule({input}, {output})
    return model
end

--model = gatedPixelUnit(10,10,3, false)
model = createModel(1, 12, 5, 4, 9,  11)
print (type(model))
graph.dot(model.fg,'CNN','CNNVerify')
