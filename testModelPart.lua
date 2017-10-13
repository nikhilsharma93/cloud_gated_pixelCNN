require 'nn'
require 'SpatialConvolution_masked'

local function maskChannels(weights, n_ip, n_op, kW, noChannels)
    -- There are 2*n_op features that come out of the conv layer
    -- Mask them individually
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
        --print (pixelPos)
        --print (weights:size())
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

n_ip = 12
n_op = 12
kernelW = 3
kernelH = 1
padW = 2
padH = 0
noChannels = 3
hConv = nn.SpatialConvolution_masked(n_ip, 2*n_op, kernelW, kernelH,
                                1,1, padW, padH, maskChannels, noChannels)
