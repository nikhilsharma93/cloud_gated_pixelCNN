------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nngraph'
require 'math'


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




local function gatedPixelUnit(n_ip, n_op, filtSize, isFirstLayer)

    -- Define the previous vertical and horizontal stack
    local vStackIn = - nn.Identity() --make it a nngraph node
    local hStackIn = - nn.Identity() --make it a nngraph node


    -------------------------------------
    --Compute the output vertical stack
    -------------------------------------

    -- As hinted in the paper, the n*n masked convolution can be done by
    -- ceil(n/2)*n kernel with appropriate padding and cropping
    local kernelW = filtSize
    local kernelH = math.ceil(filtSize/2)
    local padW = math.floor(filtSize/2) -- so that width of op = width of ip
    local padH = math.ceil(filtSize/2)
        -- so that 1st row of op is does not depend on ip,
        -- 2nd row of op depends on 1st row of ip,
        -- 3rd row depends on 1st and 2nd row of ip, and so on

    local vConv = nn.SpatialConvolutionMM(n_ip, 2*n_op, kernelW, kernelH,
                                          1,1, padW, padH)

    -- Note that the op of this layer has extra rows at the bottom, due to padding
    -- Crop those rows out
    local n_extraRows = math.ceil(filtSize/2) + 1 --number of extra rows
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
    -- ceil(n/2)*1 kernel with appropriate padding and cropping
    kernelW = math.ceil(filtSize/2)
    kernelH = 1
    padW = math.ceil(filtSize/2)
        -- so that 1st pixel in a given row does not depend on ip
        -- 2nd pixel in that row depends on the 1st
        -- 3rd pixel in that row depends on the 1st and 2nd, and so on
    padH = 0

    local hConv = nn.SpatialConvolutionMM(n_ip, 2*n_op, kernelW, kernelH,
                                          1,1, padW, padH)

    -- Here cropping depends on whether or not it is the first layer
    -- If yes, then it is of mask type A in the paper
    -- If no, then it is of mask type B in the paper
    local hCropped
    if isFirstLayer == true then
        hCropped = nn.SpatialZeroPadding(0, -n_extraRows, 0, 0)
    else
        hCropped = nn.SpatialZeroPadding(-1, -n_extraRows+1, 0, 0)
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
        layers:add(gatedPixelUnit(noFeatures, noFeatures, genFiltSize, false))
    end

    local output = {input, nn.Identity()(input)}
        - gatedPixelUnit(noChannels, noFeatures, firstFiltSize, true)
        - layers

    local model = nn.gModule({input}, {output})
    return model
end

--model = gatedPixelUnit(10,10,3, false)
model = createModel(1, 10, 5, 4, 7, 5)
print (type(model))
graph.dot(model.fg,'CNN','CNNVerify')
