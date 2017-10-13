local SpatialConvolutionMM_masked, parent = torch.class('nn.SpatialConvolutionMM_masked', 'nn.SpatialConvolutionMM')

function SpatialConvolutionMM_masked:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, mask)
   self.mask = mask
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
end

function SpatialConvolutionMM_masked:reset(stdv)
   parent.reset(self, stdv)
   self.mask(self.weight)
end

function SpatialConvolutionMM_masked:updateOutput(input)
   self.mask(self.weight)
   return parent.updateOutput(self, input)
end
