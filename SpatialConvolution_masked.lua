------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------


local SpatialConvolution_masked, parent = torch.class('nn.SpatialConvolution_masked', 'nn.SpatialConvolution')

function SpatialConvolution_masked:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, mask, noChannels)
   self.mask = mask
   self.noChannels = noChannels
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
end

function SpatialConvolution_masked:reset(stdv)
   parent.reset(self, stdv)
   self.mask(self.weight, self.nInputPlane, self.nOutputPlane, self.kW, self.noChannels)
end

function SpatialConvolution_masked:updateOutput(input)
   self.mask(self.weight, self.nInputPlane, self.nOutputPlane, self.kW, self.noChannels)
   return parent.updateOutput(self, input)
end
