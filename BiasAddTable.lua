------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------


local BiasAddTable, parent = torch.class('nn.BiasAddTable', 'nn.CAddTable')

function BiasAddTable:__init(ip)
   parent.__init(self, ip)
end

function BiasAddTable:updateOutput(input)
   local inp1 = input[1]
   local inp2 = input[2]
   local numX = inp1:size(3)
   local numY = inp1:size(4)

   --replicate input 2 to match dimension of inp1
   local repInp2 = nn.Replicate(numX, 2, 1):forward(inp2)
   repInp2 = nn.Replicate(numY, 3, 2):forward(repInp2)
   return parent.updateOutput(self, {inp1, repInp2})
end

function BiasAddTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)

   local numChannels = gradOutput:size(2)
   local viewer = nn.View(numChannels,-1):setNumInputDims(3) --batch mode assumed
   local gradOutput2Avg = viewer:forward(gradOutput:clone()):mean(3)
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[2]:resizeAs(input[2]):copy(gradOutput2Avg)

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
    end
    return self.gradInput
end
