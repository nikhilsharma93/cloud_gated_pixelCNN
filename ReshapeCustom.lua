------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

local ReshapeCustom, parent = torch.class('nn.ReshapeCustom', 'nn.Reshape')

function ReshapeCustom:__init(noChannels, ...)
    self.noChannels = noChannels
    parent.__init(self, ...)
end

function ReshapeCustom:updateOutput(input)
    self.batchMode = true
    self.size = torch.LongStorage(input:dim())
    self.batchSize = torch.LongStorage(1)
    self.size[4] = input:size(4)
    self.size[3] = input:size(3)
    self.size[2] = input:size(2)/self.noChannels
    self.size[1] = input:size(1)*self.noChannels
    if not input:isContiguous() then
       self._input = self._input or input.new()
       self._input:resizeAs(input)
       self._input:copy(input)
       input = self._input
    end
    self.output:view(input, self.size)
    return self.output
end
