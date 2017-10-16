------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss

-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'testV1.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')

local x = torch.Tensor(opt.batchSize,testData.data:size(2),
                       testData.data:size(3), testData.data:size(4))

local ytHelper = torch.Tensor(opt.batchSize,testData.labels:size(2),
                              testData.labels:size(3), testData.labels:size(4))


if opt.type == 'cuda' then
   x = x:cuda()
   ytHelper = ytHelper:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')

-- test function
function test(testData)
   model:evaluate()

   -- local vars
   local time = sys.clock()

      local nllTest = 0
      local batchEpochCountTest = 0
   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,testData:size(),opt.batchSize do
     batchEpochCountTest = batchEpochCountTest + 1
      -- disp progress
      xlua.progress(t, testData:size())

      -- batch fits?
      if (t + opt.batchSize - 1) > testData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[i]
         ytHelper[idx] = trainData.labels[i]
         idx = idx + 1
      end

      -- create yt from ytHelper
      yt = nn.SplitTable(1,3):forward(ytHelper)


      -- test sample
      local y = model:forward(x)

      local ETest
      local avgLoss
      ETest, avgLoss = loss(y,yt)
      nllTest = nllTest +avgLoss


   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   print ("Average NLL: "..nllTest/batchEpochCountTest)
   testLogger:add{['Error On Epochs'] = nllTest/batchEpochCountTest}
end

-- Export:
return test