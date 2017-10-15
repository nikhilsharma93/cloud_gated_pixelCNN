------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'os'

----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local loss = t.loss

----------------------------------------------------------------------
-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'trainV1.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = 0--opt.learningRateDecay
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')

local x = torch.Tensor(opt.batchSize,trainData.data:size(2),
                       trainData.data:size(3), trainData.data:size(4))

local ytHelper = torch.Tensor(opt.batchSize,trainData.labels:size(2),
                              trainData.labels:size(3), trainData.labels:size(4))


if opt.type == 'cuda' then
   x = x:cuda()
   ytHelper = ytHelper:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch
local saveDir = trainData.saveDir

local function train(trainData)

   model:training()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   local nll = 0
   local batchEpochCount = 0
   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())


   if epoch % 1 == 0 then
     print (sys.COLORS.blue..'Change Learning Rate?')
     local handle = io.popen("bash readEpochChange.sh")
     local content = handle:read("*a")
     handle:close()
     if string.sub(content,1,3) == "yes" then
       print (sys.COLORS.blue .. 'Chaning Learning Rate')
       optimState['learningRate'] = optimState['learningRate']/10.0--opt.learningRate/(10^math.floor(epoch/22))
     else
       print (sys.COLORS.blue .. 'Learning Rate Unchanged')
     end
   end

   -- do one epoch
   print(sys.COLORS.green .. '\n==> doing epoch on training data:')
   print ('Learning Rate for Epoch '..tostring(epoch)..': '..tostring(optimState['learningRate']))
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do

      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      batchEpochCount = batchEpochCount + 1
      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         x[idx] = trainData.data[shuffle[i]]
         ytHelper[idx] = trainData.labels[shuffle[i]]
         idx = idx + 1
      end

      -- create yt from ytHelper
      yt = nn.SplitTable(1,3):forward(ytHelper)



      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(x)


         -- Save the results to visualize
         -- Optional
         if ((epoch % 10 == 0 or epoch <=2 ) and t < 20*opt.batchSize) then
           for loopPred = 1,opt.batchSize do
               mapPred = torch.Tensor(trainData.labels:size(2),
                                      trainData.labels:size(3), trainData.labels:size(4))
              for i = 1, #y do
                  local indx
                  _, indx = torch.max(y[i][loopPred], 1)
                  mapPred[i] = indx-1
              end
             image.save(saveDir..tostring(epoch)..'_'..tostring(t)..'_'..tostring(loopPred)..'_'..'mapPred.png', mapPred:type('torch.ByteTensor'))
             trueLabel = torch.Tensor(trainData.labels:size(2),
                                    trainData.labels:size(3), trainData.labels:size(4))
             for i = 1, #y do
                 trueLabel[i] = yt[i][loopPred] - 1
             end
             image.save(saveDir..tostring(epoch)..'_'..tostring(t)..'_'..tostring(loopPred)..'_'..'label.png', trueLabel:type('torch.ByteTensor'))
           end
         end


         local E
         local dE_dy
         local avgLoss
         E, dE_dy, avgLoss = loss(y, yt, true)
         nll = nll + avgLoss

         -- backward through the model
         model:backward(x,dE_dy)

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.rmsprop(eval_E, w, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
   print ("Average NLL: "..nll/batchEpochCount)
   trainLogger:add{['Error On Epochs'] = nll/batchEpochCount}

   -- Save the model. Ask before saving
   if epoch % 1 == 0 then
     print (sys.COLORS.blue..'Save Model?')
     local handle = io.popen("bash readEpochChange.sh")
     local content = handle:read("*a")
     handle:close()
     if string.sub(content,1,3) == "yes" then
       print (sys.COLORS.blue .. 'Saving Model')
       local filename = paths.concat(opt.save, 'modelV1.t7')
       os.execute('mkdir -p ' .. sys.dirname(filename))
       print('==> saving model to '..filename)
       model1 = model:clone()
       torch.save(filename, model1:clearState())
     else
       print (sys.COLORS.blue .. 'Did Not Save The Model')
     end
   end

   -- next epoch
   epoch = epoch + 1
end

-- Export:
return train
