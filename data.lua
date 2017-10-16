------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'nn'
require 'image'
require 'sys'
require 'os'
require 'lfs'


function ls(path) return sys.split(sys.ls(path),'\n') end

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  'Loading the dataset..' .. sys.COLORS.black ..'\n')

local currentDir = lfs.currentdir()
local lastSlash = string.find(string.reverse(currentDir), "/")
local rootDir = string.sub(currentDir, 1, string.len(currentDir) - lastSlash + 1)

local baseFileDir = currentDir..'/Data/cifar-10-batches-py/extracted/'

local datasetNum = torch.LongStorage({1})
local testName = 'test_batch'


local numberTrain = 0
for i = 1, datasetNum:size() do
  local currentDir = baseFileDir..'data_batch_'..tostring(datasetNum[i])..'/'
  numberTrain = numberTrain + #ls(currentDir)
end

local numberTest = #ls(baseFileDir..testName..'/')

numberTrain = 4000
numberTest = 200
local totalNoImages = numberTrain + numberTest


windowX = 32
windowY = 32

local allImages = torch.Tensor(totalNoImages, 3, windowY, windowX)
local allLabels = torch.Tensor(totalNoImages, 3, windowY, windowX)


--Loading the Training Images
print ('Loading the Images..')

local loopVar1 = 0

--Training Data
for i = 1, datasetNum:size() do
  local currentDir = baseFileDir..'data_batch_'..tostring(datasetNum[i])..'/'
  print ('Loading from dataset '..tostring(i))
  for imgName in lfs.dir(currentDir) do
      if loopVar1 < numberTrain then
  	ok,img=pcall(image.load, currentDir..imgName, 3,'byte')
    if ok then
      allImages[loopVar1+1] = img:clone()
      allLabels[loopVar1+1] = img:type('torch.DoubleTensor') + 1
      loopVar1 = loopVar1 + 1
    end
  end
  end
end
print ('Loaded training data: '..tostring(loopVar1))

--Testing Data
local testDir = baseFileDir..testName..'/'
for imgName in lfs.dir(testDir) do
    if loopVar1 < totalNoImages then
  ok,img=pcall(image.load, testDir..imgName, 3,'byte')
  if ok then
    allImages[loopVar1+1] = img:clone()
    allLabels[loopVar1+1] = img:type('torch.DoubleTensor') + 1
    loopVar1 = loopVar1 + 1
  end
  end
end
print ('Loaded testing data: '..tostring(loopVar1))


----------------------------------------------------------------------
--Data shuffling
torch.manualSeed(123)
local labelsShuffleTrain = torch.randperm(numberTrain)
local labelsShuffleTest = torch.randperm(numberTest)

-- create train set:
trainData = {
   data = torch.Tensor(numberTrain, 3, windowY, windowX),
   labels = torch.Tensor(numberTrain, 3, windowY, windowX),
   size = function() return numberTrain end,
   saveDir = currentDir..'/results/imagesV1/'
}
--create test set:
testData = {
    data = torch.Tensor(numberTrain, 3, windowY, windowX),
    labels = torch.Tensor(numberTrain, 3, windowY, windowX),
    size = function() return numberTest end
}


for i=1,numberTrain do
   trainData.data[i] = allImages[labelsShuffleTrain[i]]:clone()
   trainData.labels[i] = allLabels[labelsShuffleTrain[i]]:clone()
end
for i=1,numberTest do
   testData.data[i] = allImages[labelsShuffleTest[i]+numberTrain]:clone()
   testData.labels[i] = allLabels[labelsShuffleTest[i]+numberTrain]:clone()
end


--Clear memory and delete allImages and allLabels
allImages = nil
allLabels = nil

--[[]
----------------------------------------------------------------------
print(sys.COLORS.red ..  'Preprocessing the data..' .. sys.COLORS.black ..'\n')
local channels = {'r','g','b'}

print ('Global Normalization\n')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end
print (mean)
print (std)
-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '\nVerify Statistics:' ..sys.COLORS.black .. '\n')


for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('       training data, '..channel..'-channel, mean:               ' .. trainMean)
   print('       training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '\nVisualization..' ..sys.COLORS.black .. '\n')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if false then --opt.visualize then
   -- Showing some training exaples
   local first128Samples = trainData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some training examples'}
   --image.save('/home/nikhil/myCode/learning/Torch/AI2/model3-PreTrained-ExtraImg/trainImages.jpg', first128Samples)
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,16} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
   --image.save('/home/nikhil/myCode/learning/Torch/AI2/model3-PreTrained-ExtraImg/testImages.jpg', first128Samples)
   local first128Samples = trainData.labels[{ {1,128} }]:reshape(128,outputSize,outputSize)
   --print (first128Samples:size())
   image.display{image=first128Samples, nrow=16, legend='Some traininglabels'}
end
]]

return {
  trainData = trainData,
  testData = testData,
  mean = mean,
  std = std
}
