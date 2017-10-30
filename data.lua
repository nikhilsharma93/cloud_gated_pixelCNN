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

local datasetNum = torch.LongStorage({1,2,3,4,5})
local testName = 'test_batch'


local numberTrain = 0
for i = 1, datasetNum:size() do
  local currentDir = baseFileDir..'data_batch_'..tostring(datasetNum[i])..'/'
  numberTrain = numberTrain + #ls(currentDir)
end

local numberTest = #ls(baseFileDir..testName..'/')

numberTrain = 2000 * datasetNum:size()
numberTest = 500
--print (numberTest)
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
  local eachSet = 1
  for imgName in lfs.dir(currentDir) do
      if eachSet <= 2000 then
  	ok,img=pcall(image.load, currentDir..imgName, 3, 'byte')
    if ok then
      img = img:type('torch.DoubleTensor')
      allLabels[loopVar1+1] = img:clone() + 1
      allImages[loopVar1+1] = (img - 127.5)/127.5
      loopVar1 = loopVar1 + 1
      eachSet = eachSet+1
    end
  end
  end
end
print ('Loaded training data: '..tostring(loopVar1))

--Testing Data
local testDir = baseFileDir..testName..'/'
for imgName in lfs.dir(testDir) do
    if loopVar1 < totalNoImages then
  ok,img=pcall(image.load, testDir..imgName, 3, 'byte')
  if ok then
      img = img:type('torch.DoubleTensor')
      allLabels[loopVar1+1] = img:clone() + 1
      allImages[loopVar1+1] = (img - 127.5)/127.5
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
   saveDir = currentDir..'/'..opt.save..'/imagesV_B'..tostring(opt.batchSize)..'_M'..tostring(opt.momentum)..'/'
}
--create test set:
testData = {
    data = torch.Tensor(numberTest, 3, windowY, windowX),
    labels = torch.Tensor(numberTest, 3, windowY, windowX),
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

----------------------------------------------------------------------
print(sys.COLORS.red ..  '\nVisualization..' ..sys.COLORS.black .. '\n')

if opt.visualize then
   local first128Samples = trainData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some training examples'}
   local first128Samples = testData.data[{ {1,64} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
end

return {
  trainData = trainData,
  testData = testData,
}
