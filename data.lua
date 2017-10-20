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

local baseFileDir = currentDir..'/data_mnist/mnist_png/'

local datasetNum = torch.LongStorage({0,1,2,3,4,5,6,7,8,9})


local numberTrain = 0
for i = 1, datasetNum:size() do
  local currentDir = baseFileDir..'data_batch_'..tostring(datasetNum[i])..'/'
  numberTrain = numberTrain + #ls(currentDir)
end

local numberTest-- = #ls(baseFileDir..testName..'/')

numberTrain = 10*500
numberTest = 10*50
local totalNoImages = numberTrain + numberTest


windowX = 28
windowY = 28

local allImages = torch.Tensor(totalNoImages, 1, windowY, windowX)
local allLabels = torch.Tensor(totalNoImages, windowY, windowX)


--Loading the Training Images
print ('Loading the Images..')

local loopVar1 = 0

--Training Data
for i = 1, datasetNum:size() do
  local currentDir = baseFileDir..'training/'..tostring(datasetNum[i])..'/'
  print ('Loading from dataset '..tostring(i))
  local eachSet = 1
  for imgName in lfs.dir(currentDir) do
      if (eachSet <= 500) then
  	ok,img=pcall(image.load, currentDir..imgName)
    if ok then
      img:apply(function(x) if x < 0.5 then return 0.0 else return 1.0 end end);
      allImages[loopVar1+1] = img:clone()
      allLabels[loopVar1+1] = img:clone()
      loopVar1 = loopVar1 + 1
      eachSet = eachSet+1
    end
  end
  end
end
print ('Loaded training data: '..tostring(loopVar1))

--Testing Data
for i = 1, datasetNum:size() do
  local currentDir = baseFileDir..'training/'..tostring(datasetNum[i])..'/'
  print ('Loading from dataset '..tostring(i))
  local eachSet = 1
  for imgName in lfs.dir(currentDir) do
      if (eachSet <= 50) then
  	ok,img=pcall(image.load, currentDir..imgName)
    if ok then
      img:apply(function(x) if x < 0.5 then return 0.0 else return 1.0 end end);
      allImages[loopVar1+1] = img:clone()
      allLabels[loopVar1+1] = img:clone()
      loopVar1 = loopVar1 + 1
      eachSet = eachSet+1
    end
  end
  end
end
print ('Loaded testing data: '..tostring(loopVar1))
print (allImages:size())
print (allLabels:size())


----------------------------------------------------------------------
--Data shuffling
torch.manualSeed(123)
local labelsShuffleTrain = torch.randperm(numberTrain)
local labelsShuffleTest = torch.randperm(numberTest)

-- create train set:
trainData = {
   data = torch.Tensor(numberTrain, 1, windowY, windowX),
   labels = torch.Tensor(numberTrain, windowY, windowX),
   size = function() return numberTrain end,
   saveDir = currentDir..'/results1/imagesV_B'..tostring(opt.batchSize)..'_M'..tostring(opt.momentum)..'/'
}
--create test set:
testData = {
    data = torch.Tensor(numberTrain, 1, windowY, windowX),
    labels = torch.Tensor(numberTrain, windowY, windowX),
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



return {
  trainData = trainData,
  testData = testData
}
