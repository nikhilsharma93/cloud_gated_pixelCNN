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
local eachSetMax = 4000
numberTrain = eachSetMax * datasetNum:size()
numberTest = 500
--print (numberTest)
local totalNoImages = numberTrain + numberTest
local embeddingSize = 10 -- equal to the number of classes

windowX = 32
windowY = 32

local allImages = torch.Tensor(totalNoImages, 3, windowY, windowX)
local allLabels = torch.Tensor(totalNoImages, 3, windowY, windowX)
local allEmbeddings = torch.zeros(totalNoImages, embeddingSize)



--Loading the Training Images
print ('Loading the Images..')

local loopVar1 = 0

--Training Data
for i = 1, datasetNum:size() do
  local takenIdxs_train = torch.zeros(10)
  local currentDir = baseFileDir..'data_batch_'..tostring(datasetNum[i])..'/'
  print ('Loading from dataset '..tostring(i))
  local eachSet = 1
  for imgName in lfs.dir(currentDir) do
      if (string.find(imgName, '.png')) then
          local name = tonumber(string.sub(imgName,7,7)) + 1
          if (eachSet <= eachSetMax and takenIdxs_train[name]<eachSetMax/10) then
          	ok,img=pcall(image.load, currentDir..imgName, 3, 'byte')
            if ok then
              --imgName starts with Label_i_, <i> represents label class
              takenIdxs_train[name] = takenIdxs_train[name] + 1
              img = img:type('torch.DoubleTensor')
              allLabels[loopVar1+1] = img:clone() + 1
              allImages[loopVar1+1] = (img - 127.5)/127.5
              --Add embedding:
              allEmbeddings[{loopVar1+1,name}] = 1
              loopVar1 = loopVar1 + 1
              eachSet = eachSet+1
            end
        end
    end
  end
  print (takenIdxs_train)
end
print ('Loaded training data: '..tostring(loopVar1))

--Testing Data
local testDir = baseFileDir..testName..'/'
local takenIdxs_test = torch.zeros(10)
for imgName in lfs.dir(testDir) do
    if (string.find(imgName,'.png')) then
        local name = tonumber(string.sub(imgName,7,7)) + 1
        if (loopVar1 < totalNoImages and takenIdxs_test[name]<numberTest/10) then
          ok,img=pcall(image.load, testDir..imgName, 3, 'byte')
          if ok then
              takenIdxs_test[name] = takenIdxs_test[name] + 1
              img = img:type('torch.DoubleTensor')
              allLabels[loopVar1+1] = img:clone() + 1
              allImages[loopVar1+1] = (img - 127.5)/127.5
              allEmbeddings[{loopVar1+1,name}] = 1
              loopVar1 = loopVar1 + 1
          end
      end
  end
end
print (takenIdxs_test)
print ('Loaded testing data: '..tostring(loopVar1))

----------------------------------------------------------------------
--Data shuffling
torch.manualSeed(123)
local labelsShuffleTrain = torch.randperm(numberTrain)
local labelsShuffleTest = torch.randperm(numberTest)

-- create train set:
trainData = {
   data = torch.Tensor(numberTrain, 3, windowY, windowX),
   embeddings = torch.zeros(numberTrain, embeddingSize),
   labels = torch.Tensor(numberTrain, 3, windowY, windowX),
   embeddingSize = embeddingSize,
   size = function() return numberTrain end,
   saveDir = currentDir..'/'..opt.save..'/imagesV_B'..tostring(opt.batchSize)..'_M'..tostring(opt.momentum)..'/'
}
--create test set:
testData = {
    data = torch.Tensor(numberTest, 3, windowY, windowX),
    embeddings = torch.zeros(numberTest, embeddingSize),
    embeddingSize = embeddingSize,
    labels = torch.Tensor(numberTest, 3, windowY, windowX),
    size = function() return numberTest end
}


for i=1,numberTrain do
   trainData.data[i] = allImages[labelsShuffleTrain[i]]:clone()
   trainData.labels[i] = allLabels[labelsShuffleTrain[i]]:clone()
   trainData.embeddings[i] = allEmbeddings[labelsShuffleTrain[i]]:clone()
end
for i=1,numberTest do
   testData.data[i] = allImages[labelsShuffleTest[i]+numberTrain]:clone()
   testData.labels[i] = allLabels[labelsShuffleTest[i]+numberTrain]:clone()
   testData.embeddings[i] = allEmbeddings[labelsShuffleTest[i]+numberTrain]:clone()
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
