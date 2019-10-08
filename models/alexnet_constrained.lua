function createModel()
   require 'cudnn'
   local function ContConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
         local C= nn.Sequential()
          C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
          
--           C:add(cudnn.noBias())                                     -- no bias
         --  C:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3)) -- dropout
          C:add(cudnn.ReLU(true))
          return C
   end
   local function AveragePooling(kW, kH, dW, dH, padW, padH)
    return nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
   end

local features = nn.Sequential()
   features:add(ContConvolution(3,96,11,11,4,4,2,2))       
   features:add(AveragePooling(3,3,2,2))                   -- max --> average
   features:add(ContConvolution(96,256,5,5,1,1,2,2))        
   features:add(AveragePooling(3,3,2,2))                   -- max --> average
   features:add(ContConvolution(256,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialDropout(opt.dropout))            -- regularize
   features:add(ContConvolution(384,384,3,3,1,1,1,1)) 
   features:add(nn.SpatialDropout(opt.dropout))            -- regularize
   features:add(ContConvolution(384,256,3,3,1,1,1,1)) 
   features:add(AveragePooling(3,3,2,2))                   -- max --> average
   features:add(nn.SpatialDropout(opt.dropout))            -- Change or no change
   features:add(ContConvolution(256,4096,6,6))
   features:add(nn.SpatialDropout(opt.dropout))           
   features:add(ContConvolution(4096,4096,1,1))           
   features:add(nn.SpatialDropout(opt.dropout))            -- regularize?
   features:add(cudnn.SpatialConvolution(4096, nClasses,1,1))
--    features:add(cudnn.noBias())                           -- no bias
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
 
   local model = features
   

   return model
end
