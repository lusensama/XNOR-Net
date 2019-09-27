function createModel()
   require 'cudnn'
   local function ContConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
         local C= nn.Sequential()
          C:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))   
          C:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
          C:add(cudnn.ReLU(true))
          return C
   end
   local function MaxPooling(kW, kH, dW, dH, padW, padH)
    return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   end

local features = nn.Sequential()
   features:add(ContConvolution(3,64,3,3,2,2,1,1))       -- 224 -> 55
   features:add(MaxPooling(2,2,1,1))                   -- 55 ->  27
   features:add(ContConvolution(64,192,3,3,1,1,1,1))       --  27 -> 27  
   features:add(MaxPooling(2,2,1,1))                     --  27 ->  13
   features:add(ContConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(ContConvolution(384,256,3,3,1,1,1,1)) 
   features:add(ContConvolution(256,256,3,3,1,1,1,1)) 
   features:add(MaxPooling(2,2,1,1))           
   features:add(nn.SpatialDropout(opt.dropout))
   features:add(ContConvolution(256,4096,2,2))
   features:add(nn.SpatialDropout(opt.dropout))           
   features:add(ContConvolution(4096,4096,1,1)) 
   features:add(cudnn.SpatialConvolution(4096, nClasses,1,1))
   features:add(nn.View(nClasses))
   features:add(nn.LogSoftMax())
 
   local model = features
   

   return model
end
