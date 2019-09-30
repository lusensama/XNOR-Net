m = torch.load('alexnet_BWN.t7')
m = m:float()
torch.save('alexnet_BWN.cpu.t7',m) 
