import torch
import torch.nn as nn

# torch.set_printoptions(profile="full")
print("test tensor")
t1 = torch.ones(1)  
print(t1)
print(t1.shape)

t1 = torch.ones(4)  
print(t1)
print(t1.shape)

t1 = torch.ones(1,4)  
print(t1)
print(t1.shape)

# t2 = torch.ones(4,3)
# print(t2)

# t2 = torch.ones(2,3,4)
# print(t2)

# t2 = torch.ones(2,2,3,4)
# print(t2)

t = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(t)
t = t.reshape(2,2,2,2)
print(t)