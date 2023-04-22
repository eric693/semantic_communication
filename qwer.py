import torch

def to_fixed2(f, e):
    a = f * (2**e)
    b = torch.round(a).long()
    neg_mask = a < 0
    # Apply 2's complement only to negative numbers
    b[neg_mask] = torch.abs(b[neg_mask])
    b[neg_mask] = ~b[neg_mask] + 1
        
    return b


"""
x is the input fixed number which is of integer datatype
e is the number of fractional bits for example in Q1.15 e = 15
"""
def to_float(x,e):
    c = torch.abs(x)
    sign = torch.sign(x)
    if torch.any(torch.lt(x, 0)):
        # convert back from two's complement
        c = x - 1 
        c = torch.bitwise_not(c)
        sign = -1
    f = c.to(torch.float) / (2 ** e)
    f = torch.round(f * 10000) / 10000  # 將浮點數四捨五入到小數點後四位
    f = f * sign
    return f


NUM_BITS = 16

def fixed_point2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, dtype=torch.int16)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2fixed_point(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, dtype=torch.int16)
    return torch.sum(mask * b, -1).to(torch.int16)

print("float_point :")
f = torch.tensor([-0.0933, 0.2283, -0.9999, 0.9982])
print(f)
e = 15
fixed_point = to_fixed2(f, e)
print("float_point convert fixed_point :")
print(fixed_point)

fixed_point = fixed_point2bin(fixed_point, NUM_BITS)
print("fixed_point convert bit vector :")
pred1 = fixed_point.squeeze(0)
print(pred1)

print("bit vector convert fixed_point :")
pred1 = bin2fixed_point(pred1, NUM_BITS)
pred2 = pred1.unsqueeze(0)
print(pred2)

#fixed = torch.tensor([-500, 35825])
#pred3 = fixed.unsqueeze(0)
#print(pred3)
e = 15
print("fixed_point convert float_point :")
float_ = to_float(pred2, e)
print(float_)



