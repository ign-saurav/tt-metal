import torch
import pickle


#####################INPUT###########################
# Load both tensors
with open("linear_1_input_tensor_torch_pcc.pkl", "rb") as f:
    t1 = pickle.load(f)

with open("linear_1_input_tensor_tt_pcc.pkl", "rb") as f:
    t2 = pickle.load(f)
    t2 = t2.to(torch.float32)

# Compare
if torch.equal(t1, t2):
    print("Input Tensors are identical")
else:
    print("Input Tensors differ")

    # Optional: show detailed difference
    diff = (t1 - t2).abs()
    print("Max difference:", diff.max().item())
    print("Mean difference:", diff.mean().item())
    print("Shapes:", t1.shape, t2.shape)
    print("Dtypes:", t1.dtype, t2.dtype)
    print("Devices:", t1.device, t2.device)
    print("Equal:", torch.equal(t1, t2))
    print("Allclose:", torch.allclose(t1, t2))

#######################OUTPUT##########################
# Load both tensors
with open("linear_1_out_tensor_tt_unit.pkl", "rb") as f:
    t1 = pickle.load(f)

with open("linear_1_out_tensor_tt_pcc.pkl", "rb") as f:
    t2 = pickle.load(f)

# Compare
if torch.equal(t1, t2):
    print("TT Output Tensors are identical")
else:
    print("TT Output Tensors differ")

    # Optional: show detailed difference
    diff = (t1 - t2).abs()
    print("Max difference:", diff.max().item())
    print("Mean difference:", diff.mean().item())
    print("Shapes:", t1.shape, t2.shape)
    print("Dtypes:", t1.dtype, t2.dtype)
    print("Devices:", t1.device, t2.device)
    print("Equal:", torch.equal(t1, t2))
    print("Allclose:", torch.allclose(t1, t2))


with open("linear_1_out_tensor_torch_unit.pkl", "rb") as f:
    t1 = pickle.load(f)

with open("linear_1_out_tensor_torch_pcc.pkl", "rb") as f:
    t2 = pickle.load(f)

# Compare
if torch.equal(t1, t2):
    print("TORCH Output Tensors are identical")
else:
    print("TORCH Output Tensors differ")

    # Optional: show detailed difference
    diff = (t1 - t2).abs()
    print("Max difference:", diff.max().item())
    print("Mean difference:", diff.mean().item())
    print("Shapes:", t1.shape, t2.shape)
    print("Dtypes:", t1.dtype, t2.dtype)
    print("Devices:", t1.device, t2.device)
    print("Equal:", torch.equal(t1, t2))
    print("Allclose:", torch.allclose(t1, t2))

###########################BIAS##########################
import numpy as np

file1 = "bias_tt_unit.txt"
file2 = "bias_tt_pcc.txt"
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

if np.array_equal(data1, data2):
    print("bias Files are identical")
else:
    print("bias Files differ")
    diff = np.abs(data1 - data2)
    print("Max difference:", diff.max())
    print("Mean difference:", diff.mean())

###########################WEIGHTS##########################
import numpy as np

file1 = "weight_tt_unit.txt"
file2 = "weight_tt_pcc.txt"
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

if np.array_equal(data1, data2):
    print("weight Files are identical")
else:
    print("weight Files differ")
    diff = np.abs(data1 - data2)
    print("Max difference:", diff.max())
    print("Mean difference:", diff.mean())
