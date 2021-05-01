import torch
import h5py #hdf5 file

##2.3 Size, storage offset and strides  2019.12.30
a = torch.zeros(1,3)
print("zeros =", a)
print("shape of zeros:", a.shape) #you can ask tensor about its shape
print("size of zeros:", a.size())
print("storage of zeros:\n", a.storage()) #the layout of a storage is one-dimensional
points = torch.tensor([[1, 2], [3, 4], [5, 6]])
print("\ntensor of points:\n", points)
second_point = points[1].clone() #changing subtensor has side effect on the original tensor too, clone can avoid this effect
second_point[0] = 4
print("\nclone has changed the original tensor?", id(points[1]) == id(second_point))
print("transpose of points:\n", points.t())   #transpose
print("stride of points:", points.stride())
print("stride of transposed points:", points.t().stride())
print("points is contiguous?", points.is_contiguous())
points_con = points.t().contiguous()
print("transpose of points is contiguous?", points.t().is_contiguous(), " after contiguous method to points.t():", points_con.is_contiguous())

##2.4 Numeric types
double_points = torch.zeros(10, 2).double() #method
short_points = torch.tensor([[1, 2], [3, 4]], dtype = torch.short)  #argument
char_points = torch.CharTensor([[1, 2]])    #default
print("\ndtype of double float:", double_points.dtype)
print("dtype of short int:", short_points.dtype)
print("dtype of char:", char_points.dtype)

rand_points = torch.randn(10, 2).type(torch.short)  #method
print("\nyou can cast a tensor of one type as a tensor of another type by using 'type' method:\n#rand_points = torch.randn(10, 2).type(torch.short)\ndtype of rand_points:", rand_points.dtype)

##2.6 NumPy interoperability
points = torch.tensor([[1, 1, 1], [2, 2, 2]])
points_np = points.numpy()
print("from NumPy out of tensor:\n", points_np) #the array shares an underlying buffer with the tensor storage
tensor_points_np = torch.from_numpy(points_np) #same buffer sharing strategy
print("obtain a PyTorch tensor from a NumPy array:\n", tensor_points_np)

##2.7 Serializing tensors
f = h5py.File('../test_points.hdf5', 'w')
dset = f.create_dataset('test_points', data = points.numpy())
f.close()   #When you finish writing or loading data, close the file
f = h5py.File('../test_points.hdf5', 'r')
dset = f['test_points']
array_last_points = dset[:-1]  #h5py return a NumPy array-like object behaves like a NumPy array and has the same API
tensor_last_points = torch.from_numpy(array_last_points)
f.close()
print("\nloaded test points from h5py file:\n", tensor_last_points)

##2.9 The tensor API    2020.01.06
#a small number of operations exist only as methods of the tensor object (by the trailing underscore in their name such as zero_ which indicate that the method operates in-place by modifying the input instead of creating a new output tensor and returning it)
