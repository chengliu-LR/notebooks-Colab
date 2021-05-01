import torch
import imageio
import os

###3.4 Images
img_dir = "../DL_PyTorch/dlwpt-code-master/data/p1ch4/image-dog/bobby.jpg"
img_arr = imageio.imread(img_dir)
print('shape of the dog.jpg picture:(720*1280 pixels, 3 RGB)\n', img_arr.shape)
img = torch.from_numpy(img_arr)
print("\nstride of the image tensor:\n", img.stride())
img_out = torch.transpose(img, 0, 2)    #PyTorch modules that deal with image data require tensors to be laid out as C*H*W (channels, height, width) respectively. Changing a pixel in img leads to a change in img_out.
print("stride after transpose:\n", img_out.stride())

batch_size = 3  #3 figures
batch = torch.zeros(batch_size, 4, 256, 256, dtype = torch.uint8)   #each of the 4 color channel is represented as a 8-bit integer, as in most photographic formats from standard consumer cameras.
data_dir = "../DL_PyTorch/dlwpt-code-master/data/p1ch4/image-cats/"
#os.listdir returns a list containing the names of the enries in the directory given by path
filenames = [filename for filename in os.listdir(data_dir) if os.path.splitext(filename)[1] == '.png']  #splitext split the pathname into a tuple(root, ext)
print("\ninput picture for the neural networks:\n", filenames)
print('\nshape of the cat.png picture:\n', batch.shape)

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img = torch.from_numpy(img_arr)
    batch[i] = img.transpose(0, 2)  #you can also use transpose() as torch function: torch.transpose(img, 0, 2) rather than a tensor method

#transfer to float and normalization
batch = batch.float()
#batch /= 255    #255 is the maximum representable number in 8-bit unsigned
n_channels = batch.shape[1]
print("\none of the pixels before normalization:")
for c in range(n_channels):
    print(batch[:, c, 45, 45])
    #output:
    #tensor([192., 222., 204.])
    #tensor([176., 187., 165.])
    #tensor([170., 145., 125.])
    #tensor([255., 255., 255.])
    #you can see the 3 column of outputs represents different batchs, 4 rows represents 4 RGBs

#another way:scale the batch of zero mean and unit standard deviation
print("\nafter using normalization of the first picture batch:")
for c in range(n_channels): #normalize along each channel
    batch_mean  = torch.mean(batch[:, c])
    batch_std   = torch.std(batch[:, c])
    batch[:, c] = (batch[:, c] - batch_mean) / batch_std
    print(batch[:, c, 45, 45])
print('\n')

###3.5 Volumetric data
dir_path = "../DL_PyTorch/dlwpt-code-master/data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083"
vol_arr = imageio.volread(dir_path, 'DICOM')
print("\nraw volumetric data:\n", vol_arr.shape)

#due to the lack of channel information, you have to make room for channel dimensions
vol = torch.from_numpy(vol_arr).float()
vol = torch.transpose(vol, 0, 2)
vol = torch.unsqueeze(vol, 0)
print("\nAfter adding channel information, you could assemble a 5D data N*C*Depth*H*W set by stacking followed multiple volumes along the batch direction:\n", vol.shape)

#to help you better understand unsqueeze(tensor, 0), run this demo and you can see the difference between two tensors: a and b
#import torch
#a = torch.tensor([1, 2, 3, 4])
#b = torch.unsqueeze(a, 0)
#print(b)
#print(b.shape)
#print(a.shape)
