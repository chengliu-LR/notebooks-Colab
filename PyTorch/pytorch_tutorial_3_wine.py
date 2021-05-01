import torch
import csv
import numpy as np

###3.1 Tabular data (typically isn't homogeneous)
wine_path = "../DL_PyTorch/dlwpt_code_master/data/p1ch4/tabular-wine/wine_quality.csv"
wine_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter = ";", skiprows = 1)
print("number of wine samples and features:\n", wine_numpy.shape)
print("\ndata type of wine_numpy samples:\n", wine_numpy.dtype)
col_list = next(csv.reader(open(wine_path), delimiter = ';'))
print("\nname of columns:\n", col_list)
wine_tensor = torch.from_numpy(wine_numpy)
print("\ntransfered tensor of wine sample:\n", wine_tensor.size(), wine_tensor.type())
#attention: interval, ordinal and categorical values
#remove score from the input tensor and keep it in a separate wine_tensor
data = wine_tensor[:, :-1]
target = wine_tensor[:, -1].long()  #scatter_ expect object of type long

#one-hot encoding is much better fit if target is purely qualitative, as no implied ordering or distance is involved
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0) #for each row, take the index of the target label and use it as the column index to set the value 1.0
#print("unsqueezed target:", target.unsqueeze(1).shape)    #dim equals to 1
#print(target.unsqueeze(1).shape)
print("\none-hot encoding of the scores:\n", target_onehot)
data_mean = torch.mean(data, dim = 0, keepdim = False)  #dim = 0 indicates that the reduction is performed along dimension 0
data_var = torch.var(data, dim = 0)
data_normalized = (data - data_mean) / torch.sqrt(data_var) #normalize the data, which helps with the learning processs

bad_wine_indices = torch.le(target, 3)    #less or equal, return bool
bad_wine_indices_sum = bad_wine_indices.sum()
print("\nthere are", bad_wine_indices_sum.item(), "bad wines")    #item(): to convert a 0-dim tensor to a python number

bad_data = data[bad_wine_indices]   #advanced indexing, using binary tensor to index the data tensor
print("\nusing advanced indexing to filter raw data to bad_wine_data:\n", bad_data.shape, '\n')
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = torch.mean(bad_data, dim = 0)
mid_mean = torch.mean(mid_data, dim = 0)
good_mean = torch.mean(good_data, dim = 0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:25} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
    #*arg: pointer to the zip elements
total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
predicted_bad_indices = torch.lt(total_sulfur_data, total_sulfur_threshold) #predicted bad wines indices
actual_bad_indices = torch.gt(target, 5)

n_bad_matches = torch.sum(predicted_bad_indices & actual_bad_indices).item()    #number of right prediction
n_predicted_bad = torch.sum(predicted_bad_indices).item()
n_actual_bad = torch.sum(actual_bad_indices).item()
print('\nnumber of prediction & actual bad wine that matchs:', n_bad_matches, '\npercentage of high quality in your prediction:', '{:6.2}'.format(n_bad_matches / n_predicted_bad))
