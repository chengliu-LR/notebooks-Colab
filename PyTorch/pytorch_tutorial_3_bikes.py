import torch
import numpy as np
import csv

###3.2 Time series
bikes_np = np.loadtxt("../DL_PyTorch/dlwpt_code_master/data/p1ch4/bike-sharing-dataset/hour_fixed.csv",
                    dtype = np.float32,
                    delimiter = ',',
                    skiprows = 1,
                    converters = {1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_np)
daily_bikes = bikes.view(-1, 24, bikes.shape[1])    #the size -1 is inferred from other dimensions, calling view on a tensor returns a new tensor that changes the number of dimensions and the striding information without changing the storage
daily_bikes = daily_bikes.transpose(1, 2)   #get the N*C*L ordering
print("the new bike rental tensor ordered in 730 days, 17 features and 24 hours:\n", daily_bikes.shape, "and the stride:", daily_bikes.stride())
#print(daily_bikes[:,9,:].unsqueeze(1).shape)
#print(daily_bikes[:,9,:].shape)
first_day = bikes[:24].long()   #long() for scatter
weather_onehot = torch.zeros(first_day.shape[0], 4)
weather_onehot.scatter_(
                    dim = 1,
                    index = first_day[:,9].unsqueeze(1) - 1,
                    #unsqueeze to defunction the slice(after slice, the first_day[:,9] is a 0-dim tensor)
                    value = 1.0)
bikes_cat_first_day = torch.cat(tensors=(bikes[:24], weather_onehot), dim=1)  #any no-empty tensors provided must have the same shape except the cat dimension. tips: when you pick a dim, the size along this dim increases as you concatenate tensors along this dim
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
daily_weather_onehot.scatter_(
                    dim = 1,
                    index = daily_bikes[:,9,:].long().unsqueeze(1) - 1,
                    #after slice, we got a [730, 24] tensor, but we need a 3-dim tensor, that's why we use unsqueeze(1)
                    value = 1.0)
print("\nthe daily_weather_onehot:\n", daily_weather_onehot.shape)
print(daily_weather_onehot[:5], "\nabove is the first 5 days of 24 hours of weather condition encoded in onehot.\n")
daily_bikes = torch.cat(tensors=(daily_bikes, daily_weather_onehot), dim=1) #up to now, we got the new daily_bikes tensor combined with the onehot weather tensor

temp = daily_bikes[:,10,:]
print("temp before manipulating the interval:\n", temp[0])
daily_bikes[:,10,:] = (daily_bikes[:,10,:] - torch.mean(temp)) / torch.std(temp)    #in this case, the temp variable has zero mean and unitary standard deviation
print("temp after manipulating the interval:\n", daily_bikes[0,10,:])
