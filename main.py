import sys
import warnings

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from MyClasses import (MyModel, QuantModel, MyDataSet,
                       train, post_train,
                       functions)
from Draw import plot_1d, plot_2d
from copy import deepcopy

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    functionNum = int(sys.argv[1])
    # The identification number of the funtion
    # to approximate:
    # 1: x^2-y^2
    # 2: x*y*exp(-x^2-y^2)
    # 3: exp(-(x-2)^2/(2/5))
    # 4: sin(x)

    datasetSizeTrain = int(sys.argv[2])
    # Size of the training set

    layers = str(sys.argv[3])
    # The hidden layers' dimensions separated with a comma (ex: 8,8,8)

    epoch = int(sys.argv[4])
    # Number of epochs

    activation = str(sys.argv[5])
    # The activation function (ex : relu, tanh, sigmoid)

    # Definition of the different parameters
    batch = 100
    lr = 1e-3
    loss = torch.nn.MSELoss()
    activation = getattr(torch, activation)
    funct = functions[functionNum]
    architecture = [funct[2]] + [int(x) for x in layers.split(',')] + [1]
    noise = 0

    # Choosing the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"We are using the following device: {device}\n\n")

    # Defining the fp32 model and the int8 model
    model_fp32 = MyModel(architecture, activation).to(device)
    model_int8 = MyModel(architecture, activation, quant=True).to('cpu')

    print(summary(model_fp32, (1, funct[2])))

    # Creating the datasets
    # We need two datasets because the quantization function
    # do not work on GPU
    dataset = MyDataSet(func=funct[0],
                        size=(datasetSizeTrain, funct[2]),
                        interval=funct[1],
                        device=device, noise=noise)

    dataset_quant = MyDataSet(func=funct[0],
                              size=(datasetSizeTrain, funct[2]),
                              interval=funct[1],
                              device='cpu', noise=noise)

    train_loader = DataLoader(dataset, batch_size=batch,
                              shuffle=True)

    train_loader_quant = DataLoader(dataset_quant, batch_size=batch,
                                    shuffle=True)

    # Training the different models
    print("Training standard fp32 model ...\n")

    optimizer = Adam(model_fp32.parameters(), lr=lr)
    train(model_fp32, optimizer, loss, train_loader, epoch)

    print("Training int8 model ...\n")
    optimizer = Adam(model_int8.parameters(), lr=lr)
    train(model_int8, optimizer, loss, train_loader_quant,
          epoch, awaire_training=True)

    print("Training with dynamic quantization ...\n")
    model_dynamic = post_train(model=deepcopy(model_fp32).to('cpu'),
                               quant_type="dynamic-quant",
                               data_loader=train_loader_quant)

    print("Training with static quantization ...\n")
    model_static = post_train(model=QuantModel(deepcopy(model_fp32).to('cpu')),
                              quant_type="static-quant",
                              data_loader=train_loader_quant)

    # Ploting the results
    print("Ploting Curves ...")

    if funct[2] == 1:
        plot_1d(funct, model_fp32, dataset, "fp32 model ", device)
        plot_1d(funct, model_int8, dataset_quant, "int8 model ", 'cpu')
        plot_1d(funct, model_dynamic, dataset_quant, "dynamic model ", 'cpu')
        plot_1d(funct, model_static, dataset_quant, "static model ", 'cpu')

    elif funct[2] == 2:
        plot_2d(funct, model_fp32, "fp32 model ", device)
        plot_2d(funct, model_int8, "int8 model ", 'cpu')
        plot_2d(funct, model_dynamic, "dynamic model ", 'cpu')
        plot_2d(funct, model_static, "static model ", 'cpu')
