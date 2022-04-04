from typing import Callable

import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset


# Definition of the functions that will be approximated
functions = {
    1: [lambda x: x[:, 0]**2 - x[:, 1]**2,  # the function
        [-50, 50],  # the interval
        2],  # the input dimension
    2: [lambda x: x[:, 0] * x[:, 1] * torch.exp(-x[:, 0]**2 - x[:, 1]**2),
        [-4, 4],
        2],
    3: [lambda x: torch.exp(-((x - 2)**2) / (2 / 5)),
        [2 - 5 * 1 / np.sqrt(5), 2 + 5 * 1 / np.sqrt(5)],
        1],
    4: [lambda x: torch.sin(x),
        [-2*np.pi, 2*np.pi],
        1],
    }


class MyDataSet(Dataset):
    """
    This class generates a random dataset.
    """
    def __init__(self, func: Callable, size: int, interval: list,
                 device: str = 'cpu', noise: float = 0.):
        """
        :param func: The function that will be used to generate data
        :param size: The size of the dataset generated
        :param interval: The interval of the created data
        :param device: The device to use (GPU or CPU)
        :param noise: The power of the gaussian noise
        """
        super().__init__()
        self.device = device
        self.x = ((interval[1] - interval[0])
                  * torch.rand(size).to(self.device)
                  + interval[0])
        self.y = (func(self.x) + noise *
                  torch.randn(func(self.x).size()).to(self.device))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class MyModel(nn.Module):
    """
    This class builds a multilayer Perceptron.
    """
    def __init__(self, dims: int, activations: Callable,
                 output_activation: Callable = lambda x: x,
                 quant: bool = False):
        """
        :param dims: The dimensions of the layers including the input
            and output layer
        :param activations: The activation functions of the hidden layers
        :param output_activation: The activation function of the output layer
        :param quant: To indicate if we use the quantization mode
        """
        super().__init__()
        self.quant = quant
        self.activations = activations
        self.output = output_activation
        if self.quant:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                      for i in range(len(dims) - 1)])

    def forward(self, x):
        if self.quant:
            x = self.quant(x)
        for i in range(len(self.linears)-1):
            x = self.linears[i](x)
            x = self.activations(x)
        x = self.output(self.linears[-1](x))
        if self.quant:
            x = self.dequant(x)
        return x


class QuantModel(nn.Module):
    """
    This class builds a quantized multilayer Perceptron with a fp32 model.
    """
    def __init__(self, model_fp32: torch.nn.Module):
        """
        :param model_fp32: A fp32 model to quantize
        """
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def train(model: torch.nn.Module,
          optimizer: torch.optim,
          loss: Callable,
          train_loader: torch.utils.data.dataloader,
          epochs: int,
          awaire_training: bool = False):

    """
    This function trains the model with a given range of epochs,
    a given optimizer, a given loss function and a given dataset.

    :param model: The model to train
    :param optimizer: The optimizer used for training
    :param loss: The loss funtion used for training
    :param train_loader: Train dataset
    :param epochs: Number of epochs
    :param awaire_training: A boolean that indicate if we use QAT
    """

    if awaire_training:
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)

    for _ in range(epochs):
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            error = loss(output,
                         torch.reshape(labels, output.size()))  # compute loss
            error.backward()  # compute gradients
            optimizer.step()  # update model

    if awaire_training:
        model.eval()
        torch.quantization.convert(model, inplace=True)


def post_train(model: torch.nn.Module,
               quant_type: str,
               data_loader: torch.utils.data.dataloader):

    """
    This function quantizes a tained model.

    :param model: A model to quantize
    :param quant_type: Indicate the type of quantization; static or dynamic
    :param dataLoader: Dataset to calibrate the static quantization
    """

    model.eval()
    if quant_type == "static-quant":
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        for inputs, _ in data_loader:
            _ = model(inputs)
        torch.quantization.convert(model, inplace=True)

    elif quant_type == "dynamic-quant":
        model = torch.quantization.quantize_dynamic(model,
                                                    {torch.nn.Linear},
                                                    dtype=torch.qint8)

    return model
