from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

import torch


def plot_1d_function(x_train: np.array,
                     x_eval: np.array,
                     predictions1: np.array,
                     predictions2: np.array,
                     labels: np.array,
                     name: str):

    """
    This function displays 2D plots.

    :param x_train: List of trainig data
    :param x_eval: List of evaluation data
    :param predictions1: The prediction of the NN using eval data
    :param predictions2: The prediction of the NN using train data
    :param labels: The title of our figure
    :param name: The title of the figure
    """

    fig = plt.figure(1, figsize=(18, 6))

    # Ploting the model prediction and actual function
    ax = fig.add_subplot(1, 2, 1)
    ax.axvspan(x_train.flatten().min(), x_train.flatten().max(),
               alpha=0.15, color='limegreen')  # delimit the training interval
    plt.plot(x_train, predictions2, '.', color='royalblue')
    plt.plot(x_eval, labels, '-', label='output',
             color='darkorange', linewidth=2.0)
    plt.plot(x_eval, predictions1, '-', color='firebrick', linewidth=2.5)
    plt.grid(which='both')
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Prediction and Actual Function')
    plt.legend(['Training domain', 'Training set',
                'Function f(x)', 'MLP output g(x)'])

    # Absolute difference between prediction and actual function
    ax = fig.add_subplot(1, 2, 2)
    ax.axvspan(x_train.flatten().min(), x_train.flatten().max(),
               alpha=0.15, color='limegreen')
    plt.plot(x_eval, np.abs(labels-predictions1), '-', label='output',
             color='firebrick', linewidth=2.0)
    plt.grid(which='both')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Absolute Difference Between Prediction and Actual Function')
    plt.legend(['Training domain', 'Error |f(x)-g(x)|'])
    fig.suptitle(name)
    plt.savefig(fname=name, bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def plot_1d(func: Callable,
            model: torch.nn.Module,
            dataset: torch.utils.data.Dataset,
            name: str,
            device: str = 'cpu'):

    """
    This function displays 3D plots.

    :param func: List containing the function to approximate
        the training interval and the input dimension
    :param model: The approximation model
    :param dataset: The training dataset
    :param name: The figue title
    :param device: The device to use (GPU or CPU)
    """

    x_train = dataset.x.cpu().detach()
    mean = (func[1][0]+func[1][1])/2
    var = func[1][1]-mean
    x_eval = np.arange(mean-2*var, mean+2*var, 0.025)
    x = torch.arange(mean-2*var, mean+2*var, 0.025).to(device)
    x = torch.reshape(x, (x.size()[0], 1))
    predictions1 = model(x).cpu().detach().numpy().reshape(x.size()[0],)
    predictions2 = dataset.y.cpu().detach()
    labels = func[0](x.reshape(len(x))).cpu().numpy()

    plot_1d_function(x_train, x_eval, predictions1, predictions2, labels, name)


def plot_2d_function(X1: np.array, X2: np.array, Y: np.array,
                     res: np.array, X12: np.array, X22: np.array,
                     Y2: np.array, res2: np.array, name: str):

    """
    This function displays 3D plots.

    :param X1: List of eval data's fisrt dimension
    :param X2: List of eval data's second dimension
    :param Y: The output of the approximated function
    :param res: The prediction of the NN
    :param X11: List of eval data's fisrt dimension, in the
        training interval
    :param X22: List of eval data's second dimension, in the
        training interval
    :param Y2: The output of the approximated function, in the
        training interval
    :param res: The prediction of the NN, in the training interval
    :param name: The figure title
    """

    fig = plt.figure(1, figsize=(32, 12))
    plt.rcParams.update({'font.size': 18})

    # Ploting the actual function
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    im = ax.plot_trisurf(X1.flatten(), X2.flatten(), Y.flatten(),
                         cmap='viridis', linewidth=0.2,
                         antialiased=True)

    plt.xlabel('x1')
    plt.ylabel('x2')
    ax = fig.add_subplot(2, 4, 5)
    im = ax.contourf(X1, X2, Y, levels=20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    fig.colorbar(im)
    plt.title('Function.')
    # Ploting the model prediction
    ax = fig.add_subplot(2, 4, 2, projection='3d')
    im = ax.plot_trisurf(X1.flatten(), X2.flatten(), res.flatten(),
                         linewidth=0.2, antialiased=True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax = fig.add_subplot(2, 4, 6)
    im = ax.contourf(X1, X2, res, levels=20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    fig.colorbar(im)
    plt.title('Prediction.')
    # Ploting the Absolute difference between prediction and
    # actual function
    ax = fig.add_subplot(2, 4, 3, projection='3d')
    im = ax.plot_trisurf(X1.flatten(), X2.flatten(),
                         np.abs(res-Y).flatten(),
                         cmap='viridis', linewidth=0.2,
                         antialiased=True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax = fig.add_subplot(2, 4, 7)
    im = ax.contourf(X1, X2, np.abs(res-Y), levels=20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    fig.colorbar(im)
    plt.title('Absolute difference between \n \
               prediction and actual function')
    # Ploting the Absolute difference between prediction and
    # actual function in the training interval
    ax = fig.add_subplot(2, 4, 4, projection='3d')
    im = ax.plot_trisurf(X12.flatten(), X22.flatten(),
                         np.abs(res2-Y2).flatten(),
                         cmap='viridis', linewidth=0.2,
                         antialiased=True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax = fig.add_subplot(2, 4, 8)
    im = ax.contourf(X12, X22, np.abs(res2-Y2), levels=20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    fig.colorbar(im)
    plt.title('Absolute difference between \n \
               prediction and actual function Zoom')
    fig.suptitle(name)
    plt.savefig(fname=name, bbox_inches='tight')
    plt.close(fig)
    # plt.show()


def plot_2d(func: Callable,
            model: torch.nn.Module,
            name: str,
            device: str = 'cpu'):

    """
    This function displays 3D plots.

    :param func: List containing the function to approximate
        the training interval and the input dimension
    :param model: The approximation model
    :param name: The figue title
    :param device: The device to use (GPU or CPU)
    """

    step = (func[1][1] - func[1][0])/200
    x_eval = np.arange(2*func[1][0], 2*func[1][1], 2*step)
    x2, y2 = np.meshgrid(x_eval, x_eval)

    coords = []
    for a, b in zip(x2, y2):
        for a1, b1 in zip(a, b):
            coords.append((a1, b1))

    x = torch.tensor(coords).to(device)

    z = np.reshape(func[0](x).cpu().detach().numpy(),
                   (len(x_eval), len(x_eval)))
    predictions1 = model(x.float()).cpu().detach().numpy()
    Z = np.reshape(predictions1, (len(x_eval), len(x_eval)))

    x_eval = np.arange(func[1][0], func[1][1], step)
    x22, y22 = np.meshgrid(x_eval, x_eval)

    coords = []

    for a, b in zip(x22, y22):
        for a1, b1 in zip(a, b):
            coords.append((a1, b1))

    x = torch.tensor(coords).to(device)

    z2 = np.reshape(func[0](x).cpu().detach().numpy(),
                    (len(x_eval), len(x_eval)))

    predictions1 = model(x.float()).cpu().detach().numpy()
    Z2 = np.reshape(predictions1, (len(x_eval), len(x_eval)))

    plot_2d_function(x2, y2, z, Z, x22, y22, z2, Z2, name)
