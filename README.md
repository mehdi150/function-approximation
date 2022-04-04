# Zama Challenge

Requirements 
------------

Installation
------------
Clone the repository via git as follows:
```
git clone https://github.com/mehdi150/zama_challenge.git
cd zama_challenge
```

Usage
-------------
```
cd zama_challenge

python3 main.py <functionIdentifier> <datasetSizeTrain> <layers> <epoch> <activation>
```

* ```<functionIdentifier>```: The identification number of the funtion to approximate:

    1: $x**2 - y**2$

    2: `$x.y.exp(-x**2 - y**2)$`

    3: `$exp(-(x - 2)**2 / \over{(2 / 5)})$`

    4: `$sin(x)$`

* ```<fdatasetSizeTrain>```: Size of the training set.

* ```<layers>```: The hidden layers' dimensions separated with a comma (ex: 8,8,8).

* ```<epoch>```: Number of epochs for the training phase.

* ```<activation>```: The activation function used in the hidden layers (ex : relu, tanh, sigmoid). it must be implemented in the torch module.

Example
-------------

```
python3 main.py 1 10000 8,8,8 100 relu 
```

The resulting plots will be :

fp32 model

![High Level](https://github.com/mehdi150/zama_challenge/blob/main/fp32_saddle.png)

Quantized model (QAT)

![High Level](https://github.com/mehdi150/zama_challenge/blob/main/int8_saddle.png)

Quantized model (dynamic)

![High Level](https://github.com/mehdi150/zama_challenge/blob/main/dynamic_saddle.png)

Quantized model (dstatic)

![High Level](https://github.com/mehdi150/zama_challenge/blob/main/static_saddle.png)