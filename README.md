Requirements 
------------

matplotlib

numpy

torch

torchsummary

torchvision

Installation
------------
Clone the repository via git as follows:
```
git clone https://github.com/mehdi150/function-approximation.git
cd function-approximation
```

Usage
-------------
```bash
cd function-approximation

python3 main.py <functionIdentifier> <datasetSizeTrain> <layers> <epoch> <activation>
```

* `<functionIdentifier>`: The identification number of the funtion to approximate:

    1: $`x^2 - y^2`$

    2: $`x . y . exp(-x^2 - y^2)`$

    3: $`exp({{-(x - 2)^2}\over{2 / 5}})`$

    4: $`sin(x)`$

* `<fdatasetSizeTrain>`: Size of the training set.

* `<layers>`: The hidden layers' dimensions separated with a comma (ex: 8,8,8).

* `<epoch>`: Number of epochs for the training phase.

* `<activation>`: The activation function used in the hidden layers (ex : relu, tanh, sigmoid). it must be implemented in the torch module.


Docker
-------------
```Bash
docker image build -t test .
docker run -e FUNCTION_NUM=1 -e DATASET_SIZE=10000 -e HIDDEN='8,8,8' -e EPOCHS=200 -e ACTIVATION='relu' test
```

Example
-------------

```Bash
python3 main.py 1 10000 8,8,8 100 relu 
```

The resulting plots will be :

fp32 model

![High Level](https://github.com/mehdi150/function-approximation/blob/main/fig/fp32_saddle.png)

Quantized model (QAT)

![High Level](https://github.com/mehdi150/function-approximation/blob/main/fig/int8_saddle.png)

Quantized model (dynamic)

![High Level](https://github.com/mehdi150/function-approximation/blob/main/fig/dynamic_saddle.png)

Quantized model (static)

![High Level](https://github.com/mehdi150/function-approximation/blob/main/fig/static_saddle.png)
