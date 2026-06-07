# PyTorch Custom Convolution Layer

This project is about how to define a custom convolution layer in PyTorch, and use CUDA function to implement convolution.

## Content

[/cpp](https://github.com/Qwesh157/pytorch_custom_convolution_layer/blob/main/cpp) C++ extension include CUDA interface and Python module bind.

[/cuda](https://github.com/Qwesh157/pytorch_custom_convolution_layer/tree/main/cuda) Implicit gemm convolution implementation.

[/include](https://github.com/Qwesh157/pytorch_custom_convolution_layer/tree/main/include) Declaration about forward/backward convolution.

[/pytorch](https://github.com/Qwesh157/pytorch_custom_convolution_layer/tree/main/pytorch) Include setup.py script, custom convolution layer definition.

## Build

```bash
sh setup.sh
```

The script runs the unified extension build:

```bash
python ./pytorch/setup.py build_ext --inplace
```

It builds all four custom CUDA operators:

- `conv2d_baseline_fp32`
- `conv2d_optim_fp32`
- `conv2d_optim_fp16`
- `conv2d_optim_wmma`

Run the build command from the `Lab6_experiment` project root. The custom
operators require CUDA tensors and do not provide a CPU fallback.

## TEST

```bash
$ cd pytorch
vim test.py
```


## RUN

run inference program

```bash
python inference.py --model fp32
```

Available model choices are `baseline`, `fp32`, `fp16`, and `wmma`.

Useful options:

```bash
python inference.py --model fp16 --run-times 10 --accuracy-samples 50
python inference.py --model wmma --plot --save-plots-dir docs/figures/runtime
python inference.py --model fp32 --batch-sizes 8,16,32,64,128,256,512
```

The WMMA operator is intended for inference. Its backward path fails loudly
instead of returning silent zero gradients.
