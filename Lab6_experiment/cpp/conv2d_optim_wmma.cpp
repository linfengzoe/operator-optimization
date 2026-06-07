#include <torch/extension.h>
#include <vector>
#include "conv2d_fp16.h"

void conv2d_cuda_forward(param_t param);
void conv2d_cuda_backward(param_t param);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    param_t param;
    param.input  = reinterpret_cast<DTYPE*>(input.data_ptr<torch::Half>());
    param.weight = reinterpret_cast<DTYPE*>(weight.data_ptr<torch::Half>());
    param.n = input.size(0);
    param.c = input.size(1);
    param.h = input.size(2);
    param.w = input.size(3);
    param.k = weight.size(0);
    param.r = weight.size(2);
    param.s = weight.size(3);
    param.u = stride[0];
    param.v = stride[1];
    param.p = padding[0];
    param.q = padding[1];

    int64_t outh = (param.h - param.r + 2 * param.p) / param.u + 1;
    int64_t outw = (param.w - param.s + 2 * param.q) / param.v + 1;
    param.Oh = outh;
    param.Ow = outw;

    auto output = torch::zeros(
        torch::IntArrayRef({input.size(0), weight.size(0), outh, outw}),
        input.options().dtype(torch::kHalf));
    param.output = (DTYPE*)output.data_ptr();
    conv2d_cuda_forward(param);
    return output;
}

std::vector<torch::Tensor> conv2d_backward(
    torch::Tensor input,
    torch::Tensor grad_output,
    torch::Tensor weight,
    torch::IntArrayRef stride,
    torch::IntArrayRef padding)
{
    TORCH_CHECK(false,
        "conv2d_optim_wmma is forward-only: backward gradients are not implemented. "
        "Use it for inference, or switch to baseline/fp32/fp16 operators for training.");
    return {torch::Tensor(), torch::Tensor()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward",  &conv2d_forward,  "WMMA ImplicitGEMM forward (CUDA)");
    m.def("backward", &conv2d_backward, "WMMA ImplicitGEMM backward (forward-only)");
}
