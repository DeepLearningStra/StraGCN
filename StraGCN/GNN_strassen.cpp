#include<stdio.h>
#include<torch/extension.h>
#include<vector>


torch::Tensor spmm_strassen_cuda(torch::Tensor A_rowPtr,
                            torch::Tensor A_colIdx,
                            torch::Tensor A_values,
                            torch::Tensor offset,
                            torch::Tensor X);

torch::Tensor spmm_forward(
    torch::Tensor A_rowPtr,
    torch::Tensor A_colIdx,
    torch::Tensor A_values,
    torch::Tensor offset,
    torch::Tensor X,
    torch::Tensor W
    )
 {
    torch::Tensor XW = torch::mm(X, W);
    return spmm_strassen_cuda(A_rowPtr, A_colIdx, A_values, offset, XW);
}


// X.t * A.t * grad_output
std::vector<torch::Tensor> spmm_backward(
    torch::Tensor A_rowPtr,
    torch::Tensor A_colIdx,
    torch::Tensor A_values,
    torch::Tensor offset,
    torch::Tensor output_grad,
    torch::Tensor X, 
    torch::Tensor W)
{
    torch::Tensor tmp = spmm_strassen_cuda(A_rowPtr, A_colIdx, A_values, offset, output_grad);
    auto d_input = torch::mm(tmp, W.transpose(0, 1));
    auto d_output = torch::mm(X.transpose(0, 1), tmp);

    return {d_input, d_output};
}


torch::Tensor spmm_strassen(
    torch::Tensor A_rowPtr,
    torch::Tensor A_colIdx,
    torch::Tensor A_values,
    torch::Tensor offset,
    torch::Tensor X
    )
 {
    return spmm_strassen_cuda(A_rowPtr, A_colIdx, A_values, offset, X);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spmm_forward, "GCN forward");
  m.def("backward", &spmm_backward, "GNN backward");
  m.def("spmmstra", &spmm_strassen, "SPMM strassen");
}

