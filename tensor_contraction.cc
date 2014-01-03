#include <psi4-dec.h>

#include "tensor.h"

using namespace std;
using namespace psi;

double* Tensor::tA = nullptr;
double* Tensor::tB = nullptr;
double* Tensor::tC = nullptr;

void undefined(int n,int m){};

void Tensor::initialize_class()
{
}

void Tensor::finalize_class()
{
}

//for (int i = 0; i < 5; ++i){
//    for (int j = 0; j < 5; ++j){
//        tensor_to_matrix_sorter[i][j] = nullptr;
//    }
//}

void Tensor::evaluate(TensorIndexed A,TensorIndexed B,TensorIndexed C)
{
    fprintf(outfile,"\n  Performing the contraction:");
    C.print();
    fprintf(outfile," += ");
    A.print();
    B.print();

    // Find the common indices
    std::vector<std::string> A_idx = A.indices();
    std::vector<std::string> B_idx = B.indices();
    std::vector<std::string> C_idx = C.indices();

    std::vector<std::pair<size_t,std::string> > A_sum_idx;
    std::vector<std::pair<size_t,std::string> > A_fix_idx;
    std::vector<std::pair<size_t,std::string> > B_sum_idx;
    std::vector<std::pair<size_t,std::string> > B_fix_idx;
    for (size_t a = 0; a < A_idx.size(); ++a){
        bool contract = false;
        for (size_t b = 0; b < B_idx.size(); ++b){
            if(A_idx[a] == B_idx[b]){
                A_sum_idx.push_back(make_pair(a,A_idx[a]));
                contract = true;
            }
        }
        if (not contract) A_fix_idx.push_back(make_pair(a,A_idx[a]));
    }
    for (size_t b = 0; b < B_idx.size(); ++b){
        bool contract = false;
        for (size_t a = 0; a < A_idx.size(); ++a){
            if(A_idx[a] == B_idx[b]){
                B_sum_idx.push_back(make_pair(b,B_idx[b]));
                contract = true;
            }
        }
        if (not contract) B_fix_idx.push_back(make_pair(b,B_idx[b]));
    }

    fprintf(outfile,"\n  Contracting over the indices:");
    for (std::pair<size_t,std::string>& idx_sym : A_sum_idx){
        fprintf(outfile," %s",idx_sym.second.c_str());
    }

    // Sort the tensors
    tensor_to_matrix_sort(A,A_fix_idx,A_sum_idx,tA);
//    tensor_to_matrix_sort(B,B_fix_idx,A_sum_idx,tB);

//    // Multiply the two matrices

//    // Sort the result
//    matrix_to_tensor_sort(C,A_fix_idx,B_fix_idx,tC);

//    std::vector<int> A_sorter;


}

void Tensor::tensor_to_matrix_sort(TensorIndexed T,
                       std::vector<std::pair<size_t,std::string> > T_left,
                       std::vector<std::pair<size_t,std::string> > T_right,
                       double* t)
{
    size_t nleft  = T_left.size();
    size_t nright = T_right.size();
    size_t ndims = T.tensor()->ndims();

    fprintf(outfile,"\n  The tensor:");
    T.print();
    fprintf(outfile," will be sorted as the matrix: [");
    for (size_t n = 0; n < nleft; ++n){
        string index = T_left[n].second;
        fprintf(outfile,"%s",index.c_str());
    }
    fprintf(outfile,"][");
    for (size_t n = 0; n < nright; ++n){
        string index = T_right[n].second;
        fprintf(outfile,"%s",index.c_str());
    }
    fprintf(outfile,"]");

    if (nleft >= nright){
//        tensor_to_matrix_sorter[nleft][nright]();
    }else{
//        tensor_to_matrix_sorter[nright][nleft]();
//        transpose();
    }

}



void Tensor::matrix_to_tensor_sort(TensorIndexed T,
                       std::vector<std::pair<size_t,std::string> > T_left,
                       std::vector<std::pair<size_t,std::string> > T_right,
                       double* t)
{

}
