#include <psi4-dec.h>

#include <libqt/qt.h>

#include "tensor.h"

using namespace std;
using namespace psi;

double* Tensor::tA = nullptr;
double* Tensor::tB = nullptr;
double* Tensor::tC = nullptr;
double* Tensor::tD = nullptr;

void Tensor::initialize_class(size_t nmo)
{
    size_t nwork = nmo * nmo * nmo * nmo;
    tA = new double[nwork];
    tB = new double[nwork];
    tC = new double[nwork];
    tD = new double[nwork];
}

void Tensor::finalize_class()
{
    delete[] tA;
    delete[] tB;
    delete[] tC;
    delete[] tD;
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

    std::vector<std::string> A_sum_idx;
    std::vector<std::string> A_fix_idx;
    std::vector<std::string> B_sum_idx;
    std::vector<std::string> B_fix_idx;
    for (size_t a = 0; a < A_idx.size(); ++a){
        bool contract = false;
        for (size_t b = 0; b < B_idx.size(); ++b){
            if(A_idx[a] == B_idx[b]){
                A_sum_idx.push_back(A_idx[a]);
                contract = true;
            }
        }
        if (not contract) A_fix_idx.push_back(A_idx[a]);
    }
    for (size_t b = 0; b < B_idx.size(); ++b){
        bool contract = false;
        for (size_t a = 0; a < A_idx.size(); ++a){
            if(A_idx[a] == B_idx[b]){
                B_sum_idx.push_back(B_idx[b]);
                contract = true;
            }
        }
        if (not contract) B_fix_idx.push_back(B_idx[b]);
    }

    fprintf(outfile,"\n  Contracting over the indices:");
    for (std::string& idx_sym : A_sum_idx){
        fprintf(outfile," %s",idx_sym.c_str());
    }

    // Sort the tensors
    pair<size_t,size_t> Adata = tensor_to_matrix_sort(A,A_fix_idx,A_sum_idx,tA,true);
    pair<size_t,size_t> Bdata = tensor_to_matrix_sort(B,B_fix_idx,A_sum_idx,tB,true);
    pair<size_t,size_t> Cdata = tensor_to_matrix_sort(C,A_fix_idx,B_fix_idx,tC,true);

    size_t nArows = Adata.first;
    size_t nAcols = Adata.second;
    size_t nBrows = Bdata.first;
    size_t nBcols = Bdata.second;
    size_t nsum = nAcols;

    double factor = A.factor() * B.factor();

    // Multiply the two matrices
    if(use_dgemm_){
        // C := alpha * A * B^T + beta * C,
        C_DGEMM('n','t',
                Adata.first,  // the rows of A
                Bdata.first,  // the cols of B^T
                Adata.second, // the cols of A
                factor,       // the scalar alpha, which multiplies the product of matrices
                tA,           // the array that contains A
                Adata.second,  // the leading dimension of A
                tB,           // the array that contains B^T
                Bdata.second,  // the leading dimension of B^T
                1.0,          // the scalar beta
                tC,           // the array that contains C
                Cdata.second); // the leading dimension of C
    }else{
        double sum = 0.0;
        for (size_t i = 0; i < nArows; ++i){
            for (size_t j = 0; j < nBrows; ++j){
                double sum = 0.0;
                for (size_t k = 0; k < nAcols; ++k){
                    //tC[i][j] += tA[i][k] * tB[j][k]
                    sum += factor * tA[i * nAcols + k] * tB[j * nBcols + k];
                }
                tC[i * nBrows + j] += sum;
            }
        }
    }

    // Sort the result
    tensor_to_matrix_sort(C,A_fix_idx,B_fix_idx,tC,false);
    //    std::vector<int> A_sorter;
}

template<typename Sorter>
void Tensor::sort_me(std::vector<size_t> itoj,double*& matrix,bool direct,Sorter sorter)
{
    if (ndims_ == 1){
        std::vector<size_t> j(1);
        for (size_t i0 = 0; i0 < dims_[0]; ++i0){
            j[itoj[0]] = i0;
            size_t matrix_add = sorter(j);
            size_t tensor_add = one_address(i0);
            if (direct){
                double value = t_[tensor_add];
                matrix[matrix_add] = value;
            }else{
                double value = matrix[matrix_add];
                t_[tensor_add] = value;
            }
        }
    }
    if (ndims_ == 2){
        std::vector<size_t> j(2);
        for (size_t i0 = 0; i0 < dims_[0]; ++i0){
            j[itoj[0]] = i0;
            for (size_t i1 = 0; i1 < dims_[1]; ++i1){
                j[itoj[1]] = i1;
                size_t matrix_add = sorter(j);
                size_t tensor_add = two_address(i0,i1);
                if (direct){
                    double value = t_[tensor_add];
                    matrix[matrix_add] = value;
                }else{
                    double value = matrix[matrix_add];
                    t_[tensor_add] = value;
                }
            }
        }
    }
    if (ndims_ == 3){
        std::vector<size_t> j(3);
        for (size_t i0 = 0; i0 < dims_[0]; ++i0){
            j[itoj[0]] = i0;
            for (size_t i1 = 0; i1 < dims_[1]; ++i1){
                j[itoj[1]] = i1;
                for (size_t i2 = 0; i2 < dims_[2]; ++i2){
                    j[itoj[2]] = i2;
                    size_t matrix_add = sorter(j);
                    size_t tensor_add = three_address(i0,i1,i2);
                    if (direct){
                        double value = t_[tensor_add];
                        matrix[matrix_add] = value;
                    }else{
                        double value = matrix[matrix_add];
                        t_[tensor_add] = value;
                    }
                }
            }
        }
    }
    if (ndims_ == 4){
        std::vector<size_t> j(4);
        for (size_t i0 = 0; i0 < dims_[0]; ++i0){
            j[itoj[0]] = i0;
            for (size_t i1 = 0; i1 < dims_[1]; ++i1){
                j[itoj[1]] = i1;
                for (size_t i2 = 0; i2 < dims_[2]; ++i2){
                    j[itoj[2]] = i2;
                    for (size_t i3 = 0; i3 < dims_[3]; ++i3){
                        j[itoj[3]] = i3;
                        size_t matrix_add = sorter(j);
                        size_t tensor_add = four_address(i0,i1,i2,i3);
                        if (direct){
                            double value = t_[tensor_add];
                            matrix[matrix_add] = value;
                        }else{
                            double value = matrix[matrix_add];
                            t_[tensor_add] = value;
                        }
                    }
                }
            }
        }
    }
}

std::pair<size_t, size_t> Tensor::tensor_to_matrix_sort(TensorIndexed T,
                                                        std::vector<std::string> T_left,
                                                        std::vector<std::string> T_right,
                                                        double* t,bool direct)
{
    Tensor* tens = T.tensor();
    double* td = tens->t();
    size_t ndims = tens->ndims();
    std::vector<size_t> dims = T.tensor()->dims();
    std::vector<string> indices = T.indices();

    size_t nleft  = T_left.size();
    size_t nright = T_right.size();

    // maps the indices of i to those of j, itoj: i_k -> j_itoj[k]
    std::vector<size_t> itoj(10);
    // maps the indices of j to those of i, jtoi: j_k -> i_jtoi[k]
    size_t jtoi[10];

    fprintf(outfile,"\n  The tensor:");
    T.print();
    int idx_j = 0;
    fprintf(outfile," will be sorted as the matrix: [");
    for (size_t n = 0; n < nleft; ++n){
        string index = T_left[n];
        size_t position = std::find(std::begin(indices), std::end(indices), index) - std::begin(indices);
        fprintf(outfile,"%s (%zu)",index.c_str(),position);
        itoj[position] = idx_j;
        jtoi[idx_j] = position;
        idx_j += 1;
    }
    fprintf(outfile,"][");
    for (size_t n = 0; n < nright; ++n){
        string index = T_right[n];
        size_t position = std::find(std::begin(indices), std::end(indices), index) - std::begin(indices);
        fprintf(outfile,"%s (%zu)",index.c_str(),position);
        itoj[position] = idx_j;
        jtoi[idx_j] = position;
        idx_j += 1;
    }
    fprintf(outfile,"]");

    fprintf(outfile,"\n  The indices will map as:");
    for (size_t n = 0; n < nright + nleft; ++n){
        fprintf(outfile,"(%zu -> %zu)",n,itoj[n]);
    }

    std::pair<size_t,size_t> matrix_size;
    if ((nleft == 2) and (nright == 0)){
        size_t left_size = dims[jtoi[0]] * dims[jtoi[1]];
        size_t right_size = 1;
        matrix_size = {left_size,right_size};
        size_t add_right_0 = dims[jtoi[1]];
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            return j[0] * add_right_0 + j[1];
        }
        );
    }else if ((nleft == 1) and (nright == 1)){
        size_t left_size = dims[jtoi[0]];
        size_t right_size = dims[jtoi[1]];
        matrix_size = {left_size,right_size};
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            size_t left  = j[0];
            size_t right = j[1];
            return left * right_size + right;
        }
        );
    }else if ((nleft == 0) and (nright == 2)){
        size_t left_size = 1;
        size_t right_size = dims[jtoi[0]] * dims[jtoi[1]];
        matrix_size = {left_size,right_size};
        size_t add_left_0 = dims[jtoi[1]];
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            return j[0] * add_left_0 + j[1];
        }
        );
    }else if ((nleft == 3) and (nright == 1)){
        size_t left_size = dims[jtoi[0]] * dims[jtoi[1]] * dims[jtoi[2]];
        size_t right_size = dims[jtoi[3]];
        matrix_size = {left_size,right_size};
        size_t add_left_0 = dims[jtoi[1]] * dims[jtoi[2]];
        size_t add_left_1 = dims[jtoi[2]];
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            size_t left  = j[0] * add_left_0 + j[1] * add_left_1 + j[2];
            size_t right = j[3];
            return left * right_size + right;
        }
        );
    }else if ((nleft == 2) and (nright == 2)){
        size_t left_size = dims[jtoi[0]] * dims[jtoi[1]];
        size_t right_size = dims[jtoi[2]] * dims[jtoi[3]];
        matrix_size = {left_size,right_size};
        size_t add_left_0 = dims[jtoi[1]];
        size_t add_right_2 = dims[jtoi[3]];
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            size_t left  = j[0] * add_left_0 + j[1];
            size_t right = j[2] * add_right_2 + j[3];
            return left * right_size + right;
        }
        );
    }else if ((nleft == 1) and (nright == 3)){
        size_t left_size = dims[jtoi[0]];
        size_t right_size = dims[jtoi[1]] * dims[jtoi[2]] * dims[jtoi[3]];
        matrix_size = {left_size,right_size};
        size_t add_right_1 = dims[jtoi[2]] * dims[jtoi[3]];
        size_t add_right_2 = dims[jtoi[3]];
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            size_t left  = j[0];
            size_t right = j[1] * add_right_1 + j[2] * add_right_2 + j[3];
            return left * right_size + right;
        }
        );
    }else{
        fprintf(outfile,"\n  THIS SORTING IS NOT IMPLEMENTED!!");
    }

    return matrix_size;
}


//std::pair<size_t,size_t> matrix_size;
//if (ndims == 2){
//    // loop over the original indices
//    auto sorter = sorter_2(nleft,nright,dims);

//    size_t j[2];
//    for (size_t i0 = 0; i0 < dims[0]; ++i0){
//        j[itoj[0]] = i0;
//        for (size_t i1 = 0; i1 < dims[1]; ++i1){
//            j[itoj[1]] = i1;
//            size_t left  = j[0] * left_offset + j[1];
//            size_t right = j[2] * right_offset + j[3];
//            size_t add = left * add_offset + right;
//            if (direct){
//                double value = td[tens->four_address(i0,i1,i2,i3)];
//                t[add] = value;
//            }else{
//                double value = t[add];
//                td[tens->four_address(i0,i1,i2,i3)] = value;
//            }
//        }
//    }
//}
//if (ndims == 4){
//    size_t left_size = dims[jtoi[0]] * dims[jtoi[1]];
//    size_t right_size = dims[jtoi[2]] * dims[jtoi[3]];
//    matrix_size = {left_size,right_size};

//    size_t left_offset = dims[jtoi[1]];
//    size_t right_offset = dims[jtoi[3]];
//    size_t add_offset = dims[jtoi[2]] * dims[jtoi[3]];
//    // loop over the original indices
//    for (size_t i0 = 0; i0 < dims[0]; ++i0){
//        for (size_t i1 = 0; i1 < dims[1]; ++i1){
//            for (size_t i2 = 0; i2 < dims[2]; ++i2){
//                for (size_t i3 = 0; i3 < dims[3]; ++i3){
//                    size_t j[4];
//                    j[itoj[0]] = i0;
//                    j[itoj[1]] = i1;
//                    j[itoj[2]] = i2;
//                    j[itoj[3]] = i3;
//                    size_t left  = j[0] * left_offset + j[1];
//                    size_t right = j[2] * right_offset + j[3];
//                    size_t add = left * add_offset + right;
//                    if (direct){
//                        double value = td[tens->four_address(i0,i1,i2,i3)];
//                        t[add] = value;
//                    }else{
//                        double value = t[add];
//                        td[tens->four_address(i0,i1,i2,i3)] = value;
//                    }
//                }
//            }
//        }
//    }
//}
