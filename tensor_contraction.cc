#include <numeric>

#include <psi4-dec.h>

#include <libqt/qt.h>

#include "tensor.h"
#include "tensor_labeled.h"
#include "tensor_product.h"

using namespace std;
using namespace psi;

double* Tensor::tA = nullptr;
double* Tensor::tB = nullptr;
double* Tensor::tC = nullptr;
double* Tensor::tD = nullptr;
size_t Tensor::nwork_ = 0;

void Tensor::initialize_class(size_t nmo)
{
    nwork_ = nmo * nmo * nmo * nmo;
    tA = new double[nwork_];
    tB = new double[nwork_];
    tC = new double[nwork_];
    tD = new double[nwork_];
}

void Tensor::finalize_class()
{
    delete[] tA;
    delete[] tB;
    delete[] tC;
    delete[] tD;
}

/// Performs the operator Z +=(=) A * B * C *
void Tensor::contract(LabeledTensorProduct tp, LabeledTensor Z, bool addition)
{
    /// This algorithm loops over all the permuations of tensor contractions and finds the optimal sequence
    size_t nterms = tp.size();
    std::vector<size_t> perm(nterms);
    std::vector<size_t> best_perm(nterms);
    std::iota(perm.begin(),perm.end(),0);
    std::pair<double,double> best_cpu_memory_cost(1.0e200,1.0e200);
    do {
        if(perm[0] > perm[1]){
            std::pair<double,double> cpu_memory_cost = tp.compute_contraction_cost(perm);
            if (cpu_memory_cost.first < best_cpu_memory_cost.first){
                best_perm = perm;
                best_cpu_memory_cost = cpu_memory_cost;
                if(print_level_ > 0) fprintf(outfile,"\n  Found a better contraction pattern");
            }
        }
    } while (std::next_permutation(perm.begin(),perm.end()));

    LabeledTensor rhs = tp.tensor(perm[0]);
    for (size_t n = 0; n < nterms - 1; ++n){
//        TensorIndexed lhs = tp.tensor(perm[n+1]);
//        // create a temporary tensor

//        Tensor result;
//        result.resize("I",);

//        binary_contraction(*rhs,lhs,result,false);

//        rhs = result;
    }
    LabeledTensor last = tp.tensor(perm[nterms-1]);
    binary_contraction(rhs,last,Z,addition);
}

/// Performs the binary contraction C +=(=) A * B
void Tensor::binary_contraction(LabeledTensor A,LabeledTensor B,LabeledTensor C, bool addition)
{
    if(print_level_ > 0){
        fprintf(outfile,"\n  Performing the contraction:");
        C.print();
        fprintf(outfile," += ");
        A.print();
        B.print();
    }

    // Find the common indices between tensor A and B
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

    if(print_level_ > 0){
        fprintf(outfile,"\n  Contracting over the indices:");
        for (std::string& idx_sym : A_sum_idx){
            fprintf(outfile," %s",idx_sym.c_str());
        }
        fflush(outfile);
    }

    // Sort the input and output tensors
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
        double beta = addition ? 1.0 : 0.0;
        C_DGEMM('n','t',
                Adata.first,  // the rows of A
                Bdata.first,  // the cols of B^T
                Adata.second, // the cols of A
                factor,       // the scalar alpha, which multiplies the product of matrices
                tA,           // the array that contains A
                Adata.second,  // the leading dimension of A
                tB,           // the array that contains B^T
                Bdata.second,  // the leading dimension of B^T
                beta,          // the scalar beta
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
}

/// Performs the operator B +=(=) A
void Tensor::add(LabeledTensor A, LabeledTensor B, bool addition)
{
    if(print_level_ > 0){
        fprintf(outfile,"\n  Performing the operation:");
        B.print();
        fprintf(outfile," %s ",addition ? "+=" : "=");
        A.print();
    }

    // Find the common indices between tensor A and B
    std::vector<std::string> A_idx = A.indices();
    std::vector<std::string> B_idx = B.indices();
    std::vector<std::string> empty_idx;

//    if(print_ > 0){
//        fprintf(outfile,"\n  Contracting over the indices:");
//        for (std::string& idx_sym : A_sum_idx){
//            fprintf(outfile," %s",idx_sym.c_str());
//        }
//    }

    double factor = A.factor();

    // Sort the input and output tensors
    pair<size_t,size_t> Adata = tensor_to_matrix_sort(A,A_idx,empty_idx,tA,true);
    if(addition){
        pair<size_t,size_t> Bdata = tensor_to_matrix_sort(B,A_idx,empty_idx,tB,true);
        for (int n = 0; n < nwork_; ++n) tB[n] += factor * tA[n];
    }else{
        for (int n = 0; n < nwork_; ++n) tB[n] = factor * tA[n];
    }

    // Sort the result
    tensor_to_matrix_sort(B,A_idx,empty_idx,tB,false);
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

std::pair<size_t, size_t> Tensor::tensor_to_matrix_sort(LabeledTensor T,
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
    std::vector<size_t> itoj(nleft + nright);
    // maps the indices of j to those of i, jtoi: j_k -> i_jtoi[k]
    std::vector<size_t> jtoi(nleft + nright);

    int idx_j = 0;
    for (size_t n = 0; n < nleft; ++n){
        string index = T_left[n];
        size_t position = std::find(std::begin(indices), std::end(indices), index) - std::begin(indices);
        itoj[position] = idx_j;
        jtoi[idx_j] = position;
        idx_j += 1;
    }
    for (size_t n = 0; n < nright; ++n){
        string index = T_right[n];
        size_t position = std::find(std::begin(indices), std::end(indices), index) - std::begin(indices);
        itoj[position] = idx_j;
        jtoi[idx_j] = position;
        idx_j += 1;
    }
    if(print_level_ > 1){
        fprintf(outfile,"\n  The tensor: ");
        T.print();
        fprintf(outfile," will be sorted as the matrix: [");
        for (size_t n = 0; n < nleft; ++n){
            fprintf(outfile,"%s",T_left[n].c_str());
        }
        fprintf(outfile,"][");
        for (size_t n = 0; n < nright; ++n){
            fprintf(outfile,"%s",T_right[n].c_str());
        }
        fprintf(outfile,"]");
        fprintf(outfile,"\n  The indices will map as:");
        for (size_t n = 0; n < nright + nleft; ++n){
            fprintf(outfile,"(%zu -> %zu)",n,itoj[n]);
        }
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
    }else if ((nleft == 4) and (nright == 0)){
        size_t left_size = dims[jtoi[0]] * dims[jtoi[1]] * dims[jtoi[2]] * dims[jtoi[3]];
        size_t right_size = 1;
        matrix_size = {left_size,right_size};
        size_t add_left_0 = dims[jtoi[1]] * dims[jtoi[2]] * dims[jtoi[3]];
        size_t add_left_1 = dims[jtoi[2]] * dims[jtoi[3]];
        size_t add_left_2 = dims[jtoi[3]];
        tens->sort_me(itoj,t,direct,
                      [&](const std::vector<size_t>& j){
            return j[0] * add_left_0 + j[1] * add_left_1 + j[2] * add_left_2 + j[3];
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
        std::string msg = "The sorting "
                            + std::to_string(nleft) + ":"
                            + std::to_string(nright)
                            + " is not implemented";
        fprintf(outfile,"\n\n  %s\n\n",msg.c_str());
        fflush(outfile);
        exit(1);
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
