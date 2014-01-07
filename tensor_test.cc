#include <cmath>
#include "tensor.h"
#include "multidimensional_arrays.h"
#include "psi4-dec.h"

#define loop_mo_p for(int p = 0; p < nmo; ++p)
#define loop_mo_q for(int q = 0; q < nmo; ++q)
#define loop_mo_r for(int r = 0; r < nmo; ++r)
#define loop_mo_s for(int s = 0; s < nmo; ++s)
#define loop_mo_t for(int t = 0; t < nmo; ++t)
#define loop_mo_u for(int u = 0; u < nmo; ++u)
#define loop_mo_v for(int v = 0; v < nmo; ++v)
#define loop_mo_x for(int x = 0; x < nmo; ++x)
#define loop_mo_y for(int y = 0; y < nmo; ++y)
#define loop_mo_z for(int z = 0; z < nmo; ++z)

#define MAXTWO 10
#define MAXFOUR 10
double a2[MAXTWO][MAXTWO];
double b2[MAXTWO][MAXTWO];
double c2[MAXTWO][MAXTWO];
double a4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double b4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];
double c4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

using namespace psi;

std::pair<std::string,bool> test_1_1_sort();
std::pair<std::string,bool> test_2_2_sort();

/// This function computes the difference between the matrix elements of a
/// tensor and a matrix
double difference_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO]);

/// This function initializes a tensor and a matrix with the same sequence of
/// random numbers
void initialize_random_2(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);

void initialize_random_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);
double difference_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);

void test_tensor_class()
{
    Tensor::initialize_class(MAXFOUR);
    test_1_1_sort();
    test_2_2_sort();
    Tensor::finalize_class();
}

std::pair<std::string,bool> test_1_1_sort()
{
    fprintf(outfile,"\n Testing a A(p,r) * B(r,q)");
    size_t n0 = 3;
    size_t n1 = 5;
    size_t n2 = 7;
    std::vector<size_t> dimsA = {n0,n1};
    std::vector<size_t> dimsB = {n1,n2};
    std::vector<size_t> dimsC = {n0,n2};

    Tensor A2("A2",dimsA);
    Tensor B2("B2",dimsB);
    Tensor C2("C2",dimsC);

    initialize_random_2(A2,a2);
    double a_diff = difference_2(A2,a2);
    fprintf(outfile,"\n A(p,q) error: %e",a_diff);

    initialize_random_2(B2,b2);
    double b_diff = difference_2(B2,b2);
    fprintf(outfile,"\n B(p,q) error: %e",b_diff);

    C2.zero();
    C2("pq") += A2("pr") * B2("rq");

    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < n2; ++k){
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }
    double c_diff = difference_2(C2,c2);
    fprintf(outfile,"\n C(p,q) error: %e",c_diff);
    return std::make_pair("A(p,r) * B(r,q)",std::fabs(c_diff) < 1.0e-12);
}

std::pair<std::string,bool> test_2_2_sort()
{
    fprintf(outfile,"\n Testing a A(p,q,t,u) * B(t,u,r,s)");
    size_t n0 = 3;
    size_t n1 = 5;
    size_t n2 = 7;
    size_t n3 = 9;
    size_t n4 = 8;
    size_t n5 = 6;
    std::vector<size_t> dimsA = {n0,n1,n4,n5};
    std::vector<size_t> dimsB = {n4,n5,n2,n3};
    std::vector<size_t> dimsC = {n0,n1,n2,n3};

    Tensor A4("A4",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_4(A4,a4);
    double a_diff = difference_4(A4,a4);
    fprintf(outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    double b_diff = difference_4(B4,b4);
    fprintf(outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("pqrs") += A4("pqtu") * B4("turs");

    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            for (size_t k = 0; k < n2; ++k){
                for (size_t l = 0; l < n3; ++l){
                    c4[i][j][k][l] = 0.0;
                    for (size_t m = 0; m < n4; ++m){
                        for (size_t n = 0; n < n5; ++n){
                            c4[i][j][k][l] += a4[i][j][m][n] * b4[m][n][k][j];
                        }
                    }
                }
            }
        }
    }
    double c_diff = difference_4(C4,c4);
    fprintf(outfile,"\n C error: %e",c_diff);
    return std::make_pair("A(p,q,t,u) * B(t,u,r,s)",std::fabs(c_diff) < 1.0e-12);
}
/*
    int nmo = 3;
    std::vector<size_t> n4 = {nmo,nmo,nmo,nmo};
    Tensor A("A",n4);
    Tensor B("B",n4);
    Tensor C("C",n4);
    double**** a;
    double**** b;
    double**** c;
    init_matrix<double>(a,nmo,nmo,nmo,nmo);
    init_matrix<double>(b,nmo,nmo,nmo,nmo);
    init_matrix<double>(c,nmo,nmo,nmo,nmo);
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        a[p][q][r][s] = 1.0 / double(1 + p + 3 * q + 5 * r + 7 * s);
        A(p,q,r,s) = 1.0 / double(1 + p + 3 * q + 5 * r + 7 * s);
        b[p][q][r][s] = (1.0 + double(1 + p + 3 * q + 5 * r + 7 * s)) / double(2.0+p+q+r+s);
        B(p,q,r,s) = (1.0 + double(1 + p + 3 * q + 5 * r + 7 * s)) / double(2.0+p+q+r+s);
    }
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        double sum = 0.0;
        loop_mo_t loop_mo_u{
            sum += a[p][q][t][u] * b[t][u][r][s] + b[p][q][t][u] * a[t][u][r][s];
        }
        c[p][q][r][s] = sum;
    }
    C.zero();
    C("pqrs") += A("pqtu") * B("turs");
    C("pqrs") += B("pqtu") * A("turs");
    double sumabs = 0.0;
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        fprintf(outfile,"\n%2d %2d %2d %2d %20.12f %20.12f %20.12e",p,q,r,s,c[p][q][r][s],C(p,q,r,s),C(p,q,r,s)-c[p][q][r][s]);
        sumabs += std::fabs(C(p,q,r,s)-c[p][q][r][s]);
    }
    fprintf(outfile,"\n sum of abs errors %e",sumabs);
    */

double difference_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO])
{
    double sum = 0.0;
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            sum += std::fabs(tensor(i,j) - matrix[i][j]);
        }
    }
    return sum;
}


void initialize_random_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO])
{
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            double randnum = 1.0 / (1.0 + static_cast<double>(std::rand()));
            tensor(i,j) = randnum;
            matrix[i][j] = randnum;
        }
    }
}

double difference_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    double sum = 0.0;
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    size_t n2 = tensor.dims(2);
    size_t n3 = tensor.dims(3);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            for (size_t k = 0; k < n2; ++k){
                for (size_t l = 0; l < n3; ++l){
                    sum += std::fabs(tensor(i,j,k,l) - matrix[i][j][k][l]);
                }
            }
        }
    }
    return sum;
}


void initialize_random_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    size_t n2 = tensor.dims(2);
    size_t n3 = tensor.dims(3);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            for (size_t k = 0; k < n2; ++k){
                for (size_t l = 0; l < n3; ++l){
                    double randnum = 1.0 / (1.0 + static_cast<double>(std::rand()));
                    tensor(i,j,k,l) = randnum;
                    matrix[i][j][k][l] = randnum;
                }
            }
        }
    }
}
