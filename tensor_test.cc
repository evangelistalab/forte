#include <cmath>
#include "btl.h"
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
double d4[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR];

std::pair<std::string,double> test_Cij_Aik_Bjk();
std::pair<std::string,double> test_Cij_Aik_Bkj();

std::pair<std::string,double> test_Cijkl_Aijmn_Bmnkl();
std::pair<std::string,double> test_Cijkl_Aijmn_Bnmkl();
std::pair<std::string,double> test_Cijkl_Aimjn_Bkmln();

std::pair<std::string,double> test_Cij_Aiklm_Bjklm();

std::pair<std::string,double> test_Cjkli_Ailjm_Bkm();
std::pair<std::string,double> test_Cklji_Aml_Bjmki();
std::pair<std::string,double> test_Ailjk_Ailjm_Bkm();

std::pair<std::string,double> test_Cji_Aklim_Bmjlk();
std::pair<std::string,double> test_Cij_Aikjl_Bkl();
std::pair<std::string,double> test_Cijkl_Aij_Bkl();

// Triple contractions
std::pair<std::string,double> test_Dijkl_Aijmn_Bkm_Cln();


/// These functions computes the difference between the matrix elements of a
/// tensor and a matrix
std::pair<double,double> difference_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO]);
std::pair<double,double> difference_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);

/// These functions initialize a tensor and a matrix with the same sequence of
/// random numbers
void initialize_random_2(Tensor& tensor, double matrix[MAXTWO][MAXTWO]);
void initialize_random_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR]);

bool verbose_test = false;

bool test_tensor_class(bool verbose)
{
    verbose_test = verbose;
    double err_threshold = 1.0e-11;
    Tensor::initialize_class(MAXFOUR);
    std::vector<std::pair<std::string,double>> results;
    results.push_back(test_Cij_Aik_Bjk());
    results.push_back(test_Cij_Aik_Bkj());

    results.push_back(test_Cij_Aikjl_Bkl());

    results.push_back(test_Cij_Aiklm_Bjklm());
    results.push_back(test_Cji_Aklim_Bmjlk());

    results.push_back(test_Cijkl_Aij_Bkl());

    results.push_back(test_Cjkli_Ailjm_Bkm());
    results.push_back(test_Cklji_Aml_Bjmki());
    results.push_back(test_Ailjk_Ailjm_Bkm());

    results.push_back(test_Cijkl_Aijmn_Bmnkl());
    results.push_back(test_Cijkl_Aijmn_Bnmkl());
    results.push_back(test_Cijkl_Aimjn_Bkmln());

    results.push_back(test_Dijkl_Aijmn_Bkm_Cln());

    bool success = true;
    for (auto sb : results){
        if (std::fabs(sb.second) > err_threshold) success = false;
    }

    if(verbose_test){
        fprintf(psi::outfile,"\n\n Summary of tests:");

        fprintf(psi::outfile,"\n %-50s %12s %s","Test","Max. error","Result");
        fprintf(psi::outfile,"\n %s",std::string(60,'-').c_str());
        for (auto sb : results){
            fprintf(psi::outfile,"\n %-50s %7e %s",sb.first.c_str(),sb.second,std::fabs(sb.second) < err_threshold ? "Passed" : "Failed");
        }
        fprintf(psi::outfile,"\n %s",std::string(60,'-').c_str());
        fprintf(psi::outfile,"\n Tests: %s",success ? "All passed" : "Some failed");
    }



    Tensor::finalize_class();
    return success;
}

std::pair<std::string,double> test_Cij_Aik_Bkj()
{
    std::string test = "C2(\"ij\") += A2(\"ik\") * B2(\"kj\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());

    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    std::vector<size_t> dimsA = {ni,nk};
    std::vector<size_t> dimsB = {nk,nj};
    std::vector<size_t> dimsC = {ni,nj};

    Tensor A2("A2",dimsA);
    Tensor B2("B2",dimsB);
    Tensor C2("C2",dimsC);

    initialize_random_2(A2,a2);
    std::pair<double,double> a_diff = difference_2(A2,a2);
    fprintf(psi::outfile,"\n A2 error: sum = %e max = %e",a_diff.first,a_diff.second);

    initialize_random_2(B2,b2);
    std::pair<double,double> b_diff = difference_2(B2,b2);
    fprintf(psi::outfile,"\n B2 error: sum = %e max = %e",b_diff.first,b_diff.second);

    C2.zero();
    C2("ij") += A2("ik") * B2("kj");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] += a2[i][k] * b2[k][j];
            }
        }
    }
    std::pair<double,double> C_diff = difference_2(C2,c2);
    fprintf(psi::outfile,"\n C(p,q) error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,C_diff.second);
}

std::pair<std::string,double> test_Cij_Aik_Bjk()
{
    std::string test = "C2(\"ij\") += A2(\"ik\") * B2(\"jk\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 9;
    size_t nj = 6;
    size_t nk = 7;
    std::vector<size_t> dimsA = {ni,nk};
    std::vector<size_t> dimsB = {nj,nk};
    std::vector<size_t> dimsC = {ni,nj};

    Tensor A2("A2",dimsA);
    Tensor B2("B2",dimsB);
    Tensor C2("C2",dimsC);

    initialize_random_2(A2,a2);
    std::pair<double,double> a_diff = difference_2(A2,a2);
    fprintf(psi::outfile,"\n A2 error: sum = %e max = %e",a_diff.first,a_diff.second);

    initialize_random_2(B2,b2);
    std::pair<double,double> b_diff = difference_2(B2,b2);
    fprintf(psi::outfile,"\n B2 error: sum = %e max = %e",b_diff.first,b_diff.second);

    C2.zero();
    C2("ij") += A2("ik") * B2("jk");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                c2[i][j] += a2[i][k] * b2[j][k];
            }
        }
    }
    std::pair<double,double> C_diff = difference_2(C2,c2);
    //    fprintf(psi::outfile,"\n C2 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Cjkli_Ailjm_Bkm()
{
    std::string test = "C4(\"jkli\") += A4(\"iljm\") * B4(\"km\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;

    std::vector<size_t> dimsA = {ni,nl,nj,nm};
    std::vector<size_t> dimsB = {nk,nm};
    std::vector<size_t> dimsC = {nj,nk,nl,ni};

    Tensor A4("A4",dimsA);
    Tensor B2("B2",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_2(B2,b2);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("jkli") += A4("iljm") * B2("km");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[j][k][l][i] = 0.0;
                    for (size_t m = 0; m < nm; ++m){
                        c4[j][k][l][i] += a4[i][l][j][m] * b2[k][m];
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(C4,c4);
    //    fprintf(psi::outfile,"\n C4 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Cklji_Aml_Bjmki()
{
    std::string test = "C4(\"klji\") += A4(\"ml\") * B4(\"jmki\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;

    std::vector<size_t> dimsA = {nm,nl};
    std::vector<size_t> dimsB = {nj,nm,nk,ni};
    std::vector<size_t> dimsC = {nk,nl,nj,ni};

    Tensor A2("A2",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_2(A2,a2);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("klji") += A2("ml") * B4("jmki");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[k][l][j][i] = 0.0;
                    for (size_t m = 0; m < nm; ++m){
                        c4[k][l][j][i] += a2[m][l] * b4[j][m][k][i];
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(C4,c4);
    //    fprintf(psi::outfile,"\n C4 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Ailjk_Ailjm_Bkm()
{
    std::string test = "A4(\"iljk\") = A4(\"iljm\") * B4(\"km\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;

    std::vector<size_t> dimsA = {ni,nl,nj,nk};
    std::vector<size_t> dimsB = {nk,nk};
    std::vector<size_t> dimsC = {ni,nl,nj,nk};

    Tensor A4("A4",dimsA);
    Tensor B2("B2",dimsB);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_2(B2,b2);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    A4("iljk") = A4("iljm") * B2("km");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][l][j][k] = 0.0;
                    for (size_t m = 0; m < nk; ++m){
                        c4[i][l][j][k] += a4[i][l][j][m] * b2[k][m];
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(A4,c4);
    //    fprintf(psi::outfile,"\n C4 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Cijkl_Aijmn_Bmnkl()
{
    std::string test = "C4(\"ijkl\") += A4(\"ijmn\") * B4(\"mnkl\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;
    size_t nn = 6;

    std::vector<size_t> dimsA = {ni,nj,nm,nn};
    std::vector<size_t> dimsB = {nm,nn,nk,nl};
    std::vector<size_t> dimsC = {ni,nj,nk,nl};

    Tensor A4("A4",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("ijkl") += A4("ijmn") * B4("mnkl");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][j][k][l] = 0.0;
                    for (size_t m = 0; m < nm; ++m){
                        for (size_t n = 0; n < nn; ++n){
                            c4[i][j][k][l] += a4[i][j][m][n] * b4[m][n][k][l];
                        }
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(C4,c4);
    //    fprintf(psi::outfile,"\n C4 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Cijkl_Aijmn_Bnmkl()
{
    std::string test = "C4(\"ijkl\") += A4(\"ijmn\") * B4(\"nmkl\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;
    size_t nn = 6;

    std::vector<size_t> dimsA = {ni,nj,nm,nn};
    std::vector<size_t> dimsB = {nn,nm,nk,nl};
    std::vector<size_t> dimsC = {ni,nj,nk,nl};

    Tensor A4("A4",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("ijkl") += A4("ijmn") * B4("nmkl");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][j][k][l] = 0.0;
                    for (size_t m = 0; m < nm; ++m){
                        for (size_t n = 0; n < nn; ++n){
                            c4[i][j][k][l] += a4[i][j][m][n] * b4[n][m][k][l];
                        }
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(C4,c4);
    //    fprintf(psi::outfile,"\n C4 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}


std::pair<std::string,double> test_Cijkl_Aimjn_Bkmln()
{
    std::string test = "C4(\"ijkl\") += A4(\"imjn\") * B4(\"kmln\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;
    size_t nn = 6;

    std::vector<size_t> dimsA = {ni,nm,nj,nn};
    std::vector<size_t> dimsB = {nk,nm,nl,nn};
    std::vector<size_t> dimsC = {ni,nj,nk,nl};

    Tensor A4("A4",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("ijkl") += A4("imjn") * B4("kmln");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][j][k][l] = 0.0;
                    for (size_t m = 0; m < nm; ++m){
                        for (size_t n = 0; n < nn; ++n){
                            c4[i][j][k][l] += a4[i][m][j][n] * b4[k][m][l][n];
                        }
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(C4,c4);
    //    fprintf(psi::outfile,"\n C4 error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}


std::pair<std::string,double> test_Cij_Aiklm_Bjklm()
{
    std::string test = "C2(\"ij\") += A4(\"iklm\") * B4(\"jklm\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;
    std::vector<size_t> dimsA = {ni,nk,nl,nm};
    std::vector<size_t> dimsB = {nj,nk,nl,nm};
    std::vector<size_t> dimsC = {ni,nj};

    Tensor A4("A4",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C2("C2",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C2.zero();
    C2("ij") += A4("iklm") * B4("jklm");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    for (size_t m = 0; m < nm; ++m){
                        c2[i][j] += a4[i][k][l][m] * b4[j][k][l][m];
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_2(C2,c2);
    //    fprintf(psi::outfile,"\n C(p,q,r,s) error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}


std::pair<std::string,double> test_Cji_Aklim_Bmjlk()
{
    std::string test = "C2(\"ji\") += A4(\"klim\") * B4(\"mjlk\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;
    std::vector<size_t> dimsA = {nk,nl,ni,nm};
    std::vector<size_t> dimsB = {nm,nj,nl,nk};
    std::vector<size_t> dimsC = {nj,ni};

    Tensor A4("A4",dimsA);
    Tensor B4("B4",dimsB);
    Tensor C2("C2",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_4(B4,b4);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C2.zero();
    C2("ji") += A4("klim") * B4("mjlk");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[j][i] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    for (size_t m = 0; m < nm; ++m){
                        c2[j][i] += a4[k][l][i][m] * b4[m][j][l][k];
                    }
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_2(C2,c2);
    //    fprintf(psi::outfile,"\n C(p,q,r,s) error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}


std::pair<std::string,double> test_Cij_Aikjl_Bkl()
{
    std::string test = "C2(\"ij\") += A4(\"ikjl\") * B2(\"kl\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    size_t nm = 8;
    std::vector<size_t> dimsA = {ni,nk,nj,nl};
    std::vector<size_t> dimsB = {nk,nl};
    std::vector<size_t> dimsC = {ni,nj};

    Tensor A4("A4",dimsA);
    Tensor B2("B4",dimsB);
    Tensor C2("C2",dimsC);

    initialize_random_4(A4,a4);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_2(B2,b2);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C2.zero();
    C2("ij") += A4("ikjl") * B2("kl");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            c2[i][j] = 0.0;
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c2[i][j] += a4[i][k][j][l] * b2[k][l];
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_2(C2,c2);
    //    fprintf(psi::outfile,"\n C(p,q,r,s) error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Cijkl_Aij_Bkl()
{
    std::string test = "C4(\"ijkl\") += A2(\"ij\") * B2(\"kl\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());
    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 9;
    std::vector<size_t> dimsA = {ni,nj};
    std::vector<size_t> dimsB = {nk,nl};
    std::vector<size_t> dimsC = {ni,nj,nk,nl};

    Tensor A2("A2",dimsA);
    Tensor B2("B2",dimsB);
    Tensor C4("C4",dimsC);

    initialize_random_2(A2,a2);
    //    double a_diff = difference_4(A4,a4);
    //    fprintf(psi::outfile,"\n A error: %e",a_diff);

    initialize_random_2(B2,b2);
    //    double b_diff = difference_4(B4,b4);
    //    fprintf(psi::outfile,"\n B error: %e",b_diff);

    C4.zero();
    C4("ijkl") += A2("ij") * B2("kl");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    c4[i][j][k][l] = a2[i][j] * b2[k][l];
                }
            }
        }
    }
    std::pair<double,double> C_diff = difference_4(C4,c4);
    //    fprintf(psi::outfile,"\n C(p,q,r,s) error: sum = %e max = %e",C_diff.first,C_diff.second);
    return std::make_pair(test,std::fabs(C_diff.second));
}

std::pair<std::string,double> test_Dijkl_Aijmn_Bkm_Cln()
{
    std::string test = "D4(\"ijkl\") += A4(\"ijmn\") * B2(\"km\") * C2(\"ln\")";
    fprintf(psi::outfile,"\n Testing %s",test.c_str());

    size_t ni = 5;
    size_t nj = 6;
    size_t nk = 7;
    size_t nl = 6;
    size_t nm = 7;
    size_t nn = 5;
    std::vector<size_t> dimsA = {ni,nj,nm,nn};
    std::vector<size_t> dimsB = {nk,nm};
    std::vector<size_t> dimsC = {nl,nn};
    std::vector<size_t> dimsD = {ni,nj,nk,nl};

    Tensor A4("A4",dimsA);
    Tensor B2("B2",dimsB);
    Tensor C2("C2",dimsC);
    Tensor D4("D4",dimsD);

    initialize_random_4(A4,a4);
    std::pair<double,double> a_diff = difference_4(A4,a4);
    fprintf(psi::outfile,"\n A4 error: sum = %e max = %e",a_diff.first,a_diff.second);

    initialize_random_2(B2,b2);
    std::pair<double,double> b_diff = difference_2(B2,b2);
    fprintf(psi::outfile,"\n B2 error: sum = %e max = %e",b_diff.first,b_diff.second);

    initialize_random_2(C2,c2);
    std::pair<double,double> C_diff = difference_2(C2,c2);
    fprintf(psi::outfile,"\n C2 error: sum = %e max = %e",C_diff.first,C_diff.second);

    D4.zero();
//    D4("ijkl") += A4("ijmn") * B2("km") * C2("ln");

    for (size_t i = 0; i < ni; ++i){
        for (size_t j = 0; j < nj; ++j){
            for (size_t k = 0; k < nk; ++k){
                for (size_t l = 0; l < nl; ++l){
                    d4[i][j][k][l]  = 0.0;
                    for (size_t m = 0; m < nm; ++m){
                        for (size_t n = 0; n < nn; ++n){
                            d4[i][j][k][l] += a4[i][j][m][n] * b2[k][m] * c2[l][n];
                        }
                    }
                }
            }
        }
    }
    std::pair<double,double> D_diff = difference_4(D4,d4);
    fprintf(psi::outfile,"\n D error: sum = %e max = %e",D_diff.first,D_diff.second);
    return std::make_pair(test,D_diff.second);
}

std::pair<double,double> difference_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO])
{
    double sum = 0.0;
    double max = 0.0;
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            sum += std::fabs(tensor(i,j) - matrix[i][j]);
            max = std::max(max,std::fabs(tensor(i,j) - matrix[i][j]));
        }
    }
    return std::make_pair(sum,max);
}


void initialize_random_2(Tensor& tensor,double matrix[MAXTWO][MAXTWO])
{
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            double randnum = double(std::rand())/double(RAND_MAX);
            tensor(i,j) = randnum;
            matrix[i][j] = randnum;
        }
    }
}

std::pair<double,double> difference_4(Tensor& tensor,double matrix[MAXFOUR][MAXFOUR][MAXFOUR][MAXFOUR])
{
    double sum = 0.0;
    double max = 0.0;
    size_t n0 = tensor.dims(0);
    size_t n1 = tensor.dims(1);
    size_t n2 = tensor.dims(2);
    size_t n3 = tensor.dims(3);
    for (size_t i = 0; i < n0; ++i){
        for (size_t j = 0; j < n1; ++j){
            for (size_t k = 0; k < n2; ++k){
                for (size_t l = 0; l < n3; ++l){
                    sum += std::fabs(tensor(i,j,k,l) - matrix[i][j][k][l]);
                    max = std::max(max,std::fabs(tensor(i,j,k,l) - matrix[i][j][k][l]));
                }
            }
        }
    }
    return std::make_pair(sum,max);
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
                    double randnum = double(std::rand())/double(RAND_MAX);
                    tensor(i,j,k,l) = randnum;
                    matrix[i][j][k][l] = randnum;
                }
            }
        }
    }
}
