#include <fstream>
#include <libpsio/psio.hpp>
#include"tensor_init.h"

namespace psi{ namespace main{

TENSOR_INIT::TENSOR_INIT()
{}

TENSOR_INIT::~TENSOR_INIT()
{}

void TENSOR_INIT::init_matrix4d(double****& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4){
    if ((size1 == 0) || (size2 == 0) || (size3 == 0) || (size4 == 0)){
        fprintf(outfile,"\n  NULL Matrix Initialization!");
        matrix = NULL;
        exit (1);
    }
    else{
        matrix = new double***[size1];
        for ( int i = 0; i < size1; i++){
            matrix[i] = new double**[size2];
            for (int j = 0; j < size2; j++){
                matrix[i][j] = new double*[size3];
                for ( int k = 0; k < size3; k++){
                    matrix[i][j][k] = new double[size4];
                    for ( int l = 0; l < size4; l++){
                        matrix[i][j][k][l] = 0.0;
                    }
                }
            }
        }
    }
}

void TENSOR_INIT::free_matrix4d(double****& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4){
    if ((size1 == 0) || (size2 == 0) || (size3 == 0) || (size4 == 0)){
        fprintf(outfile,"\n  NULL Matrix Delete!");
        exit (1);
    }
    if (matrix != NULL){
        for ( int i = 0; i < size1; i++){
            for ( int j = 0; j < size2; j++){
                for ( int k = 0; k < size3; k++){
                    delete[] matrix[i][j][k];
                }
                delete[] matrix[i][j];
            }
            delete[] matrix[i];
        }
        delete[] matrix;
    }
}

void TENSOR_INIT::init_matrix6d(double******& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4, const size_t &size5, const size_t &size6){
    if ((size1 == 0) || (size2 == 0) || (size3 == 0) || (size4 == 0) || (size5 == 0) || (size6 == 0)){
        fprintf(outfile,"\n  NULL Matrix Initialization!");
        matrix = NULL;
        exit (1);
    }
    else{
        matrix = new double*****[size1];
        for ( int i = 0; i < size1; i++){
            matrix[i] = new double****[size2];
            for (int j = 0; j < size2; j++){
                matrix[i][j] = new double***[size3];
                for ( int k = 0; k < size3; k++){
                    matrix[i][j][k] = new double**[size4];
                    for ( int l = 0; l < size4; l++){
                        matrix[i][j][k][l] = new double*[size5];
                        for (int m = 0; m < size5; m++){
                            matrix[i][j][k][l][m] = new double[size6];
                            for (int n = 0; n < size6; n++){
                                matrix[i][j][k][l][m][n] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

void TENSOR_INIT::free_matrix6d(double******& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4, const size_t &size5, const size_t &size6){
    if ((size1 == 0) || (size2 == 0) || (size3 == 0) || (size4 == 0) || (size5 == 0) || (size6 == 0)){
        fprintf(outfile,"\n  NULL Matrix Delete!");
        exit (1);
    }
    if (matrix != NULL){
        for ( int i = 0; i < size1; i++){
            for ( int j = 0; j < size2; j++){
                for ( int k = 0; k < size3; k++){
                    for ( int l = 0; l < size4; l++){
                        for ( int m = 0; m < size5; m++){
                            delete[] matrix[i][j][k][l][m];
                        }
                        delete[] matrix[i][j][k][l];
                    }
                    delete[] matrix[i][j][k];
                }
                delete[] matrix[i][j];
            }
            delete[] matrix[i];
        }
        delete[] matrix;
    }
}


}}
