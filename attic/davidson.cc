/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <cmath>
#include <fstream>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libqt/qt.h"
#include "psi4/physconst.h"
#include "psi4/psi4-dec.h"

#define BIGNUM 1E100
#define MAXIT 500

/*!
** david(): Computes the lowest few eigenvalues and eigenvectors of a
** symmetric matrix, A, using the Davidson-Liu algorithm.
**
** The matrix must be small enough to fit entirely in core.  This algorithm
** is useful if one is interested in only a few roots of the matrix
** rather than the whole spectrum.
**
** NB: This implementation will keep up to eight guess vectors for each
** root desired before collapsing to one vector per root.  In
** addition, if smart_guess=1 (the default), guess vectors are
** constructed by diagonalization of a sub-matrix of A; otherwise,
** unit vectors are used.
**
** TDC, July-August 2002
**
** \param A      = matrix to diagonalize
** \param N      = dimension of A
** \param M      = number of roots desired
** \param eps    = eigenvalues
** \param v      = eigenvectors
** \param cutoff = tolerance for convergence of eigenvalues
** \param print  = Boolean for printing additional information
**
** Returns: number of converged roots
** \ingroup QT
*/

using namespace psi;

int david2(double** A, int N, int M, double* eps, double** v, double cutoff, int print);

void test_davidson() {
    // BEGIN DEBUGGING
    // Write the Hamiltonian to disk
    outfile->Printf("\n\n  READING FILE FROM DISK...");
    std::ifstream ifs("ham.dat", std::ios::binary | std::ios::in);
    int ndets;
    ifs.read(reinterpret_cast<char*>(&ndets), sizeof(int));
    Matrix H(ndets, ndets);
    double** H_mat = H.pointer();
    ifs.read(reinterpret_cast<char*>(&(H_mat[0][0])), ndets * ndets * sizeof(double));
    ifs.close();
    outfile->Printf(" DONE.");

    int nroots = 4;

    psi::SharedMatrix evecs(new psi::Matrix("U", ndets, nroots));
    psi::SharedVector evals(new Vector("e", nroots));

    david2(H.pointer(), H.nrow(), nroots, evals->pointer(), evecs->pointer(), 1.0e-10, 1);

    int nroots_print = std::min(nroots, 25);
    for (int i = 0; i < nroots_print; ++i) {
        outfile->Printf("\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV", i + 1,
                        evals->get(i), pc_hartree2ev * (evals->get(i) - evals->get(0)));
    }
    // END DEBUGGING
}

int david2(double** A, int N, int M, double* eps, double** v, double cutoff, int print) {
    int i, j, k, L, I;
    double minimum;
    int min_pos, numf, iter, converged, maxdim, skip_check;
    int init_dim = 0;
    int smart_guess = 1;
    double **b, **bnew, **sigma, **G;
    //    double *lambda, **alpha, **f, *lambda_old;
    double **alpha, **f;
    double norm, denom, diff;

    maxdim = 8 * M;

    b = block_matrix(maxdim, N);          /* current set of guess vectors, stored by row */
    bnew = block_matrix(M, N);            /* guess vectors formed from old vectors, stored by row*/
    sigma = block_matrix(N, maxdim);      /* sigma vectors, stored by column */
    G = block_matrix(maxdim, maxdim);     /* Davidson mini-Hamitonian */
    f = block_matrix(maxdim, N);          /* residual eigenvectors, stored by row */
    alpha = block_matrix(maxdim, maxdim); /* eigenvectors of G */
    std::vector<double> lambda(maxdim);
    std::vector<double> lambda_old(maxdim);
    //    lambda = init_array(maxdim);          /* eigenvalues of G */
    //    lambda_old = init_array(maxdim);      /* approximate roots from previous iteration */
    std::vector<double> Adiag(N, 0.0);
    if (smart_guess) { /* Use eigenvectors of a sub-matrix as initial guesses */

        if (N > 7 * M)
            init_dim = 7 * M;
        else
            init_dim = M;
        //        Adiag = init_array(N);
        std::vector<int> small2big(7 * M);
        //        small2big = init_int_array(7 * M);
        for (i = 0; i < N; i++) {
            Adiag[i] = A[i][i];
        }
        for (i = 0; i < init_dim; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for (j = 1; j < N; j++)
                if (Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                    small2big[i] = j;
                }

            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        for (i = 0; i < init_dim; i++) {
            for (j = 0; j < init_dim; j++)
                G[i][j] = A[small2big[i]][small2big[j]];
        }

        sq_rsp(init_dim, init_dim, G, lambda.data(), 1, alpha, 1e-12);

        for (i = 0; i < init_dim; i++) {
            for (j = 0; j < init_dim; j++)
                b[i][small2big[j]] = alpha[j][i];
        }

        //        free(Adiag);
        //        free(small2big);
    } else { /* Use unit vectors as initial guesses */
             //        Adiag = init_array(N);
        for (i = 0; i < N; i++) {
            Adiag[i] = A[i][i];
        }
        for (i = 0; i < M; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for (j = 1; j < N; j++)
                if (Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                }

            b[i][min_pos] = 1.0;
            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        //        free(Adiag);
    }

    // L = init_dim;
    iter = 0;
    converged = 0;
    std::vector<int> conv(M, 0);
    // conv = init_int_array(M); /* boolean array for convergence of each root */
    while (converged < M && iter < MAXIT) {

        skip_check = 0;
        if (print)
            printf("\niter = %d\n", iter);

        /* form mini-matrix */
        C_DGEMM('n', 't', N, L, N, 1.0, &(A[0][0]), N, &(b[0][0]), N, 0.0, &(sigma[0][0]), maxdim);
        C_DGEMM('n', 'n', L, L, N, 1.0, &(b[0][0]), N, &(sigma[0][0]), maxdim, 0.0, &(G[0][0]),
                maxdim);

        /* diagonalize mini-matrix */
        sq_rsp(L, L, G, lambda.data(), 1, alpha, 1e-12);

        /* form preconditioned residue vectors */
        for (k = 0; k < M; k++)
            for (I = 0; I < N; I++) {
                f[k][I] = 0.0;
                for (i = 0; i < L; i++) {
                    f[k][I] += alpha[i][k] * (sigma[I][i] - lambda[k] * b[i][I]);
                }
                denom = lambda[k] - A[I][I];
                if (std::fabs(denom) > 1e-6)
                    f[k][I] /= denom;
                else
                    f[k][I] = 0.0;
            }

        /* normalize each residual */
        for (k = 0; k < M; k++) {
            norm = 0.0;
            for (I = 0; I < N; I++) {
                norm += f[k][I] * f[k][I];
            }
            norm = std::sqrt(norm);
            for (I = 0; I < N; I++) {
                if (norm > 1e-6)
                    f[k][I] /= norm;
                else
                    f[k][I] = 0.0;
            }
        }

        /* schmidt orthogonalize the f[k] against the set of b[i] and add
       new vectors */
        for (k = 0, numf = 0; k < M; k++)
            if (schmidt_add(b, L, N, f[k])) {
                L++;
                numf++;
            }

        /* If L is close to maxdim, collapse to one guess per root */
        if (maxdim - L < M) {
            if (print) {
                printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                printf("Collapsing eigenvectors.\n");
            }
            for (i = 0; i < M; i++) {
                memset((void*)bnew[i], 0, N * sizeof(double));
                for (j = 0; j < L; j++) {
                    for (k = 0; k < N; k++) {
                        bnew[i][k] += alpha[j][i] * b[j][k];
                    }
                }
            }
            /* orthonormalize the new vectors */
            /* copy new vectors into place */
            for (i = 0; i < M; i++) {
                norm = 0.0;
                // Project out the orthonormal vectors
                for (j = 0; j < i; ++j) {
                    double proj = 0.0;
                    for (k = 0; k < N; k++) {
                        proj += b[j][k] * bnew[i][k];
                    }
                    for (k = 0; k < N; k++) {
                        bnew[i][k] -= proj * b[j][k];
                    }
                }
                for (k = 0; k < N; k++) {
                    norm += bnew[i][k] * bnew[i][k];
                }
                norm = std::sqrt(norm);
                for (k = 0; k < N; k++) {
                    b[i][k] = bnew[i][k] / norm;
                }
            }

            skip_check = 1;

            L = M;
        }

        /* check convergence on all roots */
        if (!skip_check) {
            converged = 0;
            for (int& i : conv) {
                i = 0;
            }
            //            zero_int_array(conv, M);
            if (print) {
                printf("Root      Eigenvalue       Delta  Converged?\n");
                printf("---- -------------------- ------- ----------\n");
            }
            for (k = 0; k < M; k++) {
                diff = std::fabs(lambda[k] - lambda_old[k]);
                if (diff < cutoff) {
                    conv[k] = 1;
                    converged++;
                }
                lambda_old[k] = lambda[k];

                norm = 0.0;
                if (print) {
                    printf("%3d  %20.14f %4.3e    %1s\n", k, lambda[k], diff,
                           conv[k] == 1 ? "Y" : "N");
                }
            }
        }

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    // if(converged == M) {
    for (i = 0; i < M; i++) {
        eps[i] = lambda[i];
        for (j = 0; j < L; j++) {
            for (I = 0; I < N; I++) {
                v[I][i] += alpha[j][i] * b[j][I];
            }
        }
        // Normalize v
        norm = 0.0;
        for (I = 0; I < N; I++) {
            norm += v[I][i] * v[I][i];
        }
        norm = std::sqrt(norm);
        for (I = 0; I < N; I++) {
            v[I][i] /= norm;
        }
    }
    if (print)
        printf("Davidson algorithm converged in %d iterations.\n", iter);
    //    }

    free_block(b);
    free_block(bnew);
    free_block(sigma);
    free_block(G);
    free_block(f);
    free_block(alpha);
    return converged;
}
