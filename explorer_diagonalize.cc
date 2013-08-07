#include "explorer.h"

#include <cmath>

#include <boost/timer.hpp>
#include <boost/format.hpp>

#include <libqt/qt.h>


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <libciomr/libciomr.h>
//#include <libqt/qt.h>

#include "explorer.h"
#include "cartographer.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

inline double clamp(double x, double a, double b)

{
    return x < a ? a : (x > b ? b : x);
}

inline double smootherstep(double edge0, double edge1, float x)
{
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * x *( x *( x * 6. - 15.) + 10.);
}



#define BIGNUM 1E100
#define MAXIT 1000

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

int david2(double **A, int N, int M, double *eps, double **v,
           double cutoff, int print)
{
    int i, j, k, L, I;
    double minimum;
    int min_pos, numf, iter, *conv, converged, maxdim, skip_check;
    int *small2big, init_dim;
    int smart_guess = 1;
    double *Adiag, **b, **bnew, **sigma, **G;
    double *lambda, **alpha, **f, *lambda_old;
    double norm, denom, diff;

    maxdim = 8 * M;

    b = block_matrix(maxdim, N);  /* current set of guess vectors,
                   stored by row */
    bnew = block_matrix(M, N); /* guess vectors formed from old vectors,
                stored by row*/
    sigma = block_matrix(N, maxdim); /* sigma vectors, stored by column */
    G = block_matrix(maxdim, maxdim); /* Davidson mini-Hamitonian */
    f = block_matrix(maxdim, N); /* residual eigenvectors, stored by row */
    alpha = block_matrix(maxdim, maxdim); /* eigenvectors of G */
    lambda = init_array(maxdim); /* eigenvalues of G */
    lambda_old = init_array(maxdim); /* approximate roots from previous
                      iteration */

    if(smart_guess) { /* Use eigenvectors of a sub-matrix as initial guesses */

        if(N > 7*M) init_dim = 7*M;
        else init_dim = M;
        Adiag = init_array(N);
        small2big = init_int_array(7*M);
        for(i=0; i < N; i++) { Adiag[i] = A[i][i]; }
        for(i=0; i < init_dim; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for(j=1; j < N; j++)
                if(Adiag[j] < minimum) {
                    minimum = Adiag[j];
                    min_pos = j;
                    small2big[i] = j;
                }

            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        for(i=0; i < init_dim; i++) {
            for(j=0; j < init_dim; j++)
                G[i][j] = A[small2big[i]][small2big[j]];
        }

        sq_rsp(init_dim, init_dim, G, lambda, 1, alpha, 1e-12);

        for(i=0; i < init_dim; i++) {
            for(j=0; j < init_dim; j++)
                b[i][small2big[j]] = alpha[j][i];
        }

        free(Adiag);
        free(small2big);
    }
    else { /* Use unit vectors as initial guesses */
        Adiag = init_array(N);
        for(i=0; i < N; i++) { Adiag[i] = A[i][i]; }
        for(i=0; i < M; i++) {
            minimum = Adiag[0];
            min_pos = 0;
            for(j=1; j < N; j++)
                if(Adiag[j] < minimum) { minimum = Adiag[j]; min_pos = j; }

            b[i][min_pos] = 1.0;
            Adiag[min_pos] = BIGNUM;
            lambda_old[i] = minimum;
        }
        free(Adiag);
    }

    L = init_dim;
    iter =0;
    converged = 0;
    conv = init_int_array(M); /* boolean array for convergence of each
                   root */
    while(converged < M && iter < MAXIT) {

        /* orthonormalize them */
        for(i=0; i < M; i++){
            norm = 0.0;
            for(k=0; k < N; k++)
                norm += b[i][k] * b[i][k];
            norm = std::sqrt(norm);
            for(k=0; k < N; k++)
                b[i][k] /= norm;
        }

        skip_check = 0;
        if(print) printf("\niter = %d\n", iter);

        /* form mini-matrix */
        C_DGEMM('n','t', N, L, N, 1.0, &(A[0][0]), N, &(b[0][0]), N,
                0.0, &(sigma[0][0]), maxdim);
        C_DGEMM('n','n', L, L, N, 1.0, &(b[0][0]), N,
                &(sigma[0][0]), maxdim, 0.0, &(G[0][0]), maxdim);

        /* diagonalize mini-matrix */
        sq_rsp(L, L, G, lambda, 1, alpha, 1e-12);

        /* form preconditioned residue vectors */
        for(k=0; k < M; k++)
            for(I=0; I < N; I++) {
                f[k][I] = 0.0;
                for(i=0; i < L; i++) {
                    f[k][I] += alpha[i][k] * (sigma[I][i] - lambda[k] * b[i][I]);
                }
                denom = lambda[k] - A[I][I];
                if(fabs(denom) > 1e-3) f[k][I] /= denom;
                else f[k][I] = 0.0;
            }

        /* normalize each residual */
        for(k=0; k < M; k++) {
            norm = 0.0;
            for(I=0; I < N; I++) {
                norm += f[k][I] * f[k][I];
            }
            norm = sqrt(norm);
            for(I=0; I < N; I++) {
                if(norm > 1e-6) f[k][I] /= norm;
                else f[k][I] = 0.0;
            }
        }

        /* schmidt orthogonalize the f[k] against the set of b[i] and add
       new vectors */
        for(k=0,numf=0; k < M; k++)
            if(schmidt_add(b, L, N, f[k])) { L++; numf++; }

        /* If L is close to maxdim, collapse to one guess per root */
        if(maxdim - L < M) {
            if(print) {
                printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                printf("Collapsing eigenvectors.\n");
            }
            for(i=0; i < M; i++) {
                memset((void *) bnew[i], 0, N*sizeof(double));
                for(j=0; j < L; j++) {
                    for(k=0; k < N; k++) {
                        bnew[i][k] += alpha[j][i] * b[j][k];
                    }
                }
            }

            /* copy new vectors into place */
            for(i=0; i < M; i++)
                for(k=0; k < N; k++)
                    b[i][k] = bnew[i][k];

            skip_check = 1;

            L = M;
        }

        /* check convergence on all roots */
        if(!skip_check) {
            converged = 0;
            zero_int_array(conv, M);
            if(print) {
                printf("Root      Eigenvalue       Delta  Converged?\n");
                printf("---- -------------------- ------- ----------\n");
            }
            for(k=0; k < M; k++) {
                diff = fabs(lambda[k] - lambda_old[k]);
                if(diff < cutoff) {
                    conv[k] = 1;
                    converged++;
                }
                lambda_old[k] = lambda[k];
                if(print) {
                    printf("%3d  %20.14f %4.3e    %1s\n", k, lambda[k], diff,
                           conv[k] == 1 ? "Y" : "N");
                }
            }
        }

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    if(converged == M) {
        for(i=0; i < M; i++) {
            eps[i] = lambda[i];
            for(j=0; j < L; j++) {
                for(I=0; I < N; I++) {
                    v[I][i] += alpha[j][i] * b[j][I];
                }
            }
        }
        if(print) printf("Davidson algorithm converged in %d iterations.\n", iter);
    }

    free(conv);
    free_block(b);
    free_block(bnew);
    free_block(sigma);
    free_block(G);
    free_block(f);
    free_block(alpha);
    free(lambda);
    free(lambda_old);

    return converged;
}

/**
 * Find all the Slater determinants with an energy lower than determinant_threshold_
 */
void Explorer::diagonalize(psi::Options& options)
{
    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in a small space\n");

    boost::timer t_hbuild;
    SharedMatrix H = build_hamiltonian(options);
    fprintf(outfile,"\n  Time spent building H             = %f s",t_hbuild.elapsed());

    boost::timer t_hsmooth;
    smooth_hamiltonian(H);
    fprintf(outfile,"\n  Time spent smoothing H            = %f s",t_hsmooth.elapsed());

    int ndets = H->nrow();
    int nroots = ndets;

    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        nroots = std::min(1,ndets);
    }

    SharedMatrix evecs(new Matrix("U",ndets,nroots));
    SharedVector evals(new Vector("e",nroots));

    boost::timer t_hdiag;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H,evals,evecs,nroots);
    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
        fprintf(outfile,"\n  Performing full diagonalization.");
        H->diagonalize(evecs,evals);
    }
    fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());

    int ndets_print = std::min(nroots,10);
    for (int i = 0; i < ndets_print; ++ i){
        fprintf(outfile,"\n Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i),27.211 * (evals->get(i) - evals->get(0)));
    }






    //    double unscreened_range = determinant_threshold - h_buffer;

    //    if (std::fabs(h_buffer) > 0.0){
    //        for (int I = 1; I < ndets; ++I){
    //            double deltaE = H->get(I,I) - H->get(0,0);
    //            if (deltaE > unscreened_range){
    //                double excessE = deltaE - unscreened_range;
    //                double factor = smootherstep(excessE,h_buffer,h_buffer);
    //                fprintf(outfile,"\n  Det %d , De = %f, exE = %f, f = %f",I,deltaE,excessE,factor);

    //                //double factor = 1.0 - excessE / h_buffer;
    //                for (int J = 0; J <= I; ++J){
    //                    double element = H->get(I,J);
    //                    H->set(I,J,element * factor);
    //                    H->set(J,I,element * factor);
    //                }
    //            }
    //        }
    //    }

    //    for (int n = 1; n <= std::min(ndets,4000); n += 1){
    //        timer t_hbuild;
    //        for (int I = 0; I < n; ++I){
    //            boost::tuple<double,int,int>& determinantI = determinants[I];
    //            int Isa = std::get<1>(determinantI);
    //            int Isb = std::get<2>(determinantI);
    //            detI.set_bits(vec_astr[Isa].second,vec_bstr[Isb].second);
    //            for (int J = I; J < n; ++J){
    //                boost::tuple<double,int,int>& determinantJ = determinants[J];
    //                int Jsa = std::get<1>(determinantJ);
    //                int Jsb = std::get<2>(determinantJ);
    //                detJ.set_bits(vec_astr[Jsa].second,vec_bstr[Jsb].second);
    //                double HIJ = SlaterRules(detI,detJ,ints_);
    //                H->set(I,J,HIJ);
    //                H->set(J,I,HIJ);
    //            }
    //        }
    //        fprintf(outfile,"\n  Time spent building H             = %f s",t_hbuild.elapsed());

    //        timer t_hdiag;

    ////        int nroots = 5;
    ////        SharedMatrix evecs(new Matrix("U",maxdet,nroots));
    ////        SharedVector evals(new Vector("e",maxdet));
    ////        lanczos(H,evecs,evals,nroots);

    //        SharedMatrix evecs(new Matrix("U",maxdet,maxdet));
    //        SharedVector evals(new Vector("e",maxdet));
    //        H->diagonalize(evecs,evals);

    //        fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
    //        fprintf(outfile,"\n %6d",n);
    //        for (int i = 0; i < 5; ++ i){
    //            fprintf(outfile," %3d %15.9f %8.4f",i,evals->get(i)+ nuclear_energy,27.211*(evals->get(i) - evals->get(0)));
    //        }
    //    }
}

/**
 * Build the Hamiltonian matrix from the list of determinants.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix Explorer::build_hamiltonian(Options& options)
{
    int ntot_dets = static_cast<int>(determinants_.size());

    // the number of determinants used to form the Hamiltonian matrix
    int ndets = 0;

    // Determine the size of the Hamiltonian matrix
    if (options.get_str("H_TYPE") == "FIXED_SIZE"){
        ndets = std::min(options.get_int("NDETS"),ntot_dets);
        fprintf(outfile,"\n  Building the Hamiltonian using the first %d determinants\n",ndets);
        fprintf(outfile,"\n  The energy range spanned is [%f,%f]\n",determinants_[0].get<0>(),determinants_[ndets-1].get<0>());
    }else if (options.get_str("H_TYPE") == "FIXED_ENERGY"){
        fprintf(outfile,"\n\n  Building the Hamiltonian using determinants with excitation energy less than %f Eh",determinant_threshold_);
        int max_ndets_fixed_energy = 10000;
        ndets = std::min(max_ndets_fixed_energy,ntot_dets);
        if (ndets == max_ndets_fixed_energy){
            fprintf(outfile,"\n\n  WARNING: the number of determinants used to build the Hamiltonian\n"
                    "  exceeds the maximum number allowed (%d).  Reducing the size of H.\n\n",max_ndets_fixed_energy);
        }
    }

    SharedMatrix H(new Matrix("Hamiltonian Matrix",ndets,ndets));

    // Form the Hamiltonian matrix
    StringDeterminant detI(reference_determinant_);
    StringDeterminant detJ(reference_determinant_);

    for (int I = 0; I < ndets; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        detI.set_bits(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>());
        for (int J = I + 1; J < ndets; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[J];
            int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantI);
            int Jsa = determinantJ.get<2>();        //std::get<1>(determinantI);
            int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantI);
            int Jsb = determinantJ.get<4>();        //std::get<2>(determinantI);
            detJ.set_bits(vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>());
            double HIJ = detI.slater_rules(detJ);
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
        H->set(I,I,determinantI.get<0>());
    }

    return H;
}

void Explorer::smooth_hamiltonian(SharedMatrix H)
{
    int ndets = H->nrow();
    double main_space_threshold = determinant_threshold_ - smoothing_threshold_;
    // Partition the Hamiltonian into main and intermediate model space
    int ndets_main = 0;
    for (int I = 0; I < ndets; ++I){
        if (H->get(I,I) - H->get(0,0) > main_space_threshold){
            ndets_main = I;
            break;
        }
    }
    fprintf(outfile,"\n\n  The model space of dimension %d will be split into %d (main) + %d (intermediate) states",ndets,ndets_main,ndets - ndets_main);
    for (int I = 0; I < ndets; ++I){
        for (int J = ndets_main; J < ndets; ++J){
            if (I != J){
                double HIJ = H->get(I,J);
                double EI = H->get(I,I);
                double EJ = H->get(J,J);
                double EJ0 = EJ - H->get(0,0);
                double factor = 1.0 - smootherstep(0.0,smoothing_threshold_,std::fabs(EJ0 - main_space_threshold));
                H->set(I,J,factor * HIJ);
                H->set(J,I,factor * HIJ);
            }
        }
    }
}

void Explorer::davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots)
{
    david2(H->pointer(),H->nrow(),nroots,Eigenvalues->pointer(),Eigenvectors->pointer(),1.0e-10,0);

    //    int n = H->nrow();
    //    int n_small = std::min(50,n);

    //    // Diagonalize a small matrix of dimension 50 x 50 or less
    //    SharedMatrix Hsmall(new Matrix("U",n_small,n_small));
    //    SharedMatrix evecs_small(new Matrix("U",n_small,n_small));
    //    SharedVector evals_small(new Vector("e",n_small));
    //    for (int I = 0; I < n_small; ++I){
    //        for (int J = 0; J < n_small; ++J){
    //            Hsmall->set(I,J,H->get(I,J));
    //        }
    //    }
    //    Hsmall->diagonalize(evecs_small,evals_small);

    //    /* 1. Select a set of L orthonormal guess vectors, at least one for each
    //     * root desired, and place in the set {b_i}.
    //     */
    //    int L = nroots;
    //    SharedMatrix b(new Matrix("b",n,L));
    //    for (int i = 0; i < L; ++i){
    //        for (int I = 0; I < n_small; ++I){
    //            b->set(I,i,evecs_small->get(I,i));
    //        }
    //    }

    //    /* 2. 2. Use a standard diagonalization method to solve the L × L eigenvalue
    //     * problem G alpha^k = ρ^k  alpha^k, k = 1,2,...,M
    //     * where Gij = (b_i,H b_j) = (b_i,sigma_j), 1 ≤ i,j ≤ L (26)
    //     * and M is the number of roots of interest.
    //     */
    //    SharedMatrix Hb(new Matrix("Hb",n,L));
    //    SharedMatrix R(new Matrix("R",n,L));
    //    SharedMatrix G(new Matrix("G",L,L));
    //    SharedMatrix alpha(new Matrix("G Eigenvectors",L,L));
    //    SharedVector rho(new Vector("G Eigenvalues",L));
    //    SharedVector Rho(new Matrix("G Eigenvalues",L,L));
    //    Hb->gemm(false,false,1.0,H,b,0.0);
    //    G->gemm(true,false,1.0,b,Hb,0.0);
    //    G->diagonalize(alpha,rho);
    //    Rho->set_diagonal(rho);
    //    R->gemm(false,false,1.0,Hb,alpha,0.0);
    //    R->gemm(false,false,-1.0,b,alpha,1.0);

}

}} // EndNamespaces




