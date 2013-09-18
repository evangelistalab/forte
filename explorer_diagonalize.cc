#include "explorer.h"

#include <cmath>
#include <functional>
#include <algorithm>

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

inline double smootherstep(double edge0, double edge1, double x)
{
    // Scale, and clamp x to 0..1 range
    x = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    // Evaluate polynomial
    return x * x * x *( x *( x * 6. - 15.) + 10.);
}



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

    maxdim = 20 * M;

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
                if(fabs(denom) > 1e-6) f[k][I] /= denom;
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
    //if(converged == M) {
        for(i=0; i < M; i++) {
            eps[i] = lambda[i];
            for(j=0; j < L; j++) {
                for(I=0; I < N; I++) {
                    v[I][i] += alpha[j][i] * b[j][I];
                }
            }
        }
        if(print) printf("Davidson algorithm converged in %d iterations.\n", iter);
//    }

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
 * Diagonalize the
 */
void Explorer::diagonalize_p_space(psi::Options& options)
{
    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in the model + intermediate space\n");

    // 1) Build the Hamiltonian
    boost::timer t_hbuild;
    SharedMatrix H = build_hamiltonian_parallel(options);
    fprintf(outfile,"\n  Time spent building H             = %f s",t_hbuild.elapsed());
    fflush(outfile);

    // 2) Smooth out the couplings of the model and intermediate space
    boost::timer t_hsmooth;
    smooth_hamiltonian(H);
    fprintf(outfile,"\n  Time spent smoothing H            = %f s",t_hsmooth.elapsed());
    fflush(outfile);

    // 3) Setup stuff necessary to diagonalize the Hamiltonian
    int ndets = H->nrow();
    int nroots = ndets;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        nroots = std::min(options.get_int("NROOT"),ndets);
    }
    SharedMatrix evecs(new Matrix("U",ndets,nroots));
    SharedVector evals(new Vector("e",nroots));

    // 4) Diagonalize the Hamiltonian
    boost::timer t_hdiag;
    if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
        fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
        davidson_liu(H,evals,evecs,nroots);
    }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
        fprintf(outfile,"\n  Performing full diagonalization.");
        H->diagonalize(evecs,evals);
    }
    fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
    fflush(outfile);

    // 5) Print the energy
    int nroots_print = std::min(nroots,25);
    for (int i = 0; i < nroots_print; ++ i){
        fprintf(outfile,"\n  Adaptive CI Energy Root %3d = %.12f Eh = %8.4f eV",i + 1,evals->get(i),27.211 * (evals->get(i) - evals->get(0)));
    }

    // 6) Print the major contributions to the eigenvector
    double significant_threshold = 0.001;
    double significant_wave_function = 0.95;
    for (int i = 0; i < nroots_print; ++ i){
        fprintf(outfile,"\n  The most important determinants (%.0f%% of the wave functions) for root %d:",100.0 * significant_wave_function,i + 1);
        // Identify all contributions with |C_J| > significant_threshold
        double** C_mat = evecs->pointer();
        std::vector<std::pair<double,int> > C_J_sorted;
        for (int J = 0; J < ndets; ++J){
            if (std::fabs(C_mat[J][i]) > significant_threshold){
                C_J_sorted.push_back(make_pair(std::fabs(C_mat[J][i]),J));
            }
        }
        // Sort them and print
        std::sort(C_J_sorted.begin(),C_J_sorted.end(),std::greater<std::pair<double,int> >());
        double cum_wfn = 0.0;
        for (size_t I = 0, max_I = C_J_sorted.size(); I < max_I; ++I){
            int J = C_J_sorted[I].second;
            fprintf(outfile,"\n %3ld   %+9.6f   %9.6f   %.6f   %d",I,C_mat[J][i],C_mat[J][i] * C_mat[J][i],H->get(J,J),J);
            cum_wfn += C_mat[J][i] * C_mat[J][i];
            if (cum_wfn > significant_wave_function) break;
        }
    }
    fflush(outfile);
}

/**
 * Diagonalize the
 */
void Explorer::diagonalize_p_space_lowdin(psi::Options& options)
{
    fprintf(outfile,"\n\n  Diagonalizing the Hamiltonian in the P space with Lowdin's contributions from the Q space\n");
    int root = 0;
    double E = 1.0e100;
    double delta_E = 1.0e10;
    for (int cycle = 0; cycle < 20; ++cycle){
        // 1) Build the Hamiltonian
        boost::timer t_hbuild;
        SharedMatrix H = build_hamiltonian_parallel(options);
        fprintf(outfile,"\n  Time spent building H             = %f s",t_hbuild.elapsed());
        fflush(outfile);

        // 2) Add the Lowding contribution to H
        boost::timer t_hbuild_lowdin;
        lowdin_hamiltonian(H,E);
        fprintf(outfile,"\n  Time spent on Lowding corrections = %f s",t_hbuild_lowdin.elapsed());
        fflush(outfile);

        // 3) Smooth out the couplings of the model and intermediate space
        boost::timer t_hsmooth;
        smooth_hamiltonian(H);
        fprintf(outfile,"\n  Time spent smoothing H            = %f s",t_hsmooth.elapsed());
        fflush(outfile);

        // 4) Setup stuff necessary to diagonalize the Hamiltonian
        int ndets = H->nrow();
        int nroots = ndets;

        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
            nroots = std::min(options.get_int("NROOT"),ndets);
        }

        SharedMatrix evecs(new Matrix("U",ndets,nroots));
        SharedVector evals(new Vector("e",nroots));

        // 5) Diagonalize the Hamiltonian
        boost::timer t_hdiag;
        if (options.get_str("DIAG_ALGORITHM") == "DAVIDSON"){
            fprintf(outfile,"\n  Using the Davidson-Liu algorithm.");
            davidson_liu(H,evals,evecs,nroots);
        }else if (options.get_str("DIAG_ALGORITHM") == "FULL"){
            fprintf(outfile,"\n  Performing full diagonalization.");
            H->diagonalize(evecs,evals);
        }
        fprintf(outfile,"\n  Time spent diagonalizing H        = %f s",t_hdiag.elapsed());
        fflush(outfile);

        // 5) Print the energy
        delta_E = evals->get(root) - E;
        E = evals->get(root);
        fprintf(outfile,"\n  Cycle %3d  E= %.12f  DE = %.12f",cycle,evals->get(root),std::fabs(delta_E) > 10.0 ? 0 : delta_E);
        fflush(outfile);

        if (std::fabs(delta_E) < options.get_double("E_CONVERGENCE")){
            fprintf(outfile,"\n\n  Adaptive CI Energy Root %3d = %.12f Eh",root + 1,evals->get(root));
            fprintf(outfile,"\n  Lowdin iterations converged!\n");
            break;
        }
    }
    //evaluate_perturbative_corrections(evals,evecs);
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
        int max_ndets_fixed_energy = options.get_int("MAX_NDETS");
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

/**
 * Build the Hamiltonian matrix from the list of determinants.
 * It assumes that the determinants are stored in increasing energetic order.
 * @param ndets
 * @return a SharedMatrix object that contains the Hamiltonian
 */
SharedMatrix Explorer::build_hamiltonian_parallel(Options& options)
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
        double E0 = determinants_[0].get<0>();
        for (int I = 0; I < ntot_dets; ++I){
            double EI = determinants_[I].get<0>();
            if (EI - E0 > space_i_threshold_){
                break;
            }
            ndets++;
        }
        fprintf(outfile,"\n\n  Building the Hamiltonian using determinants with excitation energy less than %f Eh",space_i_threshold_);
        fprintf(outfile,"\n  This requires a total of %d determinants",ndets);
        int max_ndets_fixed_energy = options.get_int("MAX_NDETS");
        if (ndets > max_ndets_fixed_energy){
            fprintf(outfile,"\n\n  WARNING: the number of determinants required to build the Hamiltonian (%d)\n"
                    "  exceeds the maximum number allowed (%d).  Reducing the size of H.\n\n",ndets,max_ndets_fixed_energy);
            ndets = max_ndets_fixed_energy;
        }
    }

    SharedMatrix H(new Matrix("Hamiltonian Matrix",ndets,ndets));

    // Form the Hamiltonian matrix
    #pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        for (int J = I + 1; J < ndets; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[J];
            const int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();        //std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();        //std::get<2>(determinantI);
            const double HIJ = StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>(),vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>());
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

    // Partition the Hamiltonian into main and intermediate model space
    int ndets_model = 0;
    for (int I = 0; I < ndets; ++I){
        if (H->get(I,I) - H->get(0,0) > space_m_threshold_){
            break;
        }
        ndets_model++;
    }
    H->print();
    fprintf(outfile,"\n\n  The model space of dimension %d will be split into %d (main) + %d (intermediate) states",ndets,ndets_model,ndets - ndets_model);
    for (int I = 0; I < ndets; ++I){
        for (int J = 0; J < ndets; ++J){
            if (I != J){
                double HIJ = H->get(I,J);
                double EI = H->get(I,I);
                double EJ = H->get(J,J);
                double EI0 = EI - H->get(0,0);
                double EJ0 = EJ - H->get(0,0);
                double factorI = 1.0 - smootherstep(space_m_threshold_,space_i_threshold_,EI0);
                double factorJ = 1.0 - smootherstep(space_m_threshold_,space_i_threshold_,EJ0);
                H->set(I,J,factorI * factorJ * HIJ);
            }
        }
    }
    H->print();
}

void Explorer::evaluate_perturbative_corrections(SharedVector evals,SharedMatrix evecs)
{
    int root = 0;
    double E_0 = evals->get(root);

    int ntot_dets = static_cast<int>(determinants_.size());
    int ndets_p = evecs->nrow();

    fprintf(outfile,"\n\n  Computing a second-order PT correction from the external (%d) to the model space (%d)",ntot_dets - ndets_p,ndets_p);

    // Model space - external space 2nd order correction
    double E_2_PQ = 0.0;
    #pragma omp parallel for schedule(dynamic)
    for (int A = ndets_p; A < ntot_dets; ++A){
        boost::tuple<double,int,int,int,int>& determinantA = determinants_[A];
        const double EA = determinantA.get<0>();
        const int A_class_a = determinantA.get<1>();  //std::get<1>(determinantI);
        const int Asa = determinantA.get<2>();        //std::get<1>(determinantI);
        const int A_class_b = determinantA.get<3>(); //std::get<2>(determinantI);
        const int Asb = determinantA.get<4>();        //std::get<2>(determinantI);
        double coupling = 0.0;
        for (int I = 0; I < ndets_p; ++I){
            boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
            const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
            const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
            const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
            const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
            const double HIA = StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>(),vec_astr_symm_[A_class_a][Asa].get<2>(),vec_bstr_symm_[A_class_b][Asb].get<2>());
            coupling += evecs->get(I,root) * HIA;
        }
        E_2_PQ -= coupling * coupling / (EA - E_0);
    }
    fprintf(outfile,"\n\n Adaptive CI + PT2 Energy Root %3d = %.12f Eh",root + 1,E_0 + E_2_PQ);
}

void Explorer::lowdin_hamiltonian(SharedMatrix H,double E)
{
    int ntot_dets = static_cast<int>(determinants_.size());
    int ndets_p = H->nrow();

    fprintf(outfile,"\n\n  Computing a second-order PT correction from the external (%d) to the model space (%d)",ntot_dets - ndets_p,ndets_p);

    #pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < ndets_p; ++I){
        boost::tuple<double,int,int,int,int>& determinantI = determinants_[I];
        const int I_class_a = determinantI.get<1>();  //std::get<1>(determinantI);
        const int Isa = determinantI.get<2>();        //std::get<1>(determinantI);
        const int I_class_b = determinantI.get<3>(); //std::get<2>(determinantI);
        const int Isb = determinantI.get<4>();        //std::get<2>(determinantI);
        for (int J = I; J < ndets_p; ++J){
            boost::tuple<double,int,int,int,int>& determinantJ = determinants_[J];
            const int J_class_a = determinantJ.get<1>();  //std::get<1>(determinantI);
            const int Jsa = determinantJ.get<2>();        //std::get<1>(determinantI);
            const int J_class_b = determinantJ.get<3>(); //std::get<2>(determinantI);
            const int Jsb = determinantJ.get<4>();        //std::get<2>(determinantI);
            double coupling = 0.0;
            for (int A = ndets_p; A < ntot_dets; ++A){
                boost::tuple<double,int,int,int,int>& determinantA = determinants_[A];
                const double EA = determinantA.get<0>();
                const int A_class_a = determinantA.get<1>();  //std::get<1>(determinantI);
                const int Asa = determinantA.get<2>();        //std::get<1>(determinantI);
                const int A_class_b = determinantA.get<3>(); //std::get<2>(determinantI);
                const int Asb = determinantA.get<4>();        //std::get<2>(determinantI);
                const double HIA = StringDeterminant::SlaterRules(vec_astr_symm_[I_class_a][Isa].get<2>(),vec_bstr_symm_[I_class_b][Isb].get<2>(),vec_astr_symm_[A_class_a][Asa].get<2>(),vec_bstr_symm_[A_class_b][Asb].get<2>());
                const double HJA = StringDeterminant::SlaterRules(vec_astr_symm_[J_class_a][Jsa].get<2>(),vec_bstr_symm_[J_class_b][Jsb].get<2>(),vec_astr_symm_[A_class_a][Asa].get<2>(),vec_bstr_symm_[A_class_b][Asb].get<2>());
                coupling += HIA * HJA / (E - EA);
            }
            double HIJ = H->get(I,J);
            H->set(I,J,HIJ + coupling);
            H->set(J,I,HIJ + coupling);
        }
    }
}

void Explorer::davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots)
{
    david2(H->pointer(),H->nrow(),nroots,Eigenvalues->pointer(),Eigenvectors->pointer(),1.0e-10,0);

//    int n = H->nrow();
//    int n_small = std::min(50,n);
//    int max_vecs = 12;

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

//    int K = max_vecs * nroots;
//    SharedMatrix b(new Matrix("b",n,K));
//    SharedMatrix Hb(new Matrix("Hb",n,L));

//    /* 1. Select a set of L orthonormal guess vectors, at least one for each
//         * root desired, and place in the set {b_i}.
//         */
//    std::vector<SharedVector> Hb;
//    std::vector<SharedVector> b;
//    int L = nroots;
//    for (int i = 0; i < L; ++i){
//        SharedVector b_i(new Vector(n));
//        for (int I = 0; I < n_small; ++I){
//            b_i->set(I,i,evecs_small->get(I,i));
//        }
//        b.push_back(b_i);
//    }

//    SharedVector Hb_i(new Vector("Hb_i",n));
//    SharedMatrix G(new Matrix("G",K,K));
//    SharedMatrix alpha(new Matrix("G Eigenvectors",K,K));
//    SharedVector rho(new Vector("G Eigenvalues",K));


//    int nvecs = 0;
//    for (int iter = 0; iter < max_vecs; ++iter){
//        /* 2. Use a standard diagonalization method to solve the L × L eigenvalue
//             * problem G alpha^k = ρ^k  alpha^k, k = 1,2,...,M
//             * where Gij = (b_i,H b_j) = (b_i,sigma_j), 1 ≤ i,j ≤ L (26)
//             * and M is the number of roots of interest.
//             */

//        // Compute H b_i
//        for (int j = nvecs; j < nvecs + L; ++j){
//            Hb_j->gemv(false,1.0,H.get_pointer(),b[j].get_pointer(),0.0);
//            Hb.push_back(Hb_j);
//        }

//        // Compute G
//        for (int i = 0; i < nvecs; ++i){
//            for (int j = nvecs; j < nvecs + L; ++j){
//                double Gij = b[i]->dot(Hb[j].get_pointer());
//                G->set(i,j,Gij);
//                G->set(j,i,Gij);
//            }
//        }
//        for (int i = nvecs; i < nvecs + L; ++i){
//            for (int j = nvecs; j < nvecs + L; ++j){
//                double Gij = b[i]->dot(Hb[j].get_pointer());
//                G->set(i,j,Gij);
//            }
//        }
//        G->diagonalize(alpha,rho);

//    }
}

}} // EndNamespaces

//SharedMatrix R(new Matrix("R",n,L));
//SharedMatrix G(new Matrix("G",L,L));
//SharedVector Rho(new Matrix("G Eigenvalues",L,L));
//Hb->gemm(false,false,1.0,H,b,0.0);
//G->gemm(true,false,1.0,b,Hb,0.0);

//Rho->set_diagonal(rho);
//R->gemm(false,false,1.0,Hb,alpha,0.0);
//R->gemm(false,false,-1.0,b,alpha,1.0);




