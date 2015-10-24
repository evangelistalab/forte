#include <cmath>

#include <boost/timer.hpp>

#include "iterative_solvers.h"

#define PRINT_VARS(msg) \
//    std::vector<std::pair<size_t,std::string>> v = \
//        {{collapse_size_,"collapse_size_"}, \
//        {subspace_size_,"subspace_size_"}, \
//        {basis_size_,"basis_size_"}, \
//        {sigma_size_,"sigma_size_"}, \
//        {nroot_,"nroot_"}}; \
//    outfile->Printf("\n\n => %s <=",msg); \
//    for (auto vk : v){ \
//        outfile->Printf("\n    %-30s  %zu",vk.second.c_str(),vk.first); \
//    }


namespace psi{ namespace forte{

DavidsonLiuSolver::DavidsonLiuSolver(size_t size,size_t nroot)
    : size_(size), nroot_(nroot)
{
    if (size_ == 0) throw std::runtime_error("DavidsonLiuSolver called with space of dimension zero.");
}

void DavidsonLiuSolver::startup(SharedVector diagonal)
{
    // set space size
    collapse_size_ = std::min(collapse_per_root_ * nroot_,size_);
    subspace_size_ = std::min(subspace_per_root_ * nroot_,size_);

    basis_size_ = 0; // start with no vectors
    sigma_size_ = 0; // start with no sigmas vectors
    iter_ = 0;
    converged_ = 0;

    b_ = SharedMatrix(new Matrix("b",subspace_size_,size_));
    b_->zero();
    bnew = SharedMatrix(new Matrix("bnew",subspace_size_,size_));
    f = SharedMatrix(new Matrix("f",subspace_size_,size_));
    sigma_ = SharedMatrix(new Matrix("sigma",size_,subspace_size_));

    G = SharedMatrix(new Matrix("G",subspace_size_,subspace_size_));
    S = SharedMatrix(new Matrix("S",subspace_size_,subspace_size_));
    alpha = SharedMatrix(new Matrix("alpha",subspace_size_,subspace_size_));

    lambda = SharedVector(new Vector("lambda",subspace_size_));
    lambda_old = SharedVector(new Vector("lambda",subspace_size_));
    h_diag = SharedVector(new Vector("lambda",size_));

    h_diag->copy(*diagonal);
}

DavidsonLiuSolver::~DavidsonLiuSolver()
{

}

void DavidsonLiuSolver::set_print_level(size_t n)
{
    print_level_ = n;
}

void DavidsonLiuSolver::set_e_convergence(double value)
{
    e_convergence_ = value;
}

void DavidsonLiuSolver::set_r_convergence(double value)
{
    r_convergence_ = value;
}

void DavidsonLiuSolver::set_collapse_per_root(int value)
{
    collapse_per_root_ = value;
}

void DavidsonLiuSolver::set_subspace_per_root(int value)
{
    subspace_per_root_ = value;
}

size_t DavidsonLiuSolver::collapse_size() const
{
    return collapse_size_;
}

void DavidsonLiuSolver::add_guess(SharedVector vec)
{
    // Give the next b that does not have a sigma
    for (size_t j = 0; j < size_; ++j){
        b_->set(basis_size_,j,vec->get(j));
    }
    basis_size_++;
}

void DavidsonLiuSolver::get_b(SharedVector vec)
{
    PRINT_VARS("get_b")
    // Give the next b that does not have a sigma
    for (size_t j = 0; j < size_; ++j){
        vec->set(j,b_->get(sigma_size_,j));
    }
}

bool DavidsonLiuSolver::add_sigma(SharedVector vec)
{
    PRINT_VARS("add_sigma")
    // Place the new sigma vector at the end
    for (size_t j = 0; j < size_; ++j){
        sigma_->set(j,sigma_size_,vec->get(j));
    }
    sigma_size_++;
    return (sigma_size_ < basis_size_);
}

void DavidsonLiuSolver::set_project_out(std::vector<sparse_vec> project_out)
{
    project_out_ = project_out;
}

SharedVector DavidsonLiuSolver::eigenvalues() const
{
    return lambda;
}

SharedMatrix DavidsonLiuSolver::eigenvectors() const
{
    return b_;
}

SharedVector DavidsonLiuSolver::eigenvector(size_t n) const
{
    double** v = bnew->pointer();

    SharedVector evec(new Vector("V",size_));
    for(size_t I = 0; I < size_; I++) {
        evec->set(I,v[n][I]);
    }
    return evec;
}

SolverStatus DavidsonLiuSolver::update()
{
    // If converged or exceeded the maximum number of iterations return true
    if ((converged_ >= nroot_) or (iter_ > maxiter_)) return SolverStatus::Converged;

    PRINT_VARS("update")

    boost::timer t_davidson;

    // form and diagonalize mini-matrix
    G->zero();
    G->gemm(false,false,1.0,b_,sigma_,0.0);
    G->diagonalize(alpha,lambda);

    if (not last_update_collapsed_){
        if(check_convergence()){
            get_results();
            return SolverStatus::Converged;
        }
    }

    if (size_ == 1) return SolverStatus::Converged;

    check_orthogonality();

    // Do subspace collapse
    if(subspace_collapse()) {
        last_update_collapsed_ = true;
        return SolverStatus::Collapse;
    }else{
        last_update_collapsed_ = false;
    }

    // Step #3: Build the Correction Vectors
    // form preconditioned residue vectors
    form_correction_vectors();

    // Step #3b: Project out undesired roots
    project_out_roots(f);

    // Step #4: Orthonormalize the Correction Vectors
    normalize_vectors(f,nroot_);

    // schmidt orthogonalize the f[k] against the set of b[i] and add new vectors
    for(size_t k = 0; k < nroot_; k++){
        if (basis_size_ < subspace_size_){
            double norm_bnew_k = std::fabs(f->get_row(0,k)->norm());
            if (norm_bnew_k > schmidt_threshold_){
                if(schmidt_add(b_->pointer(), basis_size_, size_, f->pointer()[k])) {
                    basis_size_++;  // <- Increase L if we add one more basis vector
                }else{
                }
            }
        }
    }

    iter_++;

    timing_ += t_davidson.elapsed();

    return SolverStatus::NotConverged;
}

void DavidsonLiuSolver::form_correction_vectors()
{
    f->zero();
    double* lambda_p = lambda->pointer();
    double* Adiag_p = h_diag->pointer();
    double** b_p = b_->pointer();
    double** f_p = f->pointer();
    double** alpha_p = alpha->pointer();
    double** sigma_p = sigma_->pointer();

    for(size_t k = 0; k < nroot_; k++){  // loop over roots
        for(size_t I = 0; I < size_; I++) {  // loop over elements
            for(size_t i = 0; i < basis_size_; i++) {
                f_p[k][I] += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
            }
            double denom = lambda_p[k] - Adiag_p[I];
            if(fabs(denom) > 1e-6){
                f_p[k][I] /= denom;
            }
            else{
                f_p[k][I] = 0.0;
            }
        }
    }
}

void DavidsonLiuSolver::project_out_roots(SharedMatrix v)
{
    double** v_p = v->pointer();
    for(size_t k = 0; k < nroot_; k++) {
        for (auto& bad_root : project_out_){
            double overlap = 0.0;
            for (auto& I_CI : bad_root){
                size_t I = I_CI.first;
                double CI = I_CI.second;
                overlap += v_p[k][I] * CI;
            }
            for (auto& I_CI : bad_root){
                size_t I = I_CI.first;
                double CI = I_CI.second;
                v_p[k][I] -= overlap * CI;
            }
        }
    }
}

void DavidsonLiuSolver::normalize_vectors(SharedMatrix v,size_t n)
{
    // normalize each residual
    double** v_p = v->pointer();
    for(size_t k = 0; k < n; k++) {
        double norm = 0.0;
        for(size_t I = 0; I < size_; I++) {
            norm += v_p[k][I] * v_p[k][I];
        }
        norm = std::sqrt(norm);
        for(size_t I = 0; I < size_; I++) {
            v_p[k][I] /= norm;
        }
    }
}

bool DavidsonLiuSolver::subspace_collapse()
{
    if(collapse_size_ + nroot_ > subspace_size_){ // in this case I will never be able to add new vectors

        // collapse vectors
        collapse_vectors();

        // normalize new vectors
        normalize_vectors(bnew,collapse_size_);

        // Copy them into place
        b_->zero();
        basis_size_ = 0;
        sigma_size_ = 0;
        for(size_t k = 0; k < collapse_size_; k++){
            double norm_bnew_k = std::fabs(bnew->get_row(0,k)->norm());
            if (norm_bnew_k > schmidt_threshold_){
                if(schmidt_add(b_->pointer(),k,size_, bnew->pointer()[k])) {
                    basis_size_++;  // <- Increase L if we add one more basis vector
                }
            }
        }
        return false;
    }

    // If L is close to maxdim, collapse to one guess per root */
    if(nroot_ + basis_size_ > subspace_size_) { // this means that next iteration I cannot add more roots so I need to collapse
        if(print_level_ > 1) {
            outfile->Printf("\nSubspace too large: max subspace size = %d, basis size = %d\n", subspace_size_, basis_size_);
            outfile->Printf("Collapsing eigenvectors.\n");
        }
        // collapse vectors
        collapse_vectors();

        // normalize new vectors
        normalize_vectors(bnew,collapse_size_);

        // Copy them into place
        b_->zero();
        basis_size_ = 0;
        sigma_size_ = 0;
        for(size_t k = 0; k < collapse_size_; k++){
            if(schmidt_add(b_->pointer(),k,size_, bnew->pointer()[k])) {
                basis_size_++;  // <- Increase L if we add one more basis vector
            }
        }

        /// Need new sigma vectors to continue, so return control to caller
        return true;
    }
    return false;
}

void DavidsonLiuSolver::collapse_vectors()
{
    bnew->zero();
    double** alpha_p = alpha->pointer();
    double** b_p = b_->pointer();
    double** bnew_p = bnew->pointer();
    for(size_t i = 0; i < collapse_size_; i++) {
        for(size_t j = 0; j < basis_size_; j++) {
            for(size_t k = 0; k < size_; k++) {
                bnew_p[i][k] += alpha_p[j][i] * b_p[j][k];
            }
        }
    }
}

bool DavidsonLiuSolver::check_convergence()
{
    // check convergence on all roots
    bool has_converged = false;
    converged_ = 0;
    if(print_level_ > 1) {
        outfile->Printf("\n  Root      Eigenvalue        Delta   Converged?\n");
        outfile->Printf("  ---- -------------------- --------- ----------\n");
    }
    for(size_t k = 0; k < nroot_; k++) {
        double diff = std::fabs(lambda->get(k) - lambda_old->get(k));
        bool this_converged = false;
        if(diff < e_convergence_) {
            this_converged = true;
            converged_++;
        }
        lambda_old->set(k,lambda->get(k));
        if(print_level_ > 1) {
            outfile->Printf("  %3d  %20.14f %4.3e      %1s\n", k, lambda->get(k), diff,
                            this_converged ? "Y" : "N");
        }
    }

    if (converged_ == nroot_){
        has_converged = true;
    }
    return has_converged;
}

void DavidsonLiuSolver::get_results()
{
    /* generate final eigenvalues and eigenvectors */
    double** alpha_p = alpha->pointer();
    double** b_p = b_->pointer();
    double* eps = lambda_old->pointer();
    double** v = bnew->pointer();
    bnew->zero();

    for(size_t i = 0; i < nroot_; i++) {
        eps[i] = lambda->get(i);
        for(size_t j = 0; j < basis_size_; j++) {
            for(size_t I = 0; I < size_; I++) {
                v[i][I] += alpha_p[j][i] * b_p[j][I];
            }
        }
        // Normalize v
        double norm = 0.0;
        for(size_t I = 0; I < size_; I++) {
            norm += v[i][I] * v[i][I];
        }
        norm = std::sqrt(norm);
        for(size_t I = 0; I < size_; I++) {
            v[i][I] /= norm;
        }
    }
//    if (print_level_){
//        outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.", iter_);
//        outfile->Printf("\n  %s: %f s","Time spent diagonalizing H",timing_);
//    }
}

bool DavidsonLiuSolver::check_orthogonality()
{
    bool is_orthonormal = true;

    // Compute the overlap matrix
    S->gemm(false,true,1.0,b_,b_,0.0);

    // Check for orthogonality
    for (size_t i = 0; i < basis_size_; ++i){
        double diag = S->get(i,i);
        double zero = false;
        double one = false;
        if (std::fabs(diag - 1.0) < 1.e-6){
            one = true;
        }
        if (std::fabs(diag) < 1.e-6){
            zero = true;
        }
        if ((not zero) and (not one)){
            if (is_orthonormal) {
                outfile->Printf("\n  WARNING: Vector %d is not normalized or zero",i);
                is_orthonormal = false;
            }
        }
        double offdiag = 0.0;
        for (size_t j = i + 1; j < basis_size_; ++j){
            offdiag += std::fabs(S->get(i,j));
        }
        if (offdiag > 1.0e-6){
            if (is_orthonormal) {
                outfile->Printf("\n  WARNING: The vectors are not orthogonal");
                is_orthonormal = false;
            }

        }
    }
    return is_orthonormal;
}

}}
