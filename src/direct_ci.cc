#include <cmath>

#include <boost/timer.hpp>

#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libpsio/psio.hpp>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>

#include "forte-def.h"
#include "iterative_solvers.h"
#include "direct_ci.h"
#include "fci_vector.h"

struct PairHash{
    size_t operator()( const std::pair<size_t, size_t>& p ) const {
        return (p.first*1000) + p.second;
    }
};

namespace psi{ namespace forte{

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_max_threads() 1
    #define omp_get_thread_num() 0
#endif



SigmaBuilder::SigmaBuilder(WFNOperator& op, DeterminantMap& wfn ) : size_(wfn.size()), wfn_(wfn), op_(op)
{
    // Copy coupling lists from operator object
}

void SigmaBuilder::compute_sigma(SharedVector sigma, SharedVector b)
{
    sigma->zero();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    

    // First do the one-particle term
    {
        // Get a reference to the wavefunction map
        const det_hash<size_t>& wfn_map = wfn_.wfn();    
        
        // Get references to creation/annihilation lists 
        auto& a_ann_list = op_.a_ann_list_; 
        auto& b_ann_list = op_.b_ann_list_; 

        auto& a_cre_list = op_.a_cre_list_; 
        auto& b_cre_list = op_.b_cre_list_; 

        for( auto& det_pair : wfn_map ){
            // aa singles
            size_t J = det_pair.second;
            STLBitsetDeterminant detJ = det_pair.first;

            for (auto& aJ_mo_sign : a_ann_list[J]){
                const size_t aJ_add = aJ_mo_sign.first;
                const size_t p = std::abs(aJ_mo_sign.second) - 1;
                for (auto& aaJ_mo_sign : a_cre_list[aJ_add]){
                    const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                    if (p != q){
                        const double HIJ = detJ.slater_rules_single_alpha(p,q);
                        const size_t I = aaJ_mo_sign.first;
                        sigma_p[I] += HIJ * b_p[J];
                    }
                }
            }
            for (auto& bJ_mo_sign : b_ann_list[J]){
                const size_t bJ_add = bJ_mo_sign.first;
                const size_t p = std::abs(bJ_mo_sign.second) - 1;
                for (auto& bbJ_mo_sign : b_cre_list[bJ_add]){
                    const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                    if (p != q){
                        const double HIJ = detJ.slater_rules_single_beta(p,q);
                        const size_t I = bbJ_mo_sign.first;
                        sigma_p[I] += HIJ * b_p[J];
                    }
                }
            }
        }
    }

    // Then the two particle term

    // Get references to creation/annihilation lists 
    auto& aa_ann_list = op_.aa_ann_list_; 
    auto& ab_ann_list = op_.ab_ann_list_; 
    auto& bb_ann_list = op_.bb_ann_list_; 

    auto& aa_cre_list = op_.aa_cre_list_; 
    auto& ab_cre_list = op_.ab_cre_list_; 
    auto& bb_cre_list = op_.bb_cre_list_; 
    
    for (size_t J = 0; J < size_; ++J){
        // reference
        sigma_p[J] += diag_[J] * b_p[J];
        for (auto& aaJ_mo_sign : aa_ann_list[J]){
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]){
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                    const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = aaaaJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }
        // aabb singles
        for (auto& abJ_mo_sign : ab_ann_list[J]){
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]){
                const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                const size_t s = std::get<2>(ababJ_mo_sign);
                if ((p != r) and (q != s)){
                    const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                    const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = ababJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }
        // bbbb singles
        for (auto& bbJ_mo_sign : bb_ann_list[J]){
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]){
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                    const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = bbbbJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }
    }
}
void SigmaBuilder::get_diagonal(Vector& diag)
{
    for (size_t I = 0; I < diag_.size(); ++I){
        diag.set(I,diag_[I]);
    }
}

void DirectCI::set_spin_project(bool value)
{
    spin_project_ = value;
}

void DirectCI::set_e_convergence(double value)
{
    e_convergence_ = value;
}

void DirectCI::set_maxiter_davidson(int value)
{
    maxiter_davidson_ = value;
}

/*
void DirectCI::diagonalize_hamiltonian(const std::vector<STLBitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int nroot,int multiplicity,DiagonalizationMethod diag_method)
{
    if (space.size() <= 200 && !force_diag_method_){
        diagonalize_full(space,evals,evecs,nroot,multiplicity);
    }else{
        if (diag_method == Full){
            diagonalize_full(space,evals,evecs,nroot,multiplicity);
        }else if (diag_method == DLSolver ){
            diagonalize_davidson_liu_solver(space,evals,evecs,nroot,multiplicity);
        }else if (diag_method == DLString ){
            diagonalize_davidson_liu_string(space,evals,evecs,nroot,multiplicity, false);
        }else if (diag_method == DLDisk ){
            diagonalize_davidson_liu_string(space,evals,evecs,nroot,multiplicity, true);
        }
    }
}

void DirectCI::diagonalize_full(const std::vector<STLBitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int,int)
{
    // Find all the eigenvalues and eigenvectors of the Hamiltonian
    SharedMatrix H = build_full_hamiltonian(space);

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,dim_space));
    evals.reset(new Vector("e",dim_space));

    // Diagonalize H
    //boost::timer t_diag;
    H->diagonalize(evecs,evals);
}

void DirectCI::diagonalize_davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals, SharedMatrix& evecs, int nroot, int multiplicity)
{
	if( print_details_ ){	
		outfile->Printf("\n\n  Davidson-liu solver algorithm");
		outfile->Flush();
	}

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));

    // Diagonalize H
    SigmaVectorList svl (space, print_details_);
    SigmaVector* sigma_vector = &svl;
    davidson_liu_solver(space,sigma_vector,evals,evecs,nroot,multiplicity);
}

void DirectCI::diagonalize_davidson_liu_string(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals, SharedMatrix& evecs, int nroot, int multiplicity, bool disk)
{
	if( print_details_ and !disk ){	
		outfile->Printf("\n\n  Davidson-Liu String algorithm");
		outfile->Flush();
	}else if ( print_details_ and disk ){
		outfile->Printf("\n\n  Disk-based Davidson-Liu String Algorithm");
    }

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));

    // Diagonalize H
    SigmaVectorString svl (space, print_details_, disk);
    SigmaVector* sigma_vector = &svl;
    davidson_liu_solver(space,sigma_vector,evals,evecs,nroot,multiplicity);
}

SharedMatrix DirectCI::build_full_hamiltonian(const std::vector<STLBitsetDeterminant> &space)
{
    // Build the H matrix
    size_t dim_space = space.size();
    SharedMatrix H(new Matrix("H",dim_space,dim_space));
    //If you are using DiskDF, Kevin found that openmp does not like this! 
    int threads = 0;
    if(STLBitsetDeterminant::fci_ints_->get_integral_type()==DiskDF)
    {
       threads = 1;
    }
    else
    {
       threads = omp_get_max_threads();
    }
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t I = 0; I < dim_space; ++I){
        const STLBitsetDeterminant& detI = space[I];
        for (size_t J = I; J < dim_space; ++J){
            const STLBitsetDeterminant& detJ = space[J];
            double HIJ = detI.slater_rules(detJ);
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
    }
    return H;
}

std::vector<std::pair<std::vector<int>,std::vector<double>>> DirectCI::build_sparse_hamiltonian(const std::vector<STLBitsetDeterminant> &space)
{
    boost::timer t_h_build2;
    // Allocate as many elements as we need
    size_t dim_space = space.size();
    std::vector<std::pair<std::vector<int>,std::vector<double>>> H_sparse(dim_space);

    size_t num_nonzero = 0;

    outfile->Printf("\n  Building H using OpenMP-take2");
    outfile->Flush();

    // Form the Hamiltonian matrix

#pragma omp parallel for schedule(dynamic)
    for (size_t I = 0; I < dim_space; ++I){
        std::vector<double> H_row;
        std::vector<int> index_row;
        const STLBitsetDeterminant& detI = space[I];
        double HII = detI.slater_rules(detI);
        H_row.push_back(HII);
        index_row.push_back(I);
        for (size_t J = 0; J < dim_space; ++J){
            if (I != J){
                const STLBitsetDeterminant detJ = space[J];
                double HIJ = detI.slater_rules(detJ);
                if (std::fabs(HIJ) >= 1.0e-12){
                    H_row.push_back(HIJ);
                    index_row.push_back(J);
                }
            }
        }

#pragma omp critical(save_h_row)
        {
            H_sparse[I] = make_pair(index_row,H_row);
            num_nonzero += index_row.size();
        }
    }
    outfile->Printf("\n  The sparse Hamiltonian matrix contains %zu nonzero elements out of %zu (%f)",num_nonzero,dim_space * dim_space,double(num_nonzero)/double(dim_space * dim_space));
    outfile->Printf("\n  %s: %f s","Time spent building H (openmp)",t_h_build2.elapsed());
    outfile->Flush();
    return H_sparse;
}

std::vector<std::pair<double,std::vector<std::pair<size_t,double>>>> DirectCI::initial_guess(const std::vector<STLBitsetDeterminant>& space, int nroot, int multiplicity)
{
    size_t ndets = space.size();
    size_t nguess = std::min(static_cast<size_t>(nroot) * 100,ndets);
    std::vector<std::pair<double,std::vector<std::pair<size_t,double>>>> guess(nguess);

    // Find the ntrial lowest diagonals
    std::vector<std::pair<STLBitsetDeterminant,size_t>> guess_dets_pos;
    std::vector<std::pair<double,size_t>> smallest(ndets);
    for(size_t I = 0; I < ndets; ++I){
        smallest[I] = std::make_pair(space[I].energy(),I);
    }
    std::sort(smallest.begin(),smallest.end());

    std::vector<STLBitsetDeterminant> guess_det;
    for(size_t i = 0; i < nguess; i++) {
        size_t I = smallest[i].second;
        guess_dets_pos.push_back(std::make_pair(space[I],I));  // store a det and its position
        guess_det.push_back(space[I]);
    }

    if (spin_project_){
        STLBitsetDeterminant::enforce_spin_completeness(guess_det);
        if (guess_det.size() > nguess){
            size_t nnew_dets = guess_det.size() - nguess;
            if (print_details_) outfile->Printf("\n  Initial guess space is incomplete.\n  Trying to add %d determinant(s).",nnew_dets);
            int nfound = 0;
            for (size_t i = 0; i < nnew_dets; ++i){
                for (size_t j = nguess; j < ndets; ++j){
                    size_t J = smallest[j].second;
                    if (space[J] == guess_det[nguess + i]){
                        guess_dets_pos.push_back(std::make_pair(space[J],J));  // store a det and its position
                        nfound++;
                        break;
                    }
                }
            }
            if(print_details_) outfile->Printf("  %d determinant(s) added.",nfound);
        }
        nguess = guess_dets_pos.size();
    }

    // Form the S^2 operator matrix and diagonalize it
    Matrix S2("S^2",nguess,nguess);
    for(size_t I = 0; I < nguess; I++) {
        for(size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double S2IJ = detI.spin2(detJ);
            S2.set(I,J,S2IJ);
            S2.set(J,I,S2IJ);
        }
    }
    Matrix S2evecs("S^2",nguess,nguess);
    Vector S2evals("S^2",nguess);
    S2.diagonalize(S2evecs,S2evals);

    // Form the Hamiltonian
    Matrix H("H",nguess,nguess);
    for(size_t I = 0; I < nguess; I++) {
        for(size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double HIJ = detI.slater_rules(detJ);
            H.set(I,J,HIJ);
            H.set(J,I,HIJ);
        }
    }
   // H.print();
    // Project H onto the spin-adapted subspace
    H.transform(S2evecs);

    // Find groups of solutions with same spin
    double Stollerance = 1.0e-6;
    std::map<int,std::vector<int>> mult_list;
    for (size_t i = 0; i < nguess; ++i){
        double mult = std::sqrt(1.0 + 4.0 * S2evals.get(i)); // 2S + 1 = Sqrt(1 + 4 S (S + 1))
        int mult_int = std::round(mult);
        double error = mult - static_cast<double>(mult_int);
        if (std::fabs(error) < Stollerance){
            mult_list[mult_int].push_back(i);
        }else if (print_details_) {
            outfile->Printf("\n  Found a guess vector with spin not close to integer value (%f)",mult);
        }
    }
    if (mult_list[multiplicity].size() < static_cast<size_t>(nroot)){
        size_t nfound = mult_list[multiplicity].size();
        outfile->Printf("\n  Error: %d guess vectors with 2S+1 = %d but only %d were found!",nguess,multiplicity,nfound);
        if(nfound== 0 ){exit(1);}
    }

    std::vector<int> mult_vals;
    for (auto kv : mult_list){
        mult_vals.push_back(kv.first);
    }
    std::sort(mult_vals.begin(),mult_vals.end());

    for (int m : mult_vals){
        std::vector<int>& mult_list_s = mult_list[m];
        int nspin_states = mult_list_s.size();
        if (print_details_)outfile->Printf("\n  Initial guess found %d solutions with 2S+1 = %d %c",nspin_states,m,m == multiplicity ? '*' : ' ');
        // Extract the spin manifold
        Matrix HS2("HS2",nspin_states,nspin_states);
        Vector HS2evals("HS2",nspin_states);
        Matrix HS2evecs("HS2",nspin_states,nspin_states);
        for(int I = 0; I < nspin_states; I++) {
            for(int J = 0; J < nspin_states; J++) {
                HS2.set(I,J,H.get(mult_list_s[I],mult_list_s[J]));
            }
        }
        HS2.diagonalize(HS2evecs,HS2evals);

        // Project the spin-adapted solution onto the full manifold
        for (int r = 0; r < nspin_states; ++r){
            std::vector<std::pair<size_t,double>> det_C;
            for (size_t I = 0; I < nguess; I++) {
                double CIr = 0.0;
                for (int J = 0; J < nspin_states; ++J){
                    CIr += S2evecs.get(I,mult_list_s[J]) * HS2evecs(J,r);
                }
                det_C.push_back(std::make_pair(guess_dets_pos[I].second,CIr));
            }
            guess.push_back(std::make_pair(m,det_C));
        }
    }

    return guess;

//    // Check the spin
//    for (int r = 0; r < nguess; ++r){
//        double s2 = 0.0;
//        double e = 0.0;
//        for (size_t i = 0; i < nguess; i++) {
//            for (size_t j = 0; j < nguess; j++) {
//                size_t I = guess_dets_pos[i].second;
//                size_t J = guess_dets_pos[j].second;
//                double CI = evecs->get(I,r);
//                double CJ = evecs->get(J,r);
//                s2 += space[I].spin2(space[J]) * CI * CJ;
//                e += space[I].slater_rules(space[J]) * CI * CJ;
//            }
//        }
//        outfile->Printf("\n  Guess Root %d: <E> = %f, <S^2> = %f",r,e,s2);
//    }
 }

bool DirectCI::davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space,
                                         SigmaVector* sigma_vector,
                                         SharedVector Eigenvalues,
                                         SharedMatrix Eigenvectors,
                                         int nroot,
                                         int multiplicity)
{
//    print_details_ = true;
    size_t fci_size = sigma_vector->size();
    DavidsonLiuSolver dls(fci_size,nroot);
    dls.set_e_convergence(e_convergence_);
    dls.set_print_level(0);


    // allocate vectors
    SharedVector b(new Vector("b",fci_size));
    SharedVector sigma(new Vector("sigma",fci_size));

    // get and pass diagonal
    sigma_vector->get_diagonal(*sigma);
    dls.startup(sigma);

    size_t guess_size = dls.collapse_size();
    if (print_details_) outfile->Printf("\n  number of guess vectors: %d",guess_size);

    auto guess = initial_guess(space,nroot,multiplicity);

    std::vector<int> guess_list;
    for (size_t g = 0; g < guess.size(); ++g){
        if (guess[g].first == multiplicity) guess_list.push_back(g);
    }

    // number of guess to be used
    size_t nguess = std::min(guess_list.size(),guess_size);

    if (nguess == 0){
        throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested multiplicity.\n\n");
    }

    for (size_t n = 0; n < nguess; ++n){
        b->zero();
        for (auto& guess_vec_info : guess[guess_list[n]].second){
            b->set(guess_vec_info.first,guess_vec_info.second);
        }
        if( print_details_ ) outfile->Printf("\n  Adding guess %d (multiplicity = %f)",n,guess[guess_list[n]].first);
        dls.add_guess(b);
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    std::vector<std::vector<std::pair<size_t,double>>> bad_roots;
    for (auto& g : guess){
        if (g.first != multiplicity) bad_roots.push_back(g.second);
    }
    dls.set_project_out(bad_roots);

    SolverStatus converged = SolverStatus::NotConverged;
    
//    for( int i = 0; i < fci_size; ++i ){
//        if( b->get(i) > 0 ){
//        outfile->Printf("\n b: %f", b->get(i));
//        space[i].print();
//        outfile->Printf("\n");
//        }
//    }

    if(print_details_){
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;

//    maxiter_davidson_ = 2;
//    b->print();
    for (int cycle = 0; cycle < maxiter_davidson_; ++cycle){
        bool add_sigma = true;
        do{
            dls.get_b(b);
            sigma_vector->compute_sigma(sigma,b);
            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse){
            double avg_energy = 0.0;
            for (int r = 0; r < nroot; ++r) avg_energy += dls.eigenvalues()->get(r);
            avg_energy /= static_cast<double>(nroot);
            if (print_details_){
                outfile->Printf("\n    %3d  %20.12f  %+.3e",real_cycle,avg_energy,avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged) break;
    }

    if (print_details_){
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged){
            outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.", real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged){
        outfile->Printf("\n  FCI did not converge!");
        exit(1);
    }

//    dls.get_results();
    SharedVector evals = dls.eigenvalues();
    SharedMatrix evecs = dls.eigenvectors();
    for (int r = 0; r < nroot; ++r){
        Eigenvalues->set(r,evals->get(r));
        for (size_t I = 0; I < fci_size; ++I){
            Eigenvectors->set(I,r,evecs->get(r,I));
        }
    }
    return true;
}
*/

}}
