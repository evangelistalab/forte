//[forte-public]
#include <cmath>

#include <psifiles.h>
#include <libiwl/iwl.h>
#include <libtrans/integraltransform.h>
#include <libpsio/psio.hpp>
#include <libmints/factory.h>
#include <libmints/matrix.h>
#include <libmints/basisset.h>
#include <libmints/wavefunction.h>
#include <libqt/qt.h>
#include <libfock/jk.h>
#include <algorithm>
#include <numeric>
#include "blockedtensorfactory.h"

#include "integrals.h"
#include "memory.h"

using namespace std;
using namespace psi;
using namespace ambit;

#include <psi4-dec.h>

namespace psi{ namespace forte{

#ifdef _OPENMP
    #include <omp.h>
    bool ForteIntegrals::have_omp_ = true;
#else
    #define omp_get_max_threads() 1
    #define omp_get_thread_num()  0
    bool ForteIntegrals::have_omp_ = false;
#endif


ForteIntegrals::ForteIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : options_(options), restricted_(restricted), resort_frozen_core_(resort_frozen_core),frozen_core_energy_(0.0), scalar_(0.0),
    mo_space_info_(mo_space_info)
{
    startup();
    allocate();
    transform_one_electron_integrals();
}

ForteIntegrals::~ForteIntegrals()
{
    deallocate();
}

void ForteIntegrals::startup()
{
    // Grab the global (default) PSIO object, for file I/O
    boost::shared_ptr<PSIO> psio(_default_psio_lib_);

    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    if (not wfn){
        outfile->Printf("\n  No wave function object found!  Run a scf calculation first!\n");
        outfile->Flush();
        exit(1);
    }

    nirrep_ = wfn->nirrep();
    nso_ = wfn->nso();
    nmo_ = wfn->nmo();
    nmopi_ = wfn->nmopi();
    frzcpi_ = wfn->frzcpi();
    frzvpi_ = wfn->frzvpi();
    frzcpi_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->get_dimension("FROZEN_UOCC");
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    ncmo_ = ncmopi_.sum();

    outfile->Printf("\n\n  ==> Integral Transformation <==\n");
    outfile->Printf("\n  Number of molecular orbitals:            %5d",nmopi_.sum());
    outfile->Printf("\n  Number of correlated molecular orbitals: %5zu",ncmo_);
    outfile->Printf("\n  Number of frozen occupied orbitals:      %5d",frzcpi_.sum());
    outfile->Printf("\n  Number of frozen unoccupied orbitals:    %5d\n\n",frzvpi_.sum());


    // Indexing
    // This is important!  Set the indexing to work using the number of molecular integrals
    aptei_idx_ = nmo_;
    num_oei = INDEX2(nmo_ - 1, nmo_ - 1) + 1;
    num_tei = INDEX4(nmo_ - 1,nmo_ - 1,nmo_ - 1,nmo_ - 1) + 1;
    num_aptei = nmo_ * nmo_ * nmo_ * nmo_;
    num_threads_ = omp_get_max_threads();
    print_       = options_.get_int("PRINT");
}

void ForteIntegrals::ForteIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    one_electron_integrals_a = new double[nmo_ * nmo_];
    one_electron_integrals_b = new double[nmo_ * nmo_];

    fock_matrix_a = new double[nmo_ * nmo_];
    fock_matrix_b = new double[nmo_ * nmo_];
}

void ForteIntegrals::ForteIntegrals::deallocate()
{
    // Deallocate the memory required to store the one-electron integrals
    delete[] one_electron_integrals_a;
    delete[] one_electron_integrals_b;

    delete[] fock_matrix_a;
    delete[] fock_matrix_b;
}

void ForteIntegrals::ForteIntegrals::resort_two(double*& ints,std::vector<size_t>& map)
{
    // Store the integrals in a temporary array of dimension nmo x nmo
    double* temp_ints = new double[nmo_ * nmo_];
    for (size_t p = 0; p < nmo_ * nmo_; ++p){
        temp_ints[p] = 0.0;
    }
    for (size_t p = 0; p < ncmo_; ++p){
        for (size_t q = 0; q < ncmo_; ++q){
            temp_ints[p * ncmo_ + q] = ints[map[p] * nmo_ + map[q]];
        }
    }
    // Delete old integrals and assign the pointer
    delete[] ints;
    ints = temp_ints;
}


void ForteIntegrals::ForteIntegrals::set_oei(double** ints,bool alpha)
{
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    for (int p = 0; p < aptei_idx_; ++p){
        for (int q = 0; q < aptei_idx_; ++q){
            p_oei[p * aptei_idx_ + q] = ints[p][q];
        }
    }
}

void ForteIntegrals::set_oei(size_t p, size_t q,double value,bool alpha)
{
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    p_oei[p * aptei_idx_ + q] = value;
}

void ForteIntegrals::transform_one_electron_integrals()
{
    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<PSIO> psio_ = PSIO::shared_object();
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    SharedMatrix T = SharedMatrix(wfn->matrix_factory()->create_matrix(PSIF_SO_T));
    SharedMatrix V = SharedMatrix(wfn->matrix_factory()->create_matrix(PSIF_SO_V));
    SharedMatrix OneInt = T;
    OneInt->zero();

    T->load(psio_, PSIF_OEI);
    V->load(psio_, PSIF_OEI);

    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();

    T->add(V);
    V->copy(T);
    SharedMatrix OneIntAO = V;

    T->transform(Ca);
    V->transform(Cb);
    OneInt = T;
    OneBody_symm_ = OneInt;
    OneInts_symmetryao_ = OneIntAO;

    for (size_t pq = 0; pq < nmo_ * nmo_; ++pq) one_electron_integrals_a[pq] = 0.0;
    for (size_t pq = 0; pq < nmo_ * nmo_; ++pq) one_electron_integrals_b[pq] = 0.0;

    // Read the one-electron integrals (T + V, restricted)
    int offset = 0;
    for (int h = 0; h < nirrep_; ++h){
        for (int p = 0; p < nmopi_[h]; ++p){
            for (int q = 0; q < nmopi_[h]; ++q){
                one_electron_integrals_a[(p + offset) * nmo_ + q + offset] = T->get(h,p,q);
                one_electron_integrals_b[(p + offset) * nmo_ + q + offset] = V->get(h,p,q);
            }
        }
        offset += nmopi_[h];
    }
}
void ForteIntegrals::compute_frozen_one_body_operator()
{
    Timer FrozenOneBody;

    Dimension frozen_dim = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension nmopi      = mo_space_info_->get_dimension("ALL");
    // Need to get the inactive block of the C matrix
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    Dimension nsopi      = wfn->nsopi();
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix C_core(new Matrix("C_core",nirrep_, nsopi, frozen_dim));

    for(int h = 0; h < nirrep_; h++){
        for(int mu = 0; mu < nsopi[h]; mu++){
            for(int i = 0; i < frozen_dim[h]; i++){
                C_core->set(h, mu, i, Ca->get(h, mu, i));
            }
        }
    }

    boost::shared_ptr<JK> JK_core = JK::build_JK();

    JK_core->set_memory(Process::environment.get_memory() * 0.8);
    /// Already transform everything to C1 so make sure JK does not do this.

    /////TODO: Make this an option in my code
    //JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_core->C_left();

    Cl.clear();
    Cl.push_back(C_core);

    JK_core->compute();

    SharedMatrix F_core = JK_core->J()[0];
    SharedMatrix K_core = JK_core->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);
    F_core->transform(Ca);
    int offset = 0;
    for(int h = 0; h < nirrep_; h++){
        for(int p = 0; p < nmopi[h]; ++p){
            for(int q = 0; q < nmopi[h]; ++q){
                one_electron_integrals_a[(p + offset) * nmo_ + (q + offset)] += F_core->get(h, p, q);
                one_electron_integrals_b[(p + offset) * nmo_ + (q + offset)] += F_core->get(h, p ,q);
            }
        }
        offset += nmopi[h];
    }
    F_core->add(OneBody_symm_);

    frozen_core_energy_ = 0.0;
    double E_frozen = 0.0;
    for(int h = 0; h < nirrep_; h++){
        for(int fr = 0; fr < frozen_dim[h]; fr++)
        {
            E_frozen += OneBody_symm_->get(h, fr, fr) + F_core->get(h, fr, fr);
        }
    }

    OneBody_symm_ = F_core;
    frozen_core_energy_ = E_frozen;

    outfile->Printf("\n  Frozen-core energy        %20.12f a.u.",frozen_core_energy_);


    outfile->Printf("\n\n FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());
}
void ForteIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
{
    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = 0; q < ncmo_; ++q){
            // Builf Fock Diagonal alpha-alpha
            fock_matrix_a[p * ncmo_ + q] = oei_a(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ia[k]) {
                    fock_matrix_a[p * ncmo_ + q] += aptei_aa(p,k,q,k);
                }
                if (Ib[k]) {
                    fock_matrix_a[p * ncmo_ + q] += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_b[p * ncmo_ + q] = oei_b(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ib[k]) {
                    fock_matrix_b[p * ncmo_ + q] += aptei_bb(p,k,q,k);
                }
                if (Ia[k]) {
                    fock_matrix_b[p * ncmo_ + q] += aptei_ab(p,k,q,k);
                }
            }
        }
    }
}

void ForteIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
{
    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = p; q < ncmo_; ++q){
            // Builf Fock Diagonal alpha-alpha
            double fock_a_pq = oei_a(p,q);
            //            fock_matrix_a[p * ncmo_ + q] = oei_a(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ia[k]) {
                    fock_a_pq += aptei_aa(p,k,q,k);
                }
                if (Ib[k]) {
                    fock_a_pq += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_a[p * ncmo_ + q] = fock_matrix_a[q * ncmo_ + p] = fock_a_pq;
            double fock_b_pq = oei_b(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ib[k]) {
                    fock_b_pq += aptei_bb(p,k,q,k);
                }
                if (Ia[k]) {
                    fock_b_pq += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_b[p * ncmo_ + q] = fock_matrix_b[q * ncmo_ + p] = fock_b_pq;
        }
    }
}

void ForteIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
{
    std::vector<double>& fock_diagonal_alpha = fock_diagonals.first;
    std::vector<double>& fock_diagonal_beta = fock_diagonals.second;
    for(size_t p = 0; p < ncmo_; ++p){
        // Builf Fock Diagonal alpha-alpha
        fock_diagonal_alpha[p] =  oei_a(p,p);// roei(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ia[k]) {
                //                fock_diagonal_alpha[p] += diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
                fock_diagonal_alpha[p] += diag_aptei_aa(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ib[k]) {
                //                fock_diagonal_alpha[p] += diag_c_rtei(p,k); //rtei(p,p,k,k);
                fock_diagonal_alpha[p] += diag_aptei_ab(p,k); //rtei(p,p,k,k);
            }
        }
        fock_diagonal_beta[p] =  oei_b(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ib[k]) {
                //                fock_diagonal_beta[p] += diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
                fock_diagonal_beta[p] += diag_aptei_bb(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ia[k]) {
                //                fock_diagonal_beta[p] += diag_c_rtei(p,k); //rtei(p,p,k,k);
                fock_diagonal_beta[p] += diag_aptei_ab(p,k); //rtei(p,p,k,k);
            }
        }
    }
}

void ForteIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
{
    for(size_t p = 0; p < ncmo_; ++p){
        // Builf Fock Diagonal alpha-alpha
        fock_diagonal[p] = oei_a(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ia[k]) {
                fock_diagonal[p] += diag_aptei_aa(p,k);  //diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ib[k]) {
                fock_diagonal[p] += diag_aptei_ab(p,k); // diag_c_rtei(p,k); //rtei(p,p,k,k);
            }
        }
    }
}

void ForteIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
{
    for(size_t p = 0; p < ncmo_; ++p){
        fock_diagonals[p] = oei_b(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ia[k]) {
                fock_diagonals[p] += diag_aptei_ab(p,k);  //diag_c_rtei(p,k); //rtei(p,p,k,k);
            }
            if (Ib[k]) {
                fock_diagonals[p] += diag_aptei_bb(p,k);  //diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
        }
        }
}
void ForteIntegrals::update_integrals(bool freeze_core)
{
    Timer freezeOrbs;
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            if(resort_frozen_core_ == RemoveFrozenMOs){aptei_idx_ = ncmo_;}
        }
    }
    if(print_)
    {
        outfile->Printf("\n Frozen Orbitals takes %8.8f s", freezeOrbs.get());
    }
}

void ForteIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    gather_integrals();
    update_integrals();
}
void ForteIntegrals::freeze_core_orbitals()
{
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
}

}}
