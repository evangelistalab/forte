#include <cmath>

#include <psifiles.h>
#include <libiwl/iwl.h>
#include <libtrans/integraltransform.h>
#include <libpsio/psio.hpp>
#include <libmints/matrix.h>
#include <libmints/basisset.h>
#include <libthce/thce.h>
#include <libthce/thcew.h>
#include <libthce/lreri.h>
#include <lib3index/cholesky.h>
#include <libmints/mints.h>
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
    : options_(options), restricted_(restricted), resort_frozen_core_(resort_frozen_core), frozen_core_energy_(0.0), scalar_(0.0),
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

    if (options_["FROZEN_DOCC"].has_changed()){
        outfile->Printf("\n  Using the input to select the number of frozen core MOs.\n");
        if (options_["FROZEN_DOCC"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzcpi_[h] = options_["FROZEN_DOCC"][h].to_integer();
            }
        }else{
            outfile->Printf("\n\n  The input array FROZEN_DOCC has information for %zu irreps, this does not match the total number of irreps %zu",
                            options_["FROZEN_DOCC"].size(),nirrep_);
            outfile->Printf("\n  Exiting the program.\n");
            printf("  The input array FROZEN_DOCC has information for %d irreps, this does not match the total number of irreps %zu",
                   options_["FROZEN_DOCC"].size(),nirrep_);
            printf("\n  Exiting the program.\n");

            exit(Failure);
        }
    }
    if (options_["FROZEN_UOCC"].has_changed()){
        outfile->Printf("\n  Using the input to select the number of frozen virtual MOs.\n");
        if (options_["FROZEN_UOCC"].size() == nirrep_){
            for (int h = 0; h < nirrep_; ++h){
                frzvpi_[h] = options_["FROZEN_UOCC"][h].to_integer();
            }
        }else{
            outfile->Printf("\n\n  The input array FROZEN_UOCC has information for %zu irreps, this does not match the total number of irreps %d",
                            options_["FROZEN_UOCC"].size(),nirrep_);
            outfile->Printf("\n  Exiting the program.\n");
            printf("  The input array FROZEN_UOCC has information for %d irreps, this does not match the total number of irreps %zu",
                   options_["FROZEN_UOCC"].size(),nirrep_);
            printf("\n  Exiting the program.\n");

            exit(Failure);
        }
    }

    ncmopi_ = nmopi_;
    for (int h = 0; h < nirrep_; ++h){
        ncmopi_[h] -= frzcpi_[h] + frzvpi_[h];
    }
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
}

void ForteIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    one_electron_integrals_a = new double[nmo_ * nmo_];
    one_electron_integrals_b = new double[nmo_ * nmo_];

    fock_matrix_a = new double[nmo_ * nmo_];
    fock_matrix_b = new double[nmo_ * nmo_];
}

void ForteIntegrals::deallocate()
{
    // Deallocate the memory required to store the one-electron integrals
    delete[] one_electron_integrals_a;
    delete[] one_electron_integrals_b;

    delete[] fock_matrix_a;
    delete[] fock_matrix_b;
}

void ForteIntegrals::resort_two(double*& ints,std::vector<size_t>& map)
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


void ForteIntegrals::set_oei(double** ints,bool alpha)
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

    T->load(psio_, PSIF_OEI);
    V->load(psio_, PSIF_OEI);

    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();

    T->add(V);
    V->copy(T);
    T->transform(Ca);
    V->transform(Cb);

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

/**
     * @brief ConventionalIntegrals::ConventionalIntegrals
     * @param options - psi options class
     * @param restricted - type of integral transformation
     * @param resort_frozen_core -
     */


}}
