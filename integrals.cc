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

namespace psi{ namespace libadaptive{

#ifdef _OPENMP
    #include <omp.h>
    bool ExplorerIntegrals::have_omp_ = true;
#else
    #define omp_get_max_threads() 1
    #define omp_get_thread_num()  0
    bool ExplorerIntegrals::have_omp_ = false;
#endif


ExplorerIntegrals::ExplorerIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : options_(options), restricted_(restricted), resort_frozen_core_(resort_frozen_core), frozen_core_energy_(0.0), scalar_(0.0)
{
    startup();
    allocate();
    transform_one_electron_integrals();
}

ExplorerIntegrals::~ExplorerIntegrals()
{
    deallocate();
}

void ExplorerIntegrals::startup()
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

void ExplorerIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    one_electron_integrals_a = new double[nmo_ * nmo_];
    one_electron_integrals_b = new double[nmo_ * nmo_];

    fock_matrix_a = new double[nmo_ * nmo_];
    fock_matrix_b = new double[nmo_ * nmo_];
}

void ExplorerIntegrals::deallocate()
{
    // Deallocate the memory required to store the one-electron integrals
    delete[] one_electron_integrals_a;
    delete[] one_electron_integrals_b;

    delete[] fock_matrix_a;
    delete[] fock_matrix_b;
}

void ExplorerIntegrals::resort_two(double*& ints,std::vector<size_t>& map)
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


void ExplorerIntegrals::set_oei(double** ints,bool alpha)
{
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    for (int p = 0; p < nmo_; ++p){
        for (int q = 0; q < nmo_; ++q){
            p_oei[p * nmo_ + q] = ints[p][q];
        }
    }
}

void ExplorerIntegrals::set_oei(size_t p, size_t q,double value,bool alpha)
{
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    p_oei[p * nmo_ + q] = value;
}

void ExplorerIntegrals::transform_one_electron_integrals()
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
ConventionalIntegrals::ConventionalIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : ExplorerIntegrals(options, restricted, resort_frozen_core), ints_(nullptr){
    integral_type_ = ConventionalInts;

    outfile->Printf("\n Overall Conventional Integrals timings");
    Timer ConvTime;
    allocate();
    transform_integrals();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n Conventional integrals take %8.8f s", ConvTime.get());
}

ConventionalIntegrals::~ConventionalIntegrals()
{
    deallocate();
}

void ConventionalIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals

    // Allocate the memory required to store the two-electron integrals
    aphys_tei_aa = new double[num_aptei];
    aphys_tei_ab = new double[num_aptei];
    aphys_tei_bb = new double[num_aptei];

    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

}

void ConventionalIntegrals::deallocate()
{
    // Deallocate the integral transform object if needed
    if (ints_ != nullptr) delete ints_;

    // Allocate the memory required to store the two-electron integrals
    delete[] aphys_tei_aa;
    delete[] aphys_tei_ab;
    delete[] aphys_tei_bb;

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;

    //delete[] qt_pitzer_;
}

void ConventionalIntegrals::transform_integrals()
{
    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    // For now, we'll just transform for closed shells and generate all integrals.
    std::vector<boost::shared_ptr<MOSpace> > spaces;

    // TODO: transform only the orbitals within an energy range to save time on this step.
    spaces.push_back(MOSpace::all);

    // If the integral
    if (ints_ != nullptr) delete ints_;

    // Call IntegralTransform asking for integrals over restricted or unrestricted orbitals
    if (restricted_){
        ints_ = new IntegralTransform(wfn, spaces, IntegralTransform::Restricted, IntegralTransform::IWLOnly,IntegralTransform::PitzerOrder,IntegralTransform::None);
    }else{
        ints_ = new IntegralTransform(wfn, spaces, IntegralTransform::Unrestricted, IntegralTransform::IWLOnly,IntegralTransform::PitzerOrder,IntegralTransform::None);
    }

    // Keep the SO integrals on disk in case we want to retransform them
    ints_->set_keep_iwl_so_ints(true);
    Timer int_timer;
    ints_->transform_tei(MOSpace::all, MOSpace::all, MOSpace::all, MOSpace::all);

    outfile->Printf("\n  Integral transformation done. %8.8f s", int_timer.get());
    outfile->Flush();

    //qt_pitzer_ = ints_->alpha_corr_to_pitzer();

    //for(size_t p = 0; p < nmo_; p++){
    //  outfile->Printf("\nqt_pitzer_[%d] = %d", p, qt_pitzer_[p]);
    //}

}

double ConventionalIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    return aphys_tei_aa[aptei_index(p,q,r,s)];
}

double ConventionalIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
    return aphys_tei_ab[aptei_index(p,q,r,s)];
}

double ConventionalIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
    return aphys_tei_bb[aptei_index(p,q,r,s)];
}

ambit::Tensor ConventionalIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor ConventionalIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor ConventionalIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

void ConventionalIntegrals::set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2)
{
    double* p_tei;
    if (alpha1 == true and alpha2 == true) p_tei = aphys_tei_aa;
    if (alpha1 == true and alpha2 == false) p_tei = aphys_tei_ab;
    if (alpha1 == false and alpha2 == false) p_tei = aphys_tei_bb;
    size_t index = aptei_index(p,q,r,s);
    p_tei[index] = value;
}

void ConventionalIntegrals::update_integrals(bool freeze_core)
{
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            aptei_idx_ = ncmo_;
        }
    }
}

void ConventionalIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    transform_integrals();
    gather_integrals();
    update_integrals();
}

void ConventionalIntegrals::gather_integrals()
{
    outfile->Printf("\n  Reading the two-electron integrals from disk");
    outfile->Printf("\n  Size of two-electron integrals: %10.6f GB", double(3 * 8 * num_aptei) / 1073741824.0);
    int_mem_ = sizeof(double) * 3 * 8 * num_aptei / 1073741824.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_aa[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_ab[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_bb[pqrs] = 0.0;

    int ioffmax = 30000;
    int* myioff = new int[ioffmax];
    myioff[0] = 0;
    for(size_t i = 1; i < ioffmax; ++i)
        myioff[i] = myioff[i-1] + i;

    if (restricted_){
        double* two_electron_integrals = new double[num_tei];
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pqrs = 0; pqrs < num_tei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

        // Read the integrals
        struct iwlbuf V_AAAA;
        iwl_buf_init(&V_AAAA,PSIF_MO_TEI, 0.0, 1, 1);
        iwl_buf_rd_all(&V_AAAA, two_electron_integrals, myioff, myioff, 0, myioff, 0, "outfile");
        iwl_buf_close(&V_AAAA, 1);

        // Store the integrals
        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct   = two_electron_integrals[INDEX4(p,r,q,s)];
                        double exchange = two_electron_integrals[INDEX4(p,s,q,r)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_aa[index] = direct - exchange;
                        aphys_tei_ab[index] = direct;
                        aphys_tei_bb[index] = direct - exchange;
                    }
                }
            }
        }

        // Deallocate temp memory
        delete[] two_electron_integrals;
    }else{
        double* two_electron_integrals = new double[num_tei];
        // Alpha-alpha integrals
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pqrs = 0; pqrs < num_tei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

        // Read the integrals
        struct iwlbuf V_AAAA;
        iwl_buf_init(&V_AAAA,PSIF_MO_AA_TEI, 0.0, 1, 1);
        iwl_buf_rd_all(&V_AAAA, two_electron_integrals, myioff, myioff, 0, myioff, 0, "outfile");
        iwl_buf_close(&V_AAAA, 1);

        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct   = two_electron_integrals[INDEX4(p,r,q,s)];
                        double exchange = two_electron_integrals[INDEX4(p,s,q,r)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_aa[index] = direct - exchange;
                    }
                }
            }
        }

        // Beta-beta integrals
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pqrs = 0; pqrs < num_tei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

        // Read the integrals
        struct iwlbuf V_BBBB;
        iwl_buf_init(&V_BBBB,PSIF_MO_BB_TEI, 0.0, 1, 1);
        iwl_buf_rd_all(&V_BBBB, two_electron_integrals, myioff, myioff, 0, myioff, 0, "outfile");
        iwl_buf_close(&V_BBBB, 1);

        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct   = two_electron_integrals[INDEX4(p,r,q,s)];
                        double exchange = two_electron_integrals[INDEX4(p,s,q,r)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_bb[index] = direct - exchange;
                    }
                }
            }
        }
        // Deallocate temp memory
        delete[] two_electron_integrals;

        // Alpha-beta integrals
        Matrix Tei(num_oei,num_oei);
        double** two_electron_integrals_ab = Tei.pointer();
        // Zero the memory, because iwl_buf_rd_all copies only the nonzero entries
        for (size_t pq = 0; pq < num_oei; ++pq){
            for (size_t rs = 0; rs < num_oei; ++rs){
                two_electron_integrals_ab[pq][rs] = 0.0;
            }
        }

        // Read the integrals
        struct iwlbuf V_AABB;
        iwl_buf_init(&V_AABB,PSIF_MO_AB_TEI, 0.0, 1, 1);
        iwl_buf_rd_all2(&V_AABB, two_electron_integrals_ab, myioff, myioff, 0, myioff, 0, "outfile");
        iwl_buf_close(&V_AABB, 1);

        for (size_t p = 0; p < nmo_; ++p){
            for (size_t q = 0; q < nmo_; ++q){
                for (size_t r = 0; r < nmo_; ++r){
                    for (size_t s = 0; s < nmo_; ++s){
                        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                        double direct = two_electron_integrals_ab[INDEX2(p,r)][INDEX2(q,s)];
                        size_t index = aptei_index(p,q,r,s);
                        aphys_tei_ab[index] = direct;
                    }
                }
            }
        }
    }
    delete[] myioff;
}

void ConventionalIntegrals::resort_integrals_after_freezing()
{
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = (cmo2mo);

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmo2mo);
    resort_two(one_electron_integrals_b,cmo2mo);
    resort_two(diagonal_aphys_tei_aa,cmo2mo);
    resort_two(diagonal_aphys_tei_ab,cmo2mo);
    resort_two(diagonal_aphys_tei_bb,cmo2mo);

    resort_four(aphys_tei_aa,cmo2mo);
    resort_four(aphys_tei_ab,cmo2mo);
    resort_four(aphys_tei_bb,cmo2mo);
}

void ConventionalIntegrals::resort_four(double*& tei, std::vector<size_t>& map)
{
    // Store the integrals in a temporary array
    double* temp_ints = new double[num_aptei];
    for (size_t p = 0; p < num_aptei; ++p){
        temp_ints[p] = 0.0;
    }
    for (size_t p = 0; p < ncmo_; ++p){
        for (size_t q = 0; q < ncmo_; ++q){
            for (size_t r = 0; r < ncmo_; ++r){
                for (size_t s = 0; s < ncmo_; ++s){
                    size_t pqrs_cmo = ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;
                    size_t pqrs_mo  = nmo_ * nmo_ * nmo_ * map[p] + nmo_ * nmo_ * map[q] + nmo_ * map[r] + map[s];
                    temp_ints[pqrs_cmo] = tei[pqrs_mo];
                }
            }
        }
    }
    // Delete old integrals and assign the pointer
    delete[] tei;
    tei = temp_ints;
}

void ConventionalIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

void ConventionalIntegrals::make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b)
{
    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = 0; q < ncmo_; ++q){
            fock_matrix_a[p * ncmo_ + q] = oei_a(p,q);
            fock_matrix_b[p * ncmo_ + q] = oei_b(p,q);
        }
    }
    double zero = 1.0e-15;
    for (int r = 0; r < ncmo_; ++r) {
        for (int s = 0; s < ncmo_; ++s) {
            double gamma_a_rs = gamma_a->get(r,s);
            if (std::fabs(gamma_a_rs) > zero){
                for(size_t p = 0; p < ncmo_; ++p){
                    for(size_t q = 0; q < ncmo_; ++q){
                        fock_matrix_a[p * ncmo_ + q] += aptei_aa(p,r,q,s) * gamma_a_rs;
                        fock_matrix_b[p * ncmo_ + q] += aptei_ab(r,p,s,q) * gamma_a_rs;
                    }
                }
            }
        }
    }
    for (int r = 0; r < ncmo_; ++r) {
        for (int s = 0; s < ncmo_; ++s) {
            double gamma_b_rs = gamma_b->get(r,s);
            if (std::fabs(gamma_b_rs) > zero){
                for(size_t p = 0; p < ncmo_; ++p){
                    for(size_t q = 0; q < ncmo_; ++q){
                        fock_matrix_a[p * ncmo_ + q] += aptei_ab(p,r,q,s) * gamma_b_rs;
                        fock_matrix_b[p * ncmo_ + q] += aptei_bb(p,r,q,s) * gamma_b_rs;
                    }
                }
            }
        }
    }
}

void ConventionalIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
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

void ConventionalIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
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

void ConventionalIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
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

void ConventionalIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
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

void ConventionalIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
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

void ConventionalIntegrals::freeze_core_orbitals()
{
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
}

void ConventionalIntegrals::compute_frozen_core_energy()
{
    frozen_core_energy_ = 0.0;
    double core_print = 0.0;

    for (int hi = 0, p = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            frozen_core_energy_ += oei_a(p + i,p + i) + oei_b(p + i,p + i);
            for (int hj = 0, q = 0; hj < nirrep_; ++hj){
                for (int j = 0; j < frzcpi_[hj]; ++j){
                    frozen_core_energy_ += 0.5 * diag_aptei_aa(p + i,q + j) + 0.5 * diag_aptei_bb(p + i,q + j) + diag_aptei_ab(p + i,q + j);
                }
                q += nmopi_[hj]; // orbital offset for the irrep hj
            }
        }
        p += nmopi_[hi]; // orbital offset for the irrep hi
    }
    outfile->Printf("\n  Frozen-core energy        %20.12f a.u.",frozen_core_energy_);
}

void ConventionalIntegrals::compute_frozen_one_body_operator()
{
    int nthread = 1;
    size_t f = 0;
    for (int hi = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            size_t r = f + i;
            outfile->Printf("\n  Freezing MO %zu",r);
            #pragma omp parallel for num_threads(num_threads_)\
            schedule(dynamic) 
            for(size_t p = 0; p < nmo_; ++p){
                for(size_t q = 0; q < nmo_; ++q){
                    one_electron_integrals_a[p * nmo_ + q] += aptei_aa(r,p,r,q) + aptei_ab(r,p,r,q);
                    one_electron_integrals_b[p * nmo_ + q] += aptei_bb(r,p,r,q) + aptei_ab(r,p,r,q);
                }
            }
        }
        f += nmopi_[hi];
    }
}

DFIntegrals::~DFIntegrals()
{
    deallocate();
}

void DFIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

    //qt_pitzer_ = new int[nmo_];
}

double DFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);
     vpqrsalphaE = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + s][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + r][0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}

double DFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);

    return (vpqrsalphaC);

}

double DFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
        double vpqrsalphaC = 0.0;
        double vpqrsalphaE = 0.0;
        vpqrsalphaC = C_DDOT(nthree_,
                &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
                &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);
         vpqrsalphaE = C_DDOT(nthree_,
                &(ThreeIntegral_->pointer()[p*aptei_idx_ + s][0]),1,
                &(ThreeIntegral_->pointer()[q*aptei_idx_ + r][0]),1);

        return (vpqrsalphaC - vpqrsalphaE);

}

ambit::Tensor DFIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor DFIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor DFIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}
ambit::Tensor DFIntegrals::get_three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{A.size(), p.size(), q.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = get_three_integral(A[i[0]], p[i[1]], q[i[2]]);
    });
    return ReturnTensor;
}

void DFIntegrals::set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2)
{
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DFIntegrals::gather_integrals()
{
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    outfile->Printf("\n Computing Density fitted integrals \n");

    boost::shared_ptr<BasisSet> primary = wfn->basisset();
    boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(primary->molecule(), "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));

    size_t nprim = primary->nbf();
    size_t naux  = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n Need %8.6f GB to store DF integrals\n", (nprim * nprim * naux * sizeof(double)/1073741824.0));

    Dimension nsopi_ = wfn->nsopi();
    SharedMatrix aotoso = wfn->aotoso();
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));

    // Transform from the SO to the AO basis
    for (size_t h = 0, index = 0; h < nirrep_; ++h){
        for (size_t i = 0; i < nmopi_[h]; ++i){
            size_t nao = nso_;
            size_t nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca->pointer(h)[0][i],nmopi_[h],0.0,&Ca_ao->pointer()[0][index],nmopi_.sum());

            index += 1;
        }

    }


    //B_{pq}^Q -> MO without frozen core

    //Constructs the DF function
    //I used this version of build as this doesn't build all the apces and assume a RHF/UHF reference
    boost::shared_ptr<DFERI> df = DFERI::build(primary,auxiliary,options_);

    //Pushes a C matrix that is ordered in pitzer ordering
    //into the C_matrix object
//    df->set_C(C_ord);
    df->set_C(Ca_ao);
    //set_C clears all the orbital spaces, so this creates the space
    //This space creates the total nmo_.
    //This assumes that everything is correlated.
    df->add_space("ALL", 0, nmo_);
    //Does not add the pair_space, but says which one is should use
    df->add_pair_space("B", "ALL", "ALL");
    df->set_memory(Process::environment.get_memory()/8L);

    //Finally computes the df integrals
    //Does the timings also
    Timer timer;
    std::string str= "Computing DF Integrals";
    outfile->Printf("\n    %-36s ...", str.c_str());
    df->compute();
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    boost::shared_ptr<psi::Tensor> B = df->ints()["B"];
    df.reset();

    FILE* Bf = B->file_pointer();
    //SharedMatrix Bpq(new Matrix("Bpq", nmo_, nmo_ * naux));
    SharedMatrix Bpq(new Matrix("Bpq", nmo_ * nmo_, naux));

    //Reads the DF integrals into Bpq.  Stores them as nmo by (nmo*naux)

    std::string str_seek= "Seeking DF Integrals";
    outfile->Printf("\n    %-36s ...", str_seek.c_str());
    fseek(Bf,0L, SEEK_SET);
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    std::string str_read = "Reading DF Integrals";
    outfile->Printf("\n   %-36s . . .", str_read.c_str());
    fread(&(Bpq->pointer()[0][0]), sizeof(double),naux*(nmo_)*(nmo_), Bf);
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    //This has a different dimension than two_electron_integrals in the integral code that francesco wrote.
    //This is because francesco reads only the nonzero integrals
    //I store all of them into this array.

    // Store the integrals in the form of nmo*nmo by B
    //Makes a gemm call very easy
    //std::string re_sort = "Resorting DF Integrals";
    //outfile->Printf("\n   %-36s ...",re_sort.c_str());
    //for (size_t p = 0; p < nmo_; ++p){
    //    for (size_t q = 0; q < nmo_; ++q){
    //        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
    //        for(size_t B = 0; B < naux; B++){
    //            size_t qB = q * naux + B;
    //            tBpq->set(B,p*nmo_ + q, Bpq->get(p,qB));
    //        }
    //     }
    //}
    outfile->Printf("...Done.  Timing %15.6f s", timer.get());

    ThreeIntegral_ = Bpq;
    //outfile->Printf("\n %8.8f integral", aptei_ab(10,8,5,2));

}

void DFIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

DFIntegrals::DFIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : ExplorerIntegrals(options, restricted, resort_frozen_core){
    integral_type_ = DF;

    outfile->Printf("\n DFIntegrals overall time");
    Timer DFInt;
    allocate();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n DFIntegrals take %15.8f s", DFInt.get());
}

void DFIntegrals::update_integrals(bool freeze_core)
{
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            aptei_idx_ = ncmo_;
        }
    }
}

void DFIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    //TODO:  Remove this function from retransform
    //For DF, reread integrals and then transfrom to new basis
    gather_integrals();
    update_integrals();
}

void DFIntegrals::deallocate()
{

    // Deallocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
    //delete[] qt_pitzer_;
}
void DFIntegrals::make_fock_matrix(SharedMatrix gamma_aM,SharedMatrix gamma_bM)
{
    TensorType tensor_type = kCore;
    ambit::Tensor ThreeIntegralTensor = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,ncmo_, ncmo_ });
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a",{ncmo_, ncmo_});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b",{ncmo_, ncmo_});
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a",{ncmo_, ncmo_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b",{ncmo_, ncmo_});

    //ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double& value){
    //    value = ThreeIntegral_->get(i[0],i[1]*aptei_idx_ + i[2]);
    //});
    std::vector<size_t> vQ(nthree_);
    std::iota(vQ.begin(), vQ.end(), 0);
    std::vector<size_t> vP(ncmo_);
    std::iota(vP.begin(), vP.end(), 0);


    ThreeIntegralTensor = get_three_integral_block(vQ, vP, vP);

    gamma_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(i[0],i[1]);
    });
    gamma_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(i[0],i[1]);
    });

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });

    ///Changing the Q_pr * Q_qs  to Q_rp * Q_sq for convience for reading

    //ambit::Tensor test = ambit::Tensor::build(tensor_type, "Fock_b",{nthree_});
    //test("Q") = ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    //fock_a("p,q") += ThreeIntegralTensor("Q,p,q")*test("Q");
    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    fock_a("p,q") -=  ThreeIntegralTensor("Q,r,p") * ThreeIntegralTensor("Q,s,q") * gamma_a("r,s");
    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");

    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");
    fock_b("p,q") -=  ThreeIntegralTensor("Q,r,p") * ThreeIntegralTensor("Q,s,q") * gamma_b("r,s");
    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");



    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });

}

void DFIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
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

void DFIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
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

void DFIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
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

void DFIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
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

void DFIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
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

void DFIntegrals::resort_three(SharedMatrix& threeint,std::vector<size_t>& map)
{
    //Create a temperature threeint matrix
    SharedMatrix temp_threeint(threeint->clone());
    temp_threeint->zero();

    // Borrwed from resort_four.
    // Since L is not sorted, only need to sort the columns
    // Surprisingly, this was pretty easy.
    for (size_t L = 0; L < nthree_; ++L){
        for (size_t q = 0; q < ncmo_; ++q){
            for (size_t r = 0; r < ncmo_; ++r){
                size_t Lpq_cmo  = q * ncmo_ + r;
                size_t Lpq_mo  = map[q] * nmo_ + map[r];
                temp_threeint->set(Lpq_cmo, L , threeint->get(Lpq_mo, L));

            }
        }
    }

    //This copies the resorted integrals and the data is changed to the sorted
    //matrix
    outfile->Printf("\n Done with resorting");
    threeint->copy(temp_threeint);
}

void DFIntegrals::freeze_core_orbitals()
{
    Timer freezeOrbs;
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
    outfile->Printf("\n Frozen Orbitals takes %8.8f s", freezeOrbs.get());
}

void DFIntegrals::compute_frozen_core_energy()
{
    Timer FrozenEnergy;
    frozen_core_energy_ = 0.0;

    for (int hi = 0, p = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            frozen_core_energy_ += oei_a(p + i,p + i) + oei_b(p + i,p + i);

            for (int hj = 0, q = 0; hj < nirrep_; ++hj){
                for (int j = 0; j < frzcpi_[hj]; ++j){
                    frozen_core_energy_ += 0.5 * diag_aptei_aa(p + i,q + j) + 0.5 * diag_aptei_bb(p + i,q + j) + diag_aptei_ab(p + i,q + j);
                }
                q += nmopi_[hj]; // orbital offset for the irrep hj
            }
        }
        p += nmopi_[hi]; // orbital offset for the irrep hi
    }
    outfile->Printf("\n  Frozen-core energy        %20.12f a.u.",frozen_core_energy_);
    outfile->Printf("\n\n Frozen_Core_Energy takes   %8.8f s", FrozenEnergy.get());
}

void DFIntegrals::compute_frozen_one_body_operator()
{
    Timer FrozenOneBody;
    boost::shared_ptr<BlockedTensorFactory>BTF(new BlockedTensorFactory(options_));
    ambit::BlockedTensor::reset_mo_spaces();

    size_t f = 0; // The Offset for irrep
    size_t r = 0; // The MO number for frozen core
    size_t g = 0; //the Offset for irrep
    std::vector<size_t> frozen_vec;
    std::vector<size_t> corrleated_vec;

    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            r = f + i;
            frozen_vec.push_back(r);
        }
        f += nmopi_[hi];
        for (size_t p = frzcpi_[hi]; p < nmopi_[hi]; p++)
        {
            size_t mo = p + g;
            corrleated_vec.push_back(mo);
        }
        g += nmopi_[hi];
    }
    //Get the size of frozen MO
    size_t frozen_size = frozen_vec.size();

    //Form a map that says mo_to_rel[ABS_MO] =relative in frozen array
    std::map<size_t, size_t>  mo_to_rel;
    std::vector<size_t> motofrozen(frozen_size);
    std::iota(motofrozen.begin(), motofrozen.end(), 0);

    int i = 0;
    for (auto frozen : frozen_vec)
    {
        mo_to_rel[frozen] = motofrozen[i];
        i++;
    }

    BTF->add_mo_space("f","rs", frozen_vec, NoSpin);
    BTF->add_mo_space("k", "mn", corrleated_vec, NoSpin);
    BTF->add_composite_mo_space("a", "pq", {"f", "k"});

    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    BTF->add_mo_space("d","g",nauxpi,NoSpin);

    //Kevin is lazy.  Going to use ambit to perform this contraction
    ambit::BlockedTensor ThreeIntegral = BTF->build(kCore,"ThreeInt",{"daa"});
    ambit::BlockedTensor FullFrozenV   = BTF->build(kCore, "FullFrozenV", {"ffaa"});
    ambit::BlockedTensor FullFrozenVAB   = BTF->build(kCore, "FullFrozenV", {"ffaa"});
    ambit::BlockedTensor Test   = BTF->build(kCore, "FullFrozenV", {"faa"});

    ThreeIntegral.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        value = ThreeIntegral_->get(i[1] * nmo_ + i[2], i[0]);
    });
    boost::shared_ptr<Matrix> FrozenVMatrix(new Matrix("FrozenV", frozen_size * frozen_size, nmo_ *  nmo_));
    boost::shared_ptr<Matrix> FrozenVMatrixAB(new Matrix("FrozenVAB", frozen_size * frozen_size, nmo_ * nmo_));
    boost::shared_ptr<Matrix> TestM(new Matrix("FrozenVAB", frozen_size * nmo_ , nmo_));

    FullFrozenV["rspq"] = ThreeIntegral["grs"]*ThreeIntegral["gpq"];
    //FullFrozenV["rspq"] -=ThreeIntegral["grq"]*ThreeIntegral["gps"];
    FullFrozenVAB["rspq"] = ThreeIntegral["grs"]*ThreeIntegral["gpq"];
    Test["rpq"] =ThreeIntegral["grq"]*ThreeIntegral["gpr"];
    Test.print(stdout);


    FullFrozenV.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        FrozenVMatrix->set(mo_to_rel[i[0]] * frozen_size + mo_to_rel[i[1]], i[2] * nmo_ + i[3], value);
    });
    FullFrozenVAB.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        FrozenVMatrixAB->set(mo_to_rel[i[0]] * frozen_size + mo_to_rel[i[1]], i[2] * nmo_ + i[3], value);
    });
    Test.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        TestM->set(mo_to_rel[i[0]] * nmo_ + i[1], i[2], value);
    });
    f = 0;

    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            size_t r = f + i;
            outfile->Printf("\n  Freezing MO %lu and %lu",r, mo_to_rel[r]);
            #pragma omp parallel for num_threads(num_threads_) \
            schedule(dynamic)
            for(size_t p = 0; p < nmo_; ++p){
                for(size_t q = 0; q < nmo_; ++q){
                    one_electron_integrals_a[p * nmo_ + q] += FrozenVMatrix->get(mo_to_rel[r] * frozen_size + mo_to_rel[r], p * nmo_ + q)
                            + FrozenVMatrixAB->get(mo_to_rel[r] * frozen_size + mo_to_rel[r], p * nmo_ + q) - TestM->get(mo_to_rel[r] * nmo_ + p,q);
                    one_electron_integrals_b[p * nmo_ + q] += FrozenVMatrix->get(mo_to_rel[r] * frozen_size +mo_to_rel[r], p * nmo_ + q)
                            + FrozenVMatrixAB->get(mo_to_rel[r] * frozen_size + mo_to_rel[r], p * nmo_ + q) - TestM->get(mo_to_rel[r] * nmo_ + p,q);
                }
            }
        }
        f += nmopi_[hi];
    }

    for(size_t p = 0; p < nmo_; p++)
        for(size_t q = 0; q < nmo_; q++)
            outfile->Printf("\n\n %d %d one_int = %8.8f", p, q, one_electron_integrals_a[p * nmo_ + q]);
    ambit::BlockedTensor::reset_mo_spaces();
    outfile->Printf("\n\n FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());

}

void DFIntegrals::resort_integrals_after_freezing()
{
    Timer resort_integrals;
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = (cmo2mo); 

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmo2mo);
    resort_two(one_electron_integrals_b,cmo2mo);
    resort_two(diagonal_aphys_tei_aa,cmo2mo);
    resort_two(diagonal_aphys_tei_ab,cmo2mo);
    resort_two(diagonal_aphys_tei_bb,cmo2mo);

    resort_three(ThreeIntegral_,cmo2mo);

    outfile->Printf("\n Resorting integrals takes   %8.8fs", resort_integrals.get());
}


CholeskyIntegrals::CholeskyIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : ExplorerIntegrals(options, restricted, resort_frozen_core){
    integral_type_ = Cholesky;
    outfile->Printf("\n Cholesky integrals time");
    Timer CholInt;
    allocate();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n CholeskyIntegrals take %8.8f", CholInt.get());
}

CholeskyIntegrals::~CholeskyIntegrals()
{
    deallocate();
}
void CholeskyIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

double CholeskyIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);
     vpqrsalphaE = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + s][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + r][0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}

double CholeskyIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1 ,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1 );
    return (vpqrsalphaC);
}

double CholeskyIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0, vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);
     vpqrsalphaE = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + s][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + r][0]),1);
    //}

    return (vpqrsalphaC - vpqrsalphaE);
}
ambit::Tensor CholeskyIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CholeskyIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CholeskyIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}
ambit::Tensor CholeskyIntegrals::get_three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{A.size(), p.size(), q.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = get_three_integral(A[i[0]], p[i[1]], q[i[2]]);
    });
    return ReturnTensor;
}

void CholeskyIntegrals::gather_integrals()
{
    outfile->Printf("\n Computing the Cholesky Vectors \n");


    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    boost::shared_ptr<BasisSet> primary = wfn->basisset();


    size_t nbf = primary->nbf();


    boost::shared_ptr<IntegralFactory> integral(new IntegralFactory(primary, primary, primary, primary));
    double tol_cd = options_.get_double("CHOLESKY_TOLERANCE");

    //This is creates the cholesky decomposed AO integrals
    Timer timer;
    std::string str= "Computing CD Integrals";
    outfile->Printf("\n    %-36s ...", str.c_str());
    boost::shared_ptr<CholeskyERI> Ch (new CholeskyERI(boost::shared_ptr<TwoBodyAOInt>(integral->eri()),0.0 ,tol_cd, Process::environment.get_memory()));
    //Computes the cholesky integrals
    Ch->choleskify();
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    //The number of vectors required to do cholesky factorization
    size_t nL = Ch->Q();
    nthree_ = nL;
    outfile->Printf("\n Need %8.6f GB to store cd integrals in core\n",nL * nbf * nbf * sizeof(double) / 1073741824.0 );
    int_mem_ = (nL * nbf * nbf * sizeof(double) / 1073741824.0);

    TensorType tensor_type = kCore;

    outfile->Printf("\n Number of cholesky vectors %d to satisfy %20.12f tolerance\n", nL,tol_cd);
    SharedMatrix Lao = Ch->L();
    SharedMatrix L(new Matrix("Lmo", nL, (nmo_)*(nmo_)));
    SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix aotoso = wfn->aotoso();

    // Transform from the SO to the AO basis
    Dimension nsopi_ = wfn->nsopi();
    for (int h = 0, index = 0; h < nirrep_; ++h){
        for (int i = 0; i < nmopi_[h]; ++i){
            int nao = nso_;
            int nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca->pointer(h)[0][i],nmopi_[h],0.0,&Ca_ao->pointer()[0][index],nmopi_.sum());

            index += 1;
        }

    }

    ambit::Tensor ThreeIntegral_ao = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,nmo_, nmo_ });
    ambit::Tensor Cpq_tensor = ambit::Tensor::build(tensor_type,"C_sorted",{nbf,nmo_});
    ambit::Tensor ThreeIntegral = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,nmo_, nmo_ });

    Cpq_tensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = Ca_ao->get(i[0],i[1]);
    });
    ThreeIntegral_ao.iterate([&](const std::vector<size_t>& i,double& value){
        value = Lao->get(i[0],i[1]*nbf + i[2]);
    });
    SharedMatrix ThreeInt(new Matrix("Lmo", (nmo_)*(nmo_), nL));
    ThreeIntegral_ =ThreeInt;


    ThreeIntegral("L,p,q") = ThreeIntegral_ao("L,m,n")*Cpq_tensor("m,p")*Cpq_tensor("n,q");

    ThreeIntegral.iterate([&](const std::vector<size_t>& i,double& value){
        ThreeIntegral_->set(i[1]*nmo_ + i[2],i[0],value);
    });
}

void CholeskyIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals

    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

}

void CholeskyIntegrals::update_integrals(bool freeze_core)
{
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            aptei_idx_ = ncmo_;
        }
    }
}

void CholeskyIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    gather_integrals();
    update_integrals();
}

void CholeskyIntegrals::deallocate()
{

    // Deallocate the memory required to store the one-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;

    //delete[] qt_pitzer_;
}

void CholeskyIntegrals::make_fock_matrix(SharedMatrix gamma_aM,SharedMatrix gamma_bM)
{
    TensorType tensor_type = kCore;
    ambit::Tensor ThreeIntegralTensor = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,ncmo_, ncmo_ });
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a",{ncmo_, ncmo_});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b",{ncmo_, ncmo_});
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a",{ncmo_, ncmo_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b",{ncmo_, ncmo_});

    ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = ThreeIntegral_->get(i[1]*aptei_idx_ + i[2], i[0]);
    });
    gamma_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(i[0],i[1]);
    });
    gamma_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(i[0],i[1]);
    });

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });


    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    fock_a("p,q") -= ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_a("r,s");
    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");

    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");
    fock_b("p,q") -= ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_b("r,s");
    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });
}

void CholeskyIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
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

void CholeskyIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
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

void CholeskyIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
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

void CholeskyIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
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

void CholeskyIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
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

void CholeskyIntegrals::resort_integrals_after_freezing()
{
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = (cmo2mo);

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmo2mo);
    resort_two(one_electron_integrals_b,cmo2mo);
    resort_two(diagonal_aphys_tei_aa,cmo2mo);
    resort_two(diagonal_aphys_tei_ab,cmo2mo);
    resort_two(diagonal_aphys_tei_bb,cmo2mo);

    resort_three(ThreeIntegral_,cmo2mo);

}
void CholeskyIntegrals::resort_three(boost::shared_ptr<Matrix>& threeint,std::vector<size_t>& map)
{
    //Create a temperature threeint matrix
    SharedMatrix temp_threeint(threeint->clone());
    temp_threeint->zero();

    // Borrwed from resort_four.
    // Since L is not sorted, only need to sort the columns
    // Surprisingly, this was pretty easy.
    for (size_t L = 0; L < nthree_; ++L){
        for (size_t q = 0; q < ncmo_; ++q){
            for (size_t r = 0; r < ncmo_; ++r){
                size_t Lpq_cmo  = q * ncmo_ + r;
                size_t Lpq_mo  = map[q] * nmo_ + map[r];
                temp_threeint->set(Lpq_cmo, L, threeint->get(Lpq_mo, L));

            }
        }
    }

    //This copies the resorted integrals and the data is changed to the sorted
    //matrix
    threeint->copy(temp_threeint);
}

void CholeskyIntegrals::set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1, bool alpha2)
{
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void CholeskyIntegrals::freeze_core_orbitals()
{
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
}

void CholeskyIntegrals::compute_frozen_core_energy()
{
    frozen_core_energy_ = 0.0;
    double core_print = 0.0;

    for (int hi = 0, p = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            frozen_core_energy_ += oei_a(p + i,p + i) + oei_b(p + i,p + i);

            for (int hj = 0, q = 0; hj < nirrep_; ++hj){
                for (int j = 0; j < frzcpi_[hj]; ++j){
                    frozen_core_energy_ += 0.5 * diag_aptei_aa(p + i,q + j) + 0.5 * diag_aptei_bb(p + i,q + j) + diag_aptei_ab(p + i,q + j);
                }
                q += nmopi_[hj]; // orbital offset for the irrep hj
            }
        }
        p += nmopi_[hi]; // orbital offset for the irrep hi
    }
    outfile->Printf("\n  Frozen-core energy        %20.12f a.u.",frozen_core_energy_);
}

void CholeskyIntegrals::compute_frozen_one_body_operator()
{
    Timer FrozenOneBody;
    boost::shared_ptr<BlockedTensorFactory>BTF(new BlockedTensorFactory(options_));

    size_t f = 0; // The Offset for irrep
    size_t r = 0; // The MO number for frozen core
    size_t g = 0; //the Offset for irrep
    std::vector<size_t> frozen_vec;
    std::vector<size_t> corrleated_vec;

    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            r = f + i;
            frozen_vec.push_back(r);
        }
        f += nmopi_[hi];
        for (size_t p = frzcpi_[hi]; p < nmopi_[hi]; p++)
        {
            size_t mo = p + g;
            corrleated_vec.push_back(mo);
        }
        g += nmopi_[hi];
    }
    //Get the size of frozen MO
    size_t frozen_size = frozen_vec.size();

    //Form a map that says mo_to_rel[ABS_MO] =relative in frozen array
    std::map<size_t, size_t>  mo_to_rel;
    std::vector<size_t> motofrozen(frozen_size);
    std::iota(motofrozen.begin(), motofrozen.end(), 0);

    int i = 0;
    for (auto frozen : frozen_vec)
    {
        mo_to_rel[frozen] = motofrozen[i];
        i++;
    }

    BTF->add_mo_space("f","rs", frozen_vec, NoSpin);
    BTF->add_mo_space("k", "mn", corrleated_vec, NoSpin);
    BTF->add_composite_mo_space("a", "pq", {"f", "k"});

    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    BTF->add_mo_space("d","g",nauxpi,NoSpin);

    //Kevin is lazy.  Going to use ambit to perform this contraction
    ambit::BlockedTensor ThreeIntegral = BTF->build(kCore,"ThreeInt",{"daa"});
    ambit::BlockedTensor FullFrozenV   = BTF->build(kCore, "FullFrozenV", {"ffaa"});
    ambit::BlockedTensor FullFrozenVAB   = BTF->build(kCore, "FullFrozenV", {"ffaa"});

    ThreeIntegral.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        value = get_three_integral(i[0], i[1], i[2]);
    });
    boost::shared_ptr<Matrix> FrozenVMatrix(new Matrix("FrozenV", frozen_size * frozen_size, nmo_ *  nmo_));
    boost::shared_ptr<Matrix> FrozenVMatrixAB(new Matrix("FrozenVAB", frozen_size * frozen_size, nmo_ * nmo_));

    FullFrozenV["rspq"] = ThreeIntegral["grs"]*ThreeIntegral["gpq"];
    FullFrozenV["rspq"] -=ThreeIntegral["grq"]*ThreeIntegral["gps"];
    FullFrozenVAB["rspq"] = ThreeIntegral["grs"]*ThreeIntegral["gpq"];


    //Fill the SharedMatrix as frozen^2 by nmo^2
    //mo_to_rel[i[0]] gives the relative index in the first row ie orbital 28th which is frozen is the 3rd frozen orbital
    FullFrozenV.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        FrozenVMatrix->set(mo_to_rel[i[0]] * frozen_size + mo_to_rel[i[1]], i[2] * nmo_ + i[3], value);
    });
    FullFrozenVAB.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        FrozenVMatrixAB->set(mo_to_rel[i[0]] * frozen_size + mo_to_rel[i[1]], i[2] * nmo_ + i[3], value);
    });
    f = 0;

    //Tried to do this with ambit, but it seems that this is not really a contraction
    //Ambit complains and so I revert back to origin way, but I contract out the auxiliary index
    //This could become a problem for large systems
    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            size_t r = f + i;
            outfile->Printf("\n  Freezing MO %lu",r);
            #pragma omp parallel for num_threads(num_threads_) \
            schedule(dynamic)
            for(size_t p = 0; p < nmo_; ++p){
                for(size_t q = 0; q < nmo_; ++q){
                    one_electron_integrals_a[p * nmo_ + q] += FrozenVMatrix->get(mo_to_rel[r] * frozen_size + mo_to_rel[r], p * nmo_ + q)
                            + FrozenVMatrixAB->get(mo_to_rel[r] * frozen_size + mo_to_rel[r], p * nmo_ + q);
                    one_electron_integrals_b[p * nmo_ + q] += FrozenVMatrix->get(mo_to_rel[r] * frozen_size +mo_to_rel[r], p * nmo_ + q)
                            + FrozenVMatrixAB->get(mo_to_rel[r] * frozen_size + mo_to_rel[r], p * nmo_ + q);
                }
            }
        }
        f += nmopi_[hi];
    }

    ambit::BlockedTensor::reset_mo_spaces();
    outfile->Printf("\n\n FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());
}
DISKDFIntegrals::~DISKDFIntegrals()
{
    deallocate();
}

void DISKDFIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

    //qt_pitzer_ = new int[nmo_];
}

double DISKDFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    size_t pn, qn, rn, sn;

    if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
    {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    }
    else
    {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }
    
    size_t offset1 = rn * nthree_ + pn * (nthree_ * nmo_);
    size_t offset2 = sn * nthree_ + qn * (nthree_ * nmo_);
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    SharedVector B1(new Vector("B1", nthree_));
    SharedVector B2(new Vector("B2", nthree_));


    // Read a block of Vectors for the Columb term
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());

    vpqrsalphaC = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);


     B1->zero();
     B2->zero();
     offset1 = 0;
     offset2 = 0;

    offset1 = sn * nthree_ + pn * (nthree_ * nmo_);
    offset2 = rn * nthree_ + qn * (nthree_ * nmo_);

    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
     vpqrsalphaE = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}

double DISKDFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
 
   size_t pn, qn, rn, sn;
   if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
   {
       pn = cmotomo_[p];
       qn = cmotomo_[q];
       rn = cmotomo_[r];
       sn = cmotomo_[s];
   }
   else
   {
       pn = p;
       qn = q;
       rn = r;
       sn = s;
   }

   size_t offset1 = rn * nthree_ + pn * (nthree_ * nmo_);
   size_t offset2 = sn * nthree_ + qn * (nthree_ * nmo_);
   double vpqrsalphaC = 0.0;

   SharedVector B1(new Vector("B1", nthree_));
   SharedVector B2(new Vector("B2", nthree_));

   // Read a block of Vectors for the Columb term
   fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
   fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
   fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
   fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());

   vpqrsalphaC = C_DDOT(nthree_,
                        &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

   return (vpqrsalphaC);

}

double DISKDFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
    size_t pn, qn, rn, sn;

    if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
    {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    }
    else
    {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }

    size_t offset1 = rn * nthree_ + pn * (nthree_ * nmo_);
    size_t offset2 = sn * nthree_ + qn * (nthree_ * nmo_);
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    SharedVector B1(new Vector("B1", nthree_));
    SharedVector B2(new Vector("B2", nthree_));

    // Read a block of Vectors for the Columb term
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());

    vpqrsalphaC = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

     B1->zero();
     B2->zero();
     offset1 = 0;
     offset2 = 0;

    offset1 = sn * nthree_ + pn * (nthree_ * nmo_);
    offset2 = rn * nthree_ + qn * (nthree_ * nmo_);
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
     vpqrsalphaE = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}
ambit::Tensor DISKDFIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)

{
    ambit::Tensor ThreeIntpr = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = get_three_integral_block(Avec, p, r);
    ThreeIntqs = get_three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor ("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");
    ReturnTensor ("p,q,r,s") -= ThreeIntpr("A,p,s") * ThreeIntqs("A,q,r");
    
    return ReturnTensor;
}

ambit::Tensor DISKDFIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ThreeIntpr = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = get_three_integral_block(Avec, p, r);
    ThreeIntqs = get_three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor ("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");

    return ReturnTensor;
}

ambit::Tensor DISKDFIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ThreeIntpr = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = get_three_integral_block(Avec, p, r);
    ThreeIntqs = get_three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor ("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");
    ReturnTensor ("p,q,r,s") -= ThreeIntpr("A,p,s") * ThreeIntqs("A,q,r");
    return ReturnTensor;
}
double DISKDFIntegrals::get_three_integral(size_t A, size_t p, size_t q)
{
    size_t pn, qn;
    if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
    {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
    }
    else
    {
        pn = p;
        qn = q;
    }


    size_t offset1 = pn * (nthree_ * nmo_) + qn * nthree_ + A;
    double value = 0.0;
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&value, sizeof(double), 1, B_->file_pointer());
    return value;

}
ambit::Tensor DISKDFIntegrals::get_three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q)
{
    //Since file is formatted as p by A * q
    bool frozen_core = false;

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{A.size(), p.size(), q.size()});
    if(frzcpi_.sum() && aptei_idx_==ncmo_)
    {
        frozen_core = true;
    }

    size_t pn, qn;
    if(nthree_ == A.size())
    {
        std::vector<boost::shared_ptr<Matrix> > p_by_Aq;
        for (auto p_block : p)
        {
            if(frozen_core)
            {
                pn = cmotomo_[p_block];
            }
            else
            {
                pn = p_block;
            }

            boost::shared_ptr<Matrix> Aq(new Matrix("Aq", nmo_, nthree_));

            fseek(B_->file_pointer(), pn*nthree_*nmo_*sizeof(double), SEEK_SET);
            fread(&(Aq->pointer()[0][0]), sizeof(double), nmo_ * nthree_, B_->file_pointer());
            p_by_Aq.push_back(Aq);


        }
        if(frozen_core)
        {
            ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
                value = p_by_Aq[i[1]]->get(cmotomo_[q[i[2]]], A[i[0]]);
            });
        }
        else
        {
            ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
                value = p_by_Aq[i[1]]->get(q[i[2]], A[i[0]]);
            });

        }
    }
    else
    {
        std::vector<double>& ReturnTensorV = ReturnTensor.data();
        //If user wants blocking in A
        pn = 0;
        qn = 0;
        for(size_t p_block : p)
        {
            pn = frozen_core ? cmotomo_[p_block] : p_block;
            for(size_t q_block : q)
            {
                qn = frozen_core ? cmotomo_[q_block] : q_block;

                double* A_chunk = new double[A.size()];
                size_t offset = pn * nthree_ * nmo_ + qn * nthree_ + A[0];
                fseek(B_->file_pointer(), offset * sizeof(double), SEEK_SET);
                fread(&(A_chunk[0]), sizeof(double), A.size(), B_->file_pointer());

                for(size_t a = 0; a < A.size(); a++)
                {
                    //Weird way the tensor is formatted
                    //Fill the tensor for every chunk of A
                    ReturnTensorV[a * ncmo_ * ncmo_ + p_block * ncmo_ + q_block] = A_chunk[a];

                }
                delete[] A_chunk;

            }
        }
    }
    return ReturnTensor;
}

void DISKDFIntegrals::set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2)
{
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DISKDFIntegrals::gather_integrals()
{
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    outfile->Printf("\n Computing Density fitted integrals \n");

    boost::shared_ptr<BasisSet> primary = wfn->basisset();
    boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(primary->molecule(), "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));

    size_t nprim = primary->nbf();
    size_t naux  = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n Need %8.6f GB to store DF integrals\n", (nprim * nprim * naux * sizeof(double)/1073741824.0));
    int_mem_ = (nprim * nprim * naux * sizeof(double));

    Dimension nsopi_ = wfn->nsopi();
    SharedMatrix aotoso = wfn->aotoso();
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));

    // Transform from the SO to the AO basis
    for (size_t h = 0, index = 0; h < nirrep_; ++h){
        for (size_t i = 0; i < nmopi_[h]; ++i){
            size_t nao = nso_;
            size_t nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca->pointer(h)[0][i],nmopi_[h],0.0,&Ca_ao->pointer()[0][index],nmopi_.sum());

            index += 1;
        }

    }


    //B_{pq}^Q -> MO without frozen core

    //Constructs the DF function
    //I used this version of build as this doesn't build all the apces and assume a RHF/UHF reference
    boost::shared_ptr<DFERI> df = DFERI::build(primary,auxiliary,options_);

    //Pushes a C matrix that is ordered in pitzer ordering
    //into the C_matrix object
//    df->set_C(C_ord);
    df->set_C(Ca_ao);
    //set_C clears all the orbital spaces, so this creates the space
    //This space creates the total nmo_.
    //This assumes that everything is correlated.
    df->add_space("ALL", 0, nmo_);
    //Does not add the pair_space, but says which one is should use
    df->add_pair_space("B", "ALL", "ALL");
    df->set_memory(Process::environment.get_memory()/8L);

    //Finally computes the df integrals
    //Does the timings also
    Timer timer;
    std::string str= "Computing DF Integrals";
    outfile->Printf("\n    %-36s ...", str.c_str());
    df->print_header();
    df->compute();
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    boost::shared_ptr<Tensor> B = df->ints()["B"];
    B_ = B;
    df.reset();

    //outfile->Printf("\n %8.8f integral", aptei_ab(10,8,5,2));

}

void DISKDFIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

DISKDFIntegrals::DISKDFIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : ExplorerIntegrals(options, restricted, resort_frozen_core){

    integral_type_ = DiskDF;
    outfile->Printf("\n DISKDFIntegrals overall time");
    Timer DFInt;
    allocate();

    //Form a correlated mo to mo before I create integrals
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = cmo2mo;

    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }

    outfile->Printf("\n DISKDFIntegrals take %15.8f s", DFInt.get());
}

void DISKDFIntegrals::update_integrals(bool freeze_core)
{
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            aptei_idx_ = ncmo_;
        }
    }
}

void DISKDFIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    //TODO:  Remove this function from retransform
    //For DF, reread integrals and then transfrom to new basis
    gather_integrals();
    update_integrals();
}

void DISKDFIntegrals::deallocate()
{

    // Deallocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
    //delete[] qt_pitzer_;
}
void DISKDFIntegrals::make_fock_matrix(SharedMatrix gamma_aM,SharedMatrix gamma_bM)
{
    //Efficient calculation of fock matrix from disk
    //Since gamma_aM is very sparse (diagonal elements of core and active block)
    //Only nonzero contributions are on diagonal elements
    //Grab the nonzero elements and put to a vector

    TensorType tensor_type = kCore;

    //Create the fock_a and fock_b globally
    //Choose to block over naux rather than ncmo_
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a",{ncmo_, ncmo_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b",{ncmo_, ncmo_});

    std::vector<size_t> nonzero;
    //Figure out exactly what I need to contract the Coloumb term
    for(int i = 0; i < ncmo_; i++)
    {
        if(gamma_aM->get(i,i) > 1e-10)
        {
            nonzero.push_back(i);
        }
    }

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });


    std::vector<size_t> A(nthree_);
    std::iota(A.begin(), A.end(), 0);

    std::vector<size_t> P(ncmo_);
    std::iota(P.begin(), P.end(), 0);

    //Create a gamma that contains only nonzero terms
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a",{nonzero.size(), nonzero.size()});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b",{nonzero.size(), nonzero.size()});
    //Create the full gamma (K is not nearly as sparse as J)
    ambit::Tensor gamma_a_full = ambit::Tensor::build(tensor_type, "Gamma_a",{ncmo_, ncmo_});
    ambit::Tensor gamma_b_full = ambit::Tensor::build(tensor_type, "Gamma_b",{ncmo_, ncmo_});

    gamma_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(nonzero[i[0]],nonzero[i[1]]);
    });
    gamma_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(nonzero[i[0]],nonzero[i[1]]);
    });
    ambit::Tensor ThreeIntC2 = ambit::Tensor::build(tensor_type, "ThreeInkC", {nthree_,nonzero.size(), nonzero.size()});
    ThreeIntC2 = get_three_integral_block(A, nonzero, nonzero);

    ambit::Tensor BQA = ambit::Tensor::build(tensor_type, "BQ", {nthree_});
    ambit::Tensor BQB = ambit::Tensor::build(tensor_type, "BQ", {nthree_});
    //Do a contraction over Naux * n_h^2 * n_h^2 -> naux * n_h^2
    BQA("B") = ThreeIntC2("B,r,s") * gamma_a("r,s");
    BQB("B") = ThreeIntC2("B,r,s") * gamma_b("r,s");
    //Grab the data from this for the block iteration
    std::vector<double>& BQAv = BQA.data();
    std::vector<double>& BQBv = BQB.data();

    gamma_a_full.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(i[0],i[1]);
    });
    gamma_b_full.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(i[0],i[1]);
    });

    //====Blocking information==========
    int int_mem_int_ = (nthree_ * ncmo_ * ncmo_) * sizeof(double);
    int memory_input = Process::environment.get_memory();
    int num_block = std::ceil(double(int_mem_int_) / memory_input);
    //Hard wires num_block for testing

    int block_size = nthree_ / num_block;

    if(num_block != 1)
    {
        outfile->Printf("\n\n\n\n\t---------Blocking Information-------\n\n\n\n\t");
    outfile->Printf("\n  %d / %d = %d", int_mem_int_, memory_input, int_mem_int_ / memory_input);
    outfile->Printf("\n  Block_size = %d\n num_block = %d", block_size, num_block);
    }

    Timer block_read;
    for(int i = 0; i < num_block; i++)
    {
        std::vector<size_t> A_block;
        if(nthree_ % num_block == 0)
        {
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * block_size);
        }
        else
        {
            block_size = i==(num_block - 1) ? block_size + nthree_ % num_block : block_size;
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * (nthree_ / num_block));
        }

        //Create a tensor of TI("Q,r,p")
        ambit::Tensor BQA_small = ambit::Tensor::build(tensor_type, "BQ", {A_block.size()});
        ambit::Tensor BQB_small = ambit::Tensor::build(tensor_type, "BQ", {A_block.size()});

       //Calculate the smaller block of A from the global block of prior Brs * gamma_rs
       BQA_small.iterate([&](const std::vector<size_t>& i,double& value){
            value = BQAv[A_block[i[0]]];
        });
       BQB_small.iterate([&](const std::vector<size_t>& i,double& value){
            value = BQBv[A_block[i[0]]];
        });
        ambit::Tensor ThreeIntegralTensor = ambit::Tensor::build(tensor_type,"ThreeIndex",{A_block.size(),ncmo_, ncmo_});

        //ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double& value){
        //    value = get_three_integral(A_block[i[0]], i[1], i[2]);
        //});

        //Return a tensor of ThreeInt given the smaller block of A
        ThreeIntegralTensor = get_three_integral_block(A_block, P, P);


        //Need to rewrite this to at least read in chunks of nthree_
        //ThreeIntegralTensor = get_three_integral_block(A_block, P,P );

        fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQA_small("Q");
        fock_a("p,q") -=  ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_a_full("r,s");
        fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQB_small("Q");

        fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQB_small("Q");
        fock_b("p,q") -=  ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_b_full("r,s");
        fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQA_small("Q");

        A_block.clear();
    }
    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });

    if(num_block!=1)
    {
        outfile->Printf("\n Created Fock matrix %8.8f s", block_read.get());
    }

}

void DISKDFIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
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

void DISKDFIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
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

void DISKDFIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
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

void DISKDFIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
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

void DISKDFIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
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

void DISKDFIntegrals::resort_three(SharedMatrix& threeint,std::vector<size_t>& map)
{
    outfile->Printf("No need to resort a file.  dummy!");
}

void DISKDFIntegrals::freeze_core_orbitals()
{
    Timer freezeOrbs;
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
    outfile->Printf("\n Frozen Orbitals takes %8.8f s", freezeOrbs.get());
}

void DISKDFIntegrals::compute_frozen_core_energy()
{
    Timer FrozenEnergy;
    frozen_core_energy_ = 0.0;

    for (int hi = 0, p = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            frozen_core_energy_ += oei_a(p + i,p + i) + oei_b(p + i,p + i);

            for (int hj = 0, q = 0; hj < nirrep_; ++hj){
                for (int j = 0; j < frzcpi_[hj]; ++j){
                    frozen_core_energy_ += 0.5 * diag_aptei_aa(p + i,q + j) + 0.5 * diag_aptei_bb(p + i,q + j) + diag_aptei_ab(p + i,q + j);
                }
                q += nmopi_[hj]; // orbital offset for the irrep hj
            }
        }
        p += nmopi_[hi]; // orbital offset for the irrep hi
    }
    outfile->Printf("\n  Frozen-core energy        %20.12f a.u.",frozen_core_energy_);
    outfile->Printf("\n\n Frozen_Core_Energy takes   %8.8f s", FrozenEnergy.get());
}

void DISKDFIntegrals::compute_frozen_one_body_operator()
{
    Timer FrozenOneBody;
    boost::shared_ptr<BlockedTensorFactory>BTF(new BlockedTensorFactory(options_));
    ambit::BlockedTensor::reset_mo_spaces();

    size_t f = 0; // The Offset for irrep
    size_t r = 0; // The MO number for frozen core
    size_t g = 0; //the Offset for irrep
    std::vector<size_t> frozen_vec;
    std::vector<size_t> corrleated_vec;

    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            r = f + i;
            frozen_vec.push_back(r);
        }
        f += nmopi_[hi];
        for (size_t p = frzcpi_[hi]; p < nmopi_[hi]; p++)
        {
            size_t mo = p + g;
            corrleated_vec.push_back(mo);
        }
        g += nmopi_[hi];
    }
    //Get the size of frozen MO
    size_t frozen_size = frozen_vec.size();

    //Form a map that says mo_to_rel[ABS_MO] =relative in frozen array
    std::map<size_t, size_t>  mo_to_rel;
    std::vector<size_t> motofrozen(frozen_size);
    std::iota(motofrozen.begin(), motofrozen.end(), 0);

    int i = 0;
    for (auto frozen : frozen_vec)
    {
        mo_to_rel[frozen] = motofrozen[i];
        i++;
    }

    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    std::vector<size_t> P(nmo_);
    std::iota(P.begin(), P.end(), 0);

    // <rp || rq> + <rp | rq>
    // = B_{rr}^{Q} * B_{pq}^{Q} - B_{rp}^{Q} B_{qr}^{Q} + B_{rr}^{Q} * B_{pq}^{Q}
    // => Assume that I can store all but B_{pq}^{Q} in core (maybe a big assumption . . . . . )
    //Kevin is lazy.  Going to use ambit to perform this contraction
    //Create three tensors.  Forgive my terrible naming
    //B_{rr}^{Q}
    ambit::Tensor BQr = ambit::Tensor::build(tensor_type_, "BQr", {nthree_, frozen_vec.size(), frozen_vec.size()});
    // B_{rp}^{Q}
    ambit::Tensor BQrp = ambit::Tensor::build(tensor_type_, "BQrp", {nthree_, frozen_vec.size(), nmo_});
    // B_{pr}^{Q}
    ambit::Tensor BQpr = ambit::Tensor::build(tensor_type_, "BQrp", {nthree_, nmo_,frozen_vec.size()});

    //Read these into a tensor
    BQrp = get_three_integral_block(nauxpi, frozen_vec, P);
    BQpr = get_three_integral_block(nauxpi, P, frozen_vec);

    ambit::Tensor rpq = ambit::Tensor::build(tensor_type_, "rpq", {frozen_vec.size(), nmo_, nmo_});
    ambit::Tensor rpqK = ambit::Tensor::build(tensor_type_, "rpqK", {frozen_vec.size(), nmo_, nmo_});

    //Form the exchange part of this out of loop for blocks
    //Need to see if this ever gets too large
    rpqK("r,p,q") = BQrp("Q,r,p") * BQpr("Q,q,r");

    //====Blocking information==========
    //Hope this is smart enough to figure out not to store B_{pq}^{Q}
    //Major problem with this is that other tensors are not that small.
    //Maybe won't work
    int int_mem_int_ = (nthree_ * ncmo_ * ncmo_) * sizeof(double);
    int memory_input = Process::environment.get_memory();
    int num_block = std::ceil(double(int_mem_int_) / memory_input);
    //Hard wires num_block for testing

    int block_size = nthree_ / num_block;

    if(num_block != 1)
    {
        outfile->Printf("\n\n\n\n\t---------Blocking Information-------\n\n\n\n\t");
        outfile->Printf("\n  %d / %d = %d", int_mem_int_, memory_input, int_mem_int_ / memory_input);
        outfile->Printf("\n  Block_size = %d\n num_block = %d", block_size, num_block);
    }

    for(int i = 0; i < num_block; i++)
    {
        std::vector<size_t> A_block;
        if(nthree_ % num_block == 0)
        {
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * block_size);
        }
        else
        {
            block_size = i==(num_block - 1) ? block_size + nthree_ % num_block : block_size;
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * (nthree_ / num_block));
        }
        //This tensor is extremely large for big systems
        //Needs to be blocked over A
        ambit::Tensor BQpq = ambit::Tensor::build(tensor_type_, "BQpq", {A_block.size(), nmo_, nmo_});
        BQpq = get_three_integral_block(A_block, P, P);
        //A_block is the split of the naux -> (0, . . . ,block_size)

        ambit::Tensor Qr = ambit::Tensor::build(tensor_type_, "Qr", {A_block.size(),frozen_vec.size()});
        BQr = get_three_integral_block(A_block, frozen_vec, frozen_vec);
        //Go from B_{rr}^{Q} -> B_{r}^{Q}.  Tensor library can not do this.
        std::vector<double>& BQrP = BQr.data(); // get the data from BQr
        Qr.iterate([&](const std::vector<size_t>& i,double& value){
            value = BQrP[i[0] * frozen_size * frozen_size + i[1] * frozen_size + i[1]] ;
        });
        //^^^^^ -> B_{rr}^{Q} -> B_{r}^{Q}
        rpq("r,p,q") += 2.0 * Qr("Q,r") * BQpq("Q,p,q");
    }
    std::vector<double>& rpqP = rpq.data();
    std::vector<double>& rpqKP = rpqK.data();
    f = 0;
    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            size_t r = f + i;
            outfile->Printf("\n  Freezing MO %lu", r);
            #pragma omp parallel for num_threads(num_threads_) \
            schedule(dynamic)
            for(size_t p = 0; p < nmo_; ++p){
                for(size_t q = 0; q < nmo_; ++q){
                    one_electron_integrals_a[p * nmo_ + q] += rpqP[mo_to_rel[r] * nmo_ * nmo_ + p * nmo_ +  q]
                            - rpqKP[mo_to_rel[r] * nmo_ * nmo_ + p * nmo_ + q];
                    one_electron_integrals_b[p * nmo_ + q] += rpqP[mo_to_rel[r] * nmo_ * nmo_ +p * nmo_ + q]
                            - rpqKP[mo_to_rel[r] * nmo_ * nmo_ + p * nmo_ + q];

                }
            }
        }
        f += nmopi_[hi];
    }
    ambit::BlockedTensor::reset_mo_spaces();

    outfile->Printf("\n\n FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());

}

void DISKDFIntegrals::resort_integrals_after_freezing()
{
    Timer resort_integrals;
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmotomo_);
    resort_two(one_electron_integrals_b,cmotomo_);
    resort_two(diagonal_aphys_tei_aa,cmotomo_);
    resort_two(diagonal_aphys_tei_ab,cmotomo_);
    resort_two(diagonal_aphys_tei_bb,cmotomo_);

    //resort_three(ThreeIntegral_,cmo2mo);

    outfile->Printf("\n Resorting integrals takes   %8.8fs", resort_integrals.get());
}
}}
