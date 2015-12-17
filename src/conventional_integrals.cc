//[forte-public]
#include "integrals.h"
#include <cmath>

#include <psifiles.h>
#include <libiwl/iwl.h>
//#include <libtrans/integraltransform.h>
//#include <libpsio/psio.hpp>
//#include <libmints/matrix.h>
//#include <libmints/basisset.h>
//#include <libthce/thce.h>
//#include <libthce/thcew.h>
//#include <libthce/lreri.h>
//#include <lib3index/cholesky.h>
//#include <libqt/qt.h>
//#include <libfock/jk.h>
//#include <algorithm>
//#include <numeric>
#include "blockedtensorfactory.h"

namespace psi{ namespace forte{

/**
     * @brief ConventionalIntegrals::ConventionalIntegrals
     * @param options - psi options class
     * @param restricted - type of integral transformation
     * @param resort_frozen_core -
     */
ConventionalIntegrals::ConventionalIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, restricted, resort_frozen_core, mo_space_info), ints_(nullptr){
    integral_type_ = ConventionalInts;

    outfile->Printf("\n  Overall Conventional Integrals timings\n\n");
    Timer ConvTime;
    allocate();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n  Conventional integrals take %8.8f s", ConvTime.get());
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

    diagonal_aphys_tei_aa = new double[aptei_idx_ * aptei_idx_];
    diagonal_aphys_tei_ab = new double[aptei_idx_ * aptei_idx_];
    diagonal_aphys_tei_bb = new double[aptei_idx_ * aptei_idx_];

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

    if(print_ > 0)
    {
        outfile->Printf("\n  Integral transformation done. %8.8f s", int_timer.get());
    }
    outfile->Flush();
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

void ConventionalIntegrals::gather_integrals()
{
    transform_integrals();

    if(print_ > 0)
    {
        outfile->Printf("\n  Reading the two-electron integrals from disk");
        outfile->Printf("\n  Size of two-electron integrals: %10.6f GB", double(3 * 8 * num_aptei) / 1073741824.0);
    }
    int_mem_ = sizeof(double) * 3 * 8 * num_aptei / 1073741824.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_aa[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_ab[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_bb[pqrs] = 0.0;

    int ioffmax = 30000;
    int* myioff = new int[ioffmax];
    myioff[0] = 0;
    for(int i = 1; i < ioffmax; ++i)
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
    if(print_ > 0)
    {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }

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
    for(size_t p = 0; p < aptei_idx_; ++p){
        for(size_t q = 0; q < aptei_idx_; ++q){
            diagonal_aphys_tei_aa[p * aptei_idx_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * aptei_idx_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * aptei_idx_ + q] = aptei_bb(p,q,p,q);
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
    double zero = 1e-8;
    ///TODO: Either use ambit or use structure of gamma.
    for (size_t r = 0; r < ncmo_; ++r) {
        for (size_t s = 0; s < ncmo_; ++s) {
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
    for (size_t r = 0; r < ncmo_; ++r) {
        for (size_t s = 0; s < ncmo_; ++s) {
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



}} //End namespaces for psi and forte
