#include <cmath>

#include <psifiles.h>
#include <libmints/sointegral_twobody.h>
#include <libmints/integralparameters.h>

#include "integrals.h"
#include "blockedtensorfactory.h"

namespace psi{ namespace forte{

/**
     * @brief EffectiveIntegrals::EffectiveIntegrals
     * @param options - psi options class
     * @param restricted - type of integral transformation
     * @param resort_frozen_core -
     */
EffectiveIntegrals::EffectiveIntegrals(psi::Options &options, SharedWavefunction ref_wfn, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, restricted, resort_frozen_core, mo_space_info), ints_(nullptr){
    integral_type_ = Effective;

    wfn_ = ref_wfn;
    outfile->Printf("\n  Overall Effective Integrals timings\n\n");
    Timer ConvTime;
    allocate();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n  Effective integrals take %8.8f s", ConvTime.get());
}

EffectiveIntegrals::~EffectiveIntegrals()
{
    deallocate();
}

void EffectiveIntegrals::allocate()
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

void EffectiveIntegrals::deallocate()
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


void EffectiveIntegrals::transform_integrals()
{
    Timer int_timer;

    // Now we want the reference (SCF) wavefunction
    boost::shared_ptr<IntegralFactory> integral = wfn_->integral();

    boost::shared_ptr<SOBasisSet> sobasisset = wfn_->sobasisset();
//    boost::shared_ptr<TwoBodyAOInt> tb(integral->eri());
//    boost::shared_ptr<TwoBodySOInt> eri(new TwoBodySOInt(tb,integral));

//    ERISaver eri_saver(aptei_idx_,num_aptei);
//    SOShellCombinationsIterator shellIter(sobasisset, sobasisset, sobasisset, sobasisset);
//    for (shellIter.first(); shellIter.is_done() == false; shellIter.next()){
//        eri->compute_shell(shellIter,eri_saver);
//    }

    ambit::Tensor Vmo = ambit::Tensor::build(ambit::CoreTensor,"Vmo",{nmo_,nmo_,nmo_,nmo_});
    ambit::Tensor Vso = ambit::Tensor::build(ambit::CoreTensor,"Vso",{nso_,nso_,nso_,nso_});
    ambit::Tensor C = ambit::Tensor::build(ambit::CoreTensor,"C",{nso_,nmo_});

    double eff_coulomb_omega = options_.get_double("EFFECTIVE_COULOMB_OMEGA");
    double eff_coulomb_factor = options_.get_double("EFFECTIVE_COULOMB_FACTOR");
    double eff_coulomb_exp = options_.get_double("EFFECTIVE_COULOMB_EXPONENT");
    outfile->Printf("  Effective Coulomb Omega:              %6f\n",eff_coulomb_omega);
    outfile->Printf("  Effective Coulomb Gaussian Factor:    %6f\n",eff_coulomb_factor);
    outfile->Printf("  Effective Coulomb Gaussian Exponent:  %6f\n",eff_coulomb_exp);

    // Erf(x)/x integrals (long range)
    {
        boost::shared_ptr<TwoBodyAOInt> tb(integral->erf_eri(eff_coulomb_omega));
        boost::shared_ptr<TwoBodySOInt> eri(new TwoBodySOInt(tb,integral));

        ERISaver erf_eri_saver(aptei_idx_,num_aptei);
        SOShellCombinationsIterator shellIter(sobasisset, sobasisset, sobasisset, sobasisset);
        for (shellIter.first(); shellIter.is_done() == false; shellIter.next()){
            eri->compute_shell(shellIter,erf_eri_saver);
        }

        Vso.iterate([&](const std::vector<size_t>& i,double& value){
            value += erf_eri_saver.get(i[0],i[1],i[2],i[3]);
        });

    }

//    // Erfc(x)/x integrals (short range)
//    {
//        boost::shared_ptr<TwoBodyAOInt> tb(integral->erf_complement_eri(eff_coulomb_omega));
//        boost::shared_ptr<TwoBodySOInt> eri(new TwoBodySOInt(tb,integral));

//        ERISaver erf_complement_eri_saver(aptei_idx_,num_aptei);
//        SOShellCombinationsIterator shellIter(sobasisset, sobasisset, sobasisset, sobasisset);
//        for (shellIter.first(); shellIter.is_done() == false; shellIter.next()){
//            eri->compute_shell(shellIter,erf_complement_eri_saver);
//        }

//        Vso.iterate([&](const std::vector<size_t>& i,double& value){
//            value += erf_complement_eri_saver.get(i[0],i[1],i[2],i[3]);
//        });
//    }

    // Gaussian(gamma,c) integrals (short range)
    {
        boost::shared_ptr<Vector> coeff = boost::shared_ptr<Vector>(new Vector(1));
        boost::shared_ptr<Vector> exponent = boost::shared_ptr<Vector>(new Vector(1));
        coeff->set(0,eff_coulomb_factor);
        exponent->set(0,eff_coulomb_exp);

        boost::shared_ptr<CorrelationFactor> cf = boost::shared_ptr<CorrelationFactor>(new CorrelationFactor(1));
        cf->set_params(coeff,exponent);
        boost::shared_ptr<TwoBodyAOInt> tb(integral->f12(cf));
        boost::shared_ptr<TwoBodySOInt> eri(new TwoBodySOInt(tb,integral));

        ERISaver f12_eri_saver(aptei_idx_,num_aptei);
        SOShellCombinationsIterator shellIter(sobasisset, sobasisset, sobasisset, sobasisset);
        for (shellIter.first(); shellIter.is_done() == false; shellIter.next()){
            eri->compute_shell(shellIter,f12_eri_saver);
        }

        Vso.iterate([&](const std::vector<size_t>& i,double& value){
            value += f12_eri_saver.get(i[0],i[1],i[2],i[3]);
        });

    }

    SharedMatrix Ca = wfn_->Ca();
    // Remove symmetry from Ca
    Matrix Ca_nosym(nso_,nmo_);
    for (int h = 0, so_offset = 0, mo_offset = 0; h < nirrep_; ++h){
        for (int mu = 0; mu < nsopi_[h]; ++mu){
            for (int p = 0; p < nmopi_[h]; ++p){
                Ca_nosym.set(mu + so_offset,p + mo_offset,Ca->get(h,mu,p));
            }
        }
        so_offset += nsopi_[h];
        mo_offset += nmopi_[h];
    }

    C.iterate([&](const std::vector<size_t>& i,double& value){
        value = Ca_nosym.get(i[0],i[1]);
    });

    Vmo("pqrs") = Vso("abcd") * C("ap") * C("bq") * C("cr") * C("ds");

    outfile->Printf("\n  Reading the two-electron integrals from disk");
    outfile->Printf("\n  Size of two-electron integrals: %10.6f GB", double(3 * 8 * num_aptei) / 1073741824.0);
    int_mem_ = sizeof(double) * 3 * 8 * num_aptei / 1073741824.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_aa[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_ab[pqrs] = 0.0;
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) aphys_tei_bb[pqrs] = 0.0;

    double* two_electron_integrals = new double[num_aptei];
    for (size_t pqrs = 0; pqrs < num_aptei; ++pqrs) two_electron_integrals[pqrs] = 0.0;

    Vmo.iterate([&](const std::vector<size_t>& i,double& value){
        two_electron_integrals[aptei_index(i[0],i[1],i[2],i[3])] = value;
    });

    // Store the integrals
    for (size_t p = 0; p < nmo_; ++p){
        for (size_t q = 0; q < nmo_; ++q){
            for (size_t r = 0; r < nmo_; ++r){
                for (size_t s = 0; s < nmo_; ++s){
                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                    double direct   = two_electron_integrals[aptei_index(p,r,q,s)];
                    double exchange = two_electron_integrals[aptei_index(p,s,q,r)];
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

    outfile->Printf("\n  Integral transformation done. %8.8f s", int_timer.get());
    outfile->Flush();
}

double EffectiveIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    return aphys_tei_aa[aptei_index(p,q,r,s)];
}

double EffectiveIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
    return aphys_tei_ab[aptei_index(p,q,r,s)];
}

double EffectiveIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
    return aphys_tei_bb[aptei_index(p,q,r,s)];
}

ambit::Tensor EffectiveIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor EffectiveIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor EffectiveIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

void EffectiveIntegrals::set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2)
{
    double* p_tei;
    if (alpha1 == true and alpha2 == true) p_tei = aphys_tei_aa;
    if (alpha1 == true and alpha2 == false) p_tei = aphys_tei_ab;
    if (alpha1 == false and alpha2 == false) p_tei = aphys_tei_bb;
    size_t index = aptei_index(p,q,r,s);
    p_tei[index] = value;
}

void EffectiveIntegrals::gather_integrals()
{
    transform_integrals();
}

void EffectiveIntegrals::resort_integrals_after_freezing()
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

void EffectiveIntegrals::resort_four(double*& tei, std::vector<size_t>& map)
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

void EffectiveIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < aptei_idx_; ++p){
        for(size_t q = 0; q < aptei_idx_; ++q){
            diagonal_aphys_tei_aa[p * aptei_idx_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * aptei_idx_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * aptei_idx_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

void EffectiveIntegrals::make_fock_matrix(SharedMatrix gamma_a,SharedMatrix gamma_b)
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

