/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"

#include "base_classes/mo_space_info.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "forte-def.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "diskdf_integrals.h"

using namespace ambit;
using namespace psi;

namespace forte {

DISKDFIntegrals::DISKDFIntegrals(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<psi::Wavefunction> ref_wfn,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 IntegralSpinRestriction restricted)
    : Psi4Integrals(options, ref_wfn, mo_space_info, DiskDF, restricted) {
    initialize();
}

void DISKDFIntegrals::initialize() {
    print_info();

    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0 and (not skip_build_)) {
        local_timer int_timer;
        gather_integrals();
        freeze_core_orbitals();
        print_timing("disk-based density-fitted integrals", int_timer.get());
    }
}

double DISKDFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    size_t pn, qn, rn, sn;

    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    } else {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }

    std::vector<size_t> A_range = {0, nthree_};
    std::vector<size_t> p_range = {pn, pn + 1};
    std::vector<size_t> q_range = {qn, qn + 1};
    std::vector<size_t> r_range = {rn, rn + 1};
    std::vector<size_t> s_range = {sn, sn + 1};

    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    std::shared_ptr<psi::Matrix> B1(new psi::Matrix(1, nthree_));
    std::shared_ptr<psi::Matrix> B2(new psi::Matrix(1, nthree_));

    df_->fill_tensor("B", B1, A_range, p_range, r_range);
    df_->fill_tensor("B", B2, A_range, q_range, s_range);

    vpqrsalphaC = B1->vector_dot(B2);

    B1->zero();
    B2->zero();

    df_->fill_tensor("B", B1, A_range, p_range, s_range);
    df_->fill_tensor("B", B2, A_range, q_range, r_range);

    vpqrsalphaE = B1->vector_dot(B2);

    return (vpqrsalphaC - vpqrsalphaE);
}

double DISKDFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    size_t pn, qn, rn, sn;
    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    } else {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }
    std::vector<size_t> A_range = {0, nthree_};
    std::vector<size_t> p_range = {pn, pn + 1};
    std::vector<size_t> q_range = {qn, qn + 1};
    std::vector<size_t> r_range = {rn, rn + 1};
    std::vector<size_t> s_range = {sn, sn + 1};

    double vpqrsalphaC = 0.0;
    std::shared_ptr<psi::Matrix> B1(new psi::Matrix(1, nthree_));
    std::shared_ptr<psi::Matrix> B2(new psi::Matrix(1, nthree_));

    df_->fill_tensor("B", B1, A_range, p_range, r_range);
    df_->fill_tensor("B", B2, A_range, q_range, s_range);

    vpqrsalphaC = B1->vector_dot(B2);

    return (vpqrsalphaC);
}

double DISKDFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    size_t pn, qn, rn, sn;

    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    } else {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }
    std::vector<size_t> A_range = {0, nthree_};
    std::vector<size_t> p_range = {pn, pn + 1};
    std::vector<size_t> q_range = {qn, qn + 1};
    std::vector<size_t> r_range = {rn, rn + 1};
    std::vector<size_t> s_range = {sn, sn + 1};

    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    std::shared_ptr<psi::Matrix> B1(new psi::Matrix(1, nthree_));
    std::shared_ptr<psi::Matrix> B2(new psi::Matrix(1, nthree_));

    df_->fill_tensor("B", B1, A_range, p_range, r_range);
    df_->fill_tensor("B", B2, A_range, q_range, s_range);

    vpqrsalphaC = B1->vector_dot(B2);

    B1->zero();
    B2->zero();

    df_->fill_tensor("B", B1, A_range, p_range, s_range);
    df_->fill_tensor("B", B2, A_range, q_range, r_range);

    vpqrsalphaE = B1->vector_dot(B2);

    return (vpqrsalphaC - vpqrsalphaE);
}

ambit::Tensor DISKDFIntegrals::aptei_aa_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    auto p_size = p.size();
    auto q_size = q.size();
    auto r_size = r.size();
    auto s_size = s.size();

    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    auto Qpr = three_integral_block(Avec, p, r);
    auto Qqs = three_integral_block(Avec, q, s);

    auto out = ambit::Tensor::build(tensor_type_, "out_aa", {p_size, q_size, r_size, s_size});
    out("p,q,r,s") = Qpr("A,p,r") * Qqs("A,q,s");

    /// If p != q != r !=s need to form the Exchange part separately
    if (r != s) {
        auto Qps_K = ambit::Tensor::build(tensor_type_, "Qps_K", {nthree_, p_size, s_size});
        auto Qqr_K = ambit::Tensor::build(tensor_type_, "Qqr_K", {nthree_, q_size, r_size});
        Qps_K = three_integral_block(Avec, p, s);
        Qqr_K = three_integral_block(Avec, q, r);
        out("p,q,r,s") -= Qps_K("A,p,s") * Qqr_K("A,q,r");
    } else {
        out("p,q,r,s") -= Qpr("A,p,s") * Qqs("A,q,r");
    }

    return out;
}

ambit::Tensor DISKDFIntegrals::aptei_ab_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    auto p_size = p.size();
    auto q_size = q.size();
    auto r_size = r.size();
    auto s_size = s.size();

    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    auto out = ambit::Tensor::build(tensor_type_, "out_ab", {p_size, q_size, r_size, s_size});

    if (p == q and r == s) {
        auto Q = three_integral_block(Avec, p, r);
        out("p,q,r,s") = Q("A,p,r") * Q("A,q,s");
        return out;
    }

    auto Qpr = three_integral_block(Avec, p, r);
    auto Qqs = three_integral_block(Avec, q, s);

    out("p,q,r,s") = Qpr("A,p,r") * Qqs("A,q,s");

    return out;
}

ambit::Tensor DISKDFIntegrals::aptei_bb_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    auto p_size = p.size();
    auto q_size = q.size();
    auto r_size = r.size();
    auto s_size = s.size();

    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    auto Qpr = three_integral_block(Avec, p, r);
    auto Qqs = three_integral_block(Avec, q, s);

    auto out = ambit::Tensor::build(tensor_type_, "out_bb", {p_size, q_size, r_size, s_size});
    out("p,q,r,s") = Qpr("A,p,r") * Qqs("A,q,s");

    /// If p != q != r !=s need to form the Exchane part separately
    if (r != s) {
        auto Qps_K = ambit::Tensor::build(tensor_type_, "Qps_K", {nthree_, p_size, s_size});
        auto Qqr_K = ambit::Tensor::build(tensor_type_, "Qqr_K", {nthree_, q_size, r_size});
        Qps_K = three_integral_block(Avec, p, s);
        Qqr_K = three_integral_block(Avec, q, r);
        out("p,q,r,s") -= Qps_K("A,p,s") * Qqr_K("A,q,r");
    } else {
        out("p,q,r,s") -= Qpr("A,p,s") * Qqs("A,q,r");
    }

    return out;
}

double** DISKDFIntegrals::three_integral_pointer() { return (ThreeIntegral_->pointer()); }

ambit::Tensor DISKDFIntegrals::three_integral_block(const std::vector<size_t>& Q_vec,
                                                    const std::vector<size_t>& p_vec,
                                                    const std::vector<size_t>& q_vec) {
    std::string func_name = "DISKDFIntegrals::three_integral_block: ";

    auto Qsize = Q_vec.size();
    auto psize = p_vec.size();
    auto qsize = q_vec.size();
    auto pqsize = psize * qsize;

    auto out = ambit::Tensor::build(tensor_type_, "Return", {Qsize, psize, qsize});

    // directly return if any of the dimension is zero
    if (Qsize == 0 or psize == 0 or qsize == 0) {
        return out;
    }

    // test if indices out of range
    if (*std::max_element(Q_vec.begin(), Q_vec.end()) >= nthree_) {
        throw std::runtime_error(func_name + "auxiliary indices out of range");
    }
    if (*std::max_element(p_vec.begin(), p_vec.end()) >= aptei_idx_) {
        throw std::runtime_error(func_name + "MO indices p_vec out of range");
    }
    if (*std::max_element(q_vec.begin(), q_vec.end()) >= aptei_idx_) {
        throw std::runtime_error(func_name + "MO indices q_vec out of range");
    }

    // take care of frozen orbitals
    std::vector<size_t> cmotomo;                // from correlated MO to full MO
    if (frzcpi_.sum() && aptei_idx_ == ncmo_) { // there are frozen orbitals
        cmotomo = cmotomo_;
    } else {
        cmotomo.resize(nmo_);
        std::iota(cmotomo.begin(), cmotomo.end(), 0);
    }

    // make sure indices are contiguous
    for (size_t a = 1; a < Qsize; ++a) {
        if (Q_vec[a] != Q_vec[0] + a) {
            throw std::runtime_error(func_name + "auxiliary indices not contiguous");
        }
    }
    std::vector<size_t> Q_range{Q_vec[0], Q_vec[0] + Qsize};

    bool p_contiguous = true;
    for (size_t p = 1, p0 = cmotomo[p_vec[0]]; p < psize; ++p) {
        if (cmotomo[p_vec[p]] != p0 + p) {
            p_contiguous = false;
            break;
        }
    }

    bool q_contiguous = true;
    for (size_t q = 1, q0 = cmotomo[q_vec[0]]; q < qsize; ++q) {
        if (cmotomo[q_vec[q]] != q0 + q) {
            q_contiguous = false;
            break;
        }
    }

    auto& out_data = out.data();

    if (p_contiguous and q_contiguous) {
        std::vector<size_t> p_range{cmotomo[p_vec[0]], cmotomo[p_vec[0]] + psize};
        std::vector<size_t> q_range{cmotomo[q_vec[0]], cmotomo[q_vec[0]] + qsize};

        df_->fill_tensor("B", out_data.data(), Q_range, p_range, q_range);
    } else if ((not p_contiguous) and q_contiguous) {
        std::vector<size_t> q_range{cmotomo[q_vec[0]], cmotomo[q_vec[0]] + qsize};

        for (size_t p = 0; p < psize; ++p) {
            auto np = cmotomo[p_vec[p]];
            auto Aq = std::make_shared<psi::Matrix>("Aq", Qsize, qsize);
            df_->fill_tensor("B", Aq, Q_range, {np, np + 1}, q_range);

            for (size_t a = 0; a < Qsize; ++a) {
                for (size_t q = 0; q < qsize; ++q) {
                    out_data[a * pqsize + p * qsize + q] = Aq->get(a, q);
                }
            }
        }
    } else if (p_contiguous and (not q_contiguous)) {
        std::vector<size_t> p_range{cmotomo[p_vec[0]], cmotomo[p_vec[0]] + psize};

        for (size_t q = 0; q < qsize; ++q) {
            auto nq = cmotomo[q_vec[q]];
            auto Ap = std::make_shared<psi::Matrix>("Aq", Qsize, psize);
            df_->fill_tensor("B", Ap, Q_range, {nq, nq + 1}, p_range);

            for (size_t a = 0; a < Qsize; ++a) {
                for (size_t p = 0; p < psize; ++p) {
                    out_data[a * pqsize + p * qsize + q] = Ap->get(a, p);
                }
            }
        }
    } else {
        size_t memory = df_->get_memory() - Qsize * pqsize;
        std::vector<size_t> vec_small = psize < qsize ? p_vec : q_vec;

        size_t max_nslice = memory / (Qsize * nmo_);
        std::vector<size_t> batches(vec_small.size() / max_nslice, max_nslice);
        if (vec_small.size() % max_nslice)
            batches.push_back(vec_small.size() % max_nslice);

        for (size_t n = 0, offset = 0, nbatch = batches.size(); n < nbatch; ++n) {
            std::vector<psi::SharedMatrix> Am_vec;

            for (size_t i = 0; i < batches[n]; ++i) {
                auto ni = cmotomo[vec_small[i + offset]];
                auto Am = std::make_shared<psi::Matrix>("Am", Qsize, nmo_);
                df_->fill_tensor("B", Am, Q_range, {ni, ni + 1}, {0, nmo_});
                Am_vec.push_back(Am);
            }

            if (psize < qsize) {
                for (size_t i = 0; i < batches[n]; ++i) {
                    for (size_t a = 0; a < Qsize; ++a) {
                        for (size_t q = 0; q < qsize; ++q) {
                            auto idx = a * pqsize + (i + offset) * qsize + q;
                            out_data[idx] = Am_vec[i]->get(a, cmotomo[q_vec[q]]);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < batches[n]; ++i) {
                    for (size_t a = 0; a < Qsize; ++a) {
                        for (size_t p = 0; p < psize; ++p) {
                            auto idx = a * pqsize + p * qsize + (i + offset);
                            out_data[idx] = Am_vec[i]->get(a, cmotomo[p_vec[p]]);
                        }
                    }
                }
            }

            offset += batches[n];
        }

        //        // The following loops every index for debugging
        //        for (size_t ip = 0; ip < psize; ++ip) {
        //            auto pn = cmotomo[p_vec[ip]];
        //            std::vector<size_t> p_range = {pn, pn + 1};

        //            for (size_t iq = 0; iq < qsize; ++iq) {
        //                auto qn = cmotomo[q_vec[iq]];
        //                std::vector<size_t> q_range = {qn, qn + 1};

        //                double* A_chunk = new double[Asize];

        //                df_->fill_tensor("B", A_chunk, A_range, p_range, q_range);

        //                for (size_t a = 0; a < A_vec.size(); a++) {
        //                    out_data[a * pqsize + ip * qsize + iq] = A_chunk[a];
        //                }

        //                delete[] A_chunk;
        //            }
        //        }
    }

    return out;
}

void DISKDFIntegrals::set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
    outfile->Printf("\n  DISKDFIntegrals::set_tei : DISKDF integrals are read only");
    throw psi::PSIEXCEPTION("DISKDFIntegrals::set_tei : DISKDF integrals are read only");
}

void DISKDFIntegrals::gather_integrals() {
    outfile->Printf("\n Computing density fitted integrals\n");

    std::shared_ptr<psi::BasisSet> primary = wfn_->basisset();
    std::shared_ptr<psi::BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");

    size_t nprim = primary->nbf();
    size_t naux = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n  Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n  Need %8.6f GB to store DF integrals\n",
                    (nprim * nprim * naux * sizeof(double) / 1073741824.0));
    int_mem_ = (nprim * nprim * naux * sizeof(double));

    psi::Dimension nsopi_ = wfn_->nsopi();
    std::shared_ptr<psi::Matrix> aotoso = wfn_->aotoso();
    std::shared_ptr<psi::Matrix> Ca = wfn_->Ca();
    std::shared_ptr<psi::Matrix> Ca_ao(new psi::Matrix("Ca_ao", nso_, nmopi_.sum()));

    // Transform from the SO to the AO basis
    for (int h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi_[h]; ++i) {
            size_t nao = nso_;
            size_t nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &Ca->pointer(h)[0][i],
                    nmopi_[h], 0.0, &Ca_ao->pointer()[0][index], nmopi_.sum());

            index += 1;
        }
    }

    // B_{pq}^Q -> MO without frozen core

    // Constructs the DF function
    // I used this version of build as this doesn't build all the apces and
    // assume a RHF/UHF reference
    df_ = std::make_shared<psi::DFHelper>(primary, auxiliary);
    size_t mem_sys = psi::Process::environment.get_memory() * 0.9 / sizeof(double);
    int64_t mem = mem_sys;
    if (JK_status_ == JKStatus::initialized) {
        mem = mem_sys - JK_->memory_estimate();
        if (mem < 0) {
            auto xb = to_xb(static_cast<size_t>(-mem), sizeof(double));
            std::string msg = "Not enough memory! Need at least ";
            msg += std::to_string(xb.first) + " " + xb.second.c_str() + " more.";
            outfile->Printf("\n  %s", msg.c_str());
            throw psi::PSIEXCEPTION(msg);
        }
    }
    df_->set_memory(static_cast<size_t>(mem));
    df_->set_MO_core(false);
    df_->set_nthreads(omp_get_max_threads());
    df_->set_print_lvl(1);
    df_->initialize();
    df_->print_header();
    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    df_->add_space("ALL", Ca_ao);
    // Does not add the pair_space, but says which one is should use
    df_->add_transformation("B", "ALL", "ALL", "Qpq");

    // Finally computes the df integrals
    // Does the timings also
    local_timer timer;
    outfile->Printf("\n  Computing DF Integrals");
    df_->transform();
    print_timing("computing density-fitted integrals", timer.get());
}

void DISKDFIntegrals::resort_integrals_after_freezing() {
    local_timer resort_integrals;
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    // resort_three(ThreeIntegral_,cmo2mo);

    print_timing("resorting integrals", resort_integrals.get());
}

ambit::Tensor DISKDFIntegrals::three_integral_block_two_index(const std::vector<size_t>& A,
                                                              size_t p,
                                                              const std::vector<size_t>& q) {

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {A.size(), q.size()});

    size_t p_max, p_min;
    bool frozen_core = false;
    if (frzcpi_.sum() && aptei_idx_ == ncmo_) {
        frozen_core = true;
        p_min = cmotomo_[p];
        p_max = p_min + 1;
    } else {
        p_min = p;
        p_max = p_min + 1;
    }

    if (nthree_ == A.size()) {

        std::vector<size_t> arange = {0, nthree_};
        std::vector<size_t> qrange = {0, nmo_};
        std::vector<size_t> prange = {p_min, p_max};

        std::shared_ptr<psi::Matrix> Aq(new psi::Matrix("Aq", nthree_, nmo_));
        df_->fill_tensor("B", Aq, arange, prange, qrange);

        if (frozen_core) {
            ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
                value = Aq->get(A[i[0]], cmotomo_[q[i[1]]]);
            });
        } else {
            ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
                value = Aq->get(A[i[0]], q[i[1]]);
            });
        }
    } else {
        outfile->Printf("\n Not implemened for variable size in A");
        throw psi::PSIEXCEPTION("Can only use if 2nd parameter is a size_t and A.size==nthree_");
    }
    return ReturnTensor;
}

size_t DISKDFIntegrals::nthree() const { return nthree_; }

} // namespace forte
