/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libfock/jk.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

#include "helpers/printing.h"
#include "helpers/lbfgs/lbfgs.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/rdms.h"

#include "casscf/casscf_orb_grad.h"

using namespace psi;
using namespace ambit;

namespace forte {

CASSCF_ORB_GRAD::CASSCF_ORB_GRAD(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 std::shared_ptr<ForteIntegrals> ints)
    : options_(options), mo_space_info_(mo_space_info), ints_(ints) {
    startup();
}

void CASSCF_ORB_GRAD::startup() {
    // setup MO spaces
    setup_mos();

    // read and print options
    read_options();

    // nonredundant pairs
    nonredundant_pairs();

    // setup JK
    setup_JK();

    // allocate memory for tensors and matrices
    init_tensors();

    // compute the initial MO integrals
    build_mo_integrals();
}

void CASSCF_ORB_GRAD::setup_mos() {
    nirrep_ = mo_space_info_->nirrep();

    nsopi_ = ints_->nsopi();
    nmopi_ = mo_space_info_->dimension("ALL");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");
    ndoccpi_ = mo_space_info_->dimension("INACTIVE_DOCC");
    nfrzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    nactvpi_ = mo_space_info_->dimension("ACTIVE");

    nso_ = nsopi_.sum();
    nmo_ = nmopi_.sum();
    ncmo_ = ncmopi_.sum();
    nactv_ = nactvpi_.sum();
    nfrzc_ = nfrzcpi_.sum();

    core_mos_ = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->absolute_mo("ACTIVE");

    label_to_mos_.clear();
    label_to_mos_["f"] = mo_space_info_->absolute_mo("FROZEN_DOCC");
    label_to_mos_["c"] = core_mos_;
    label_to_mos_["a"] = actv_mos_;
    label_to_mos_["v"] = mo_space_info_->absolute_mo("RESTRICTED_UOCC");

    // in Pitzer ordering
    mos_rel_.clear();
    mos_rel_.reserve(nmo_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi_[h]; ++i) {
            mos_rel_.push_back(std::make_pair(h, i));
        }
    }

    // in Pitzer ordering
    mos_rel_space_.resize(nmo_);
    for (std::string space : {"f", "c", "a", "v"}) {
        const auto& mos = label_to_mos_[space];
        for (size_t p = 0, size = mos.size(); p < size; ++p) {
            mos_rel_space_[mos[p]] = std::make_pair(space, p);
        }
    }

    // set up ambit spaces
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    BlockedTensor::add_mo_space("f", "I,J", label_to_mos_["f"], NoSpin);
    BlockedTensor::add_mo_space("c", "i,j", core_mos_, NoSpin);
    BlockedTensor::add_mo_space("a", "t,u,v,w,y,x,z", actv_mos_, NoSpin);
    BlockedTensor::add_mo_space("v", "a,b", label_to_mos_["v"], NoSpin);

    BlockedTensor::add_composite_mo_space("o", "k,l", {"c", "a"});
    BlockedTensor::add_composite_mo_space("g", "p,q,r,s", {"c", "a", "v"});
    BlockedTensor::add_composite_mo_space("G", "P,Q,R,S", {"f", "c", "a", "v"});
}

void CASSCF_ORB_GRAD::read_options() {
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    int_type_ = options_->get_str("INT_TYPE");

    g_conv_ = options_->get_double("CASSCF_G_CONVERGENCE");

    internal_rot_ = options_->get_bool("CASSCF_INTERNAL_ROT");
    orb_type_redundant_ = options_->get_str("CASSCF_FINAL_ORBITAL");

    // zero rotations
    zero_rots_.resize(nirrep_);
    auto zero_rots = options_->get_gen_list("CASSCF_ZERO_ROT");

    if (zero_rots.size() != 0) {
        size_t npairs = zero_rots.size();
        for (size_t i = 0; i < npairs; ++i) {
            py::list pair = zero_rots[i];
            if (pair.size() != 3) {
                outfile->Printf("\n  Error: invalid input of CASSCF_ZERO_ROT.");
                outfile->Printf("\n  Each entry should take an array of three numbers.");
                throw std::runtime_error("Invalid input of CASSCF_ZERO_ROT");
            }

            int irrep = py::cast<int>(pair[0]);
            if (irrep >= nirrep_ or irrep < 0) {
                outfile->Printf("\n  Error: invalid irrep in CASSCF_ZERO_ROT.");
                outfile->Printf("\n  Check the input irrep (start from 0) not to exceed %d",
                                nirrep_ - 1);
                throw std::runtime_error("Invalid irrep in CASSCF_ZERO_ROT");
            }

            size_t i1 = py::cast<size_t>(pair[1]) - 1;
            size_t i2 = py::cast<size_t>(pair[2]) - 1;
            size_t n = nmopi_[irrep];
            if (i1 >= n or i1 < 0 or i2 >= n or i2 < 0) {
                outfile->Printf("\n  Error: invalid orbital indices in CASSCF_ZERO_ROT.");
                outfile->Printf("\n The input orbital indices (start from 1) should not exceed %d "
                                "(number of orbitals in irrep %d)",
                                n, irrep);
                throw std::runtime_error("Invalid orbital indices in CASSCF_ZERO_ROT");
            }

            zero_rots_[irrep][i1].emplace(i2);
            zero_rots_[irrep][i2].emplace(i1);
        }
    }
}

void CASSCF_ORB_GRAD::nonredundant_pairs() {
    // prepare indices for rotation pairs
    rot_mos_irrep_.clear();
    rot_mos_block_.clear();

    std::map<std::string, std::vector<int>> nrots{{"vc", std::vector<int>(nirrep_, 0)},
                                                  {"va", std::vector<int>(nirrep_, 0)},
                                                  {"ac", std::vector<int>(nirrep_, 0)}};

    for (const std::string& block : {"vc", "va", "ac"}) {
        const auto& mos1 = label_to_mos_[block.substr(0, 1)];
        const auto& mos2 = label_to_mos_[block.substr(1, 1)];

        for (int i = 0, si = mos1.size(); i < si; ++i) {
            int hi = mos_rel_[mos1[i]].first;
            auto ni = mos_rel_[mos1[i]].second;

            for (int j = 0, sj = mos2.size(); j < sj; ++j) {
                int hj = mos_rel_[mos2[j]].first;
                auto nj = mos_rel_[mos2[j]].second;

                if (hi == hj) {
                    if (zero_rots_[hi].find(ni) != zero_rots_[hi].end()) {
                        if (zero_rots_[hi][ni].find(nj) != zero_rots_[hi][ni].end())
                            break;
                    }
                    rot_mos_irrep_.push_back(std::make_tuple(hi, ni, nj));
                    rot_mos_block_.push_back(std::make_tuple(block, i, j));
                    nrots[block][hi] += 1;
                }
            }
        }
    }

    if (internal_rot_) {
        nrots["aa"] = std::vector<int>(nirrep_, 0);

        const auto& mos = label_to_mos_["a"];
        for (int i = 0, s = mos.size(); i < s; ++i) {
            int hi = mos_rel_[mos[i]].first;
            auto ni = mos_rel_[mos[i]].second;

            for (int j = i + 1; j < s; ++j) {
                int hj = mos_rel_[mos[j]].first;
                auto nj = mos_rel_[mos[j]].second;

                if (hi == hj) {
                    if (zero_rots_[hi].find(ni) != zero_rots_[hi].end()) {
                        if (zero_rots_[hi][ni].find(nj) != zero_rots_[hi][ni].end())
                            break;
                    }
                    rot_mos_irrep_.push_back(std::make_tuple(hi, nj, ni));
                    rot_mos_block_.push_back(std::make_tuple("aa", j, i));
                    nrots["aa"][hi] += 1;
                }
            }
        }
    }

    nrot_ = rot_mos_irrep_.size();

    // printing
    auto ct = psi::Process::environment.molecule()->point_group()->char_table();
    std::map<std::string, std::string> space_map{
        {"c", "RESTRICTED_DOCC"}, {"a", "ACTIVE"}, {"v", "RESTRICTED_UOCC"}};

    print_h2("Independent Orbital Rotations");
    outfile->Printf("\n    %-33s", "ORBITAL SPACES");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf("  %4s", ct.gamma(h).symbol());
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 6, '-').c_str());

    for (const auto& key_value : nrots) {
        const auto& key = key_value.first;
        auto block1 = space_map[key.substr(0, 1)];
        auto block2 = space_map[key.substr(1, 1)];
        outfile->Printf("\n    %15s / %15s", block1.c_str(), block2.c_str());

        const auto& value = key_value.second;
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf("  %4zu", value[h]);
        }
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 6, '-').c_str());
}

void CASSCF_ORB_GRAD::setup_JK() {
    local_timer jk_timer;
    print_h2("Initialize JK Builder", "==>", "<==\n");

    auto basis_set = ints_->wfn()->basisset();

    if (int_type_.find("DF") != std::string::npos) {
        if (options_->get_str("SCF_TYPE") == "DF") {
            JK_ = JK::build_JK(basis_set, ints_->wfn()->get_basisset("DF_BASIS_SCF"),
                               psi::Process::environment.options, "MEM_DF");
        } else {
            throw psi::PSIEXCEPTION("Inconsistent DF type between Psi4 and Forte");
        }
    } else if (int_type_ == "CHOLESKY") {
        psi::Options& options = psi::Process::environment.options;
        CDJK* jk = new CDJK(basis_set, options_->get_double("CHOLESKY_TOLERANCE"));

        if (options["INTS_TOLERANCE"].has_changed())
            jk->set_cutoff(options.get_double("INTS_TOLERANCE"));
        if (options["SCREENING"].has_changed())
            jk->set_csam(options.get_str("SCREENING") == "CSAM");
        if (options["PRINT"].has_changed())
            jk->set_print(options.get_int("PRINT"));
        if (options["DEBUG"].has_changed())
            jk->set_debug(options.get_int("DEBUG"));
        if (options["BENCH"].has_changed())
            jk->set_bench(options.get_int("BENCH"));
        if (options["DF_INTS_IO"].has_changed())
            jk->set_df_ints_io(options.get_str("DF_INTS_IO"));
        jk->set_condition(options.get_double("DF_FITTING_CONDITION"));
        if (options["DF_INTS_NUM_THREADS"].has_changed())
            jk->set_df_ints_num_threads(options.get_int("DF_INTS_NUM_THREADS"));

        JK_ = std::shared_ptr<JK>(jk);
    } else if (int_type_ == "CONVENTIONAL") {
        JK_ = JK::build_JK(basis_set, psi::BasisSet::zero_ao_basis_set(),
                           psi::Process::environment.options, "PK");
    }

    JK_->set_memory(psi::Process::environment.get_memory() * 0.85);
    JK_->initialize();
    JK_->C_left().clear();
    JK_->C_right().clear();

    if (print_ > 1)
        outfile->Printf("  Initializing JK took %.3f seconds.", jk_timer.get());
}

void CASSCF_ORB_GRAD::init_tensors() {
    // save a copy of initial MO
    C0_ = ints_->wfn()->Ca()->clone();
    C0_->set_name("MCSCF Orbital Coefficients");
    C_ = C0_->clone();

    // save a copy of AO OEI
    H_ao_ = ints_->wfn()->H()->clone();

    // Fock matrices
    Fd_.resize(nmo_);
    Fock_ = std::make_shared<psi::Matrix>("Fock_MO", nmopi_, nmopi_);
    F_closed_ = std::make_shared<psi::Matrix>("Fock_inactive", nmopi_, nmopi_);

    auto tensor_type = ambit::CoreTensor;
    Fc_ = ambit::BlockedTensor::build(tensor_type, "Fc", {"GG"});
    F_ = ambit::BlockedTensor::build(tensor_type, "F", {"GG"});

    // two-electron integrals
    V_ = ambit::BlockedTensor::build(tensor_type, "V", {"Gaaa"});

    // 1-RDM and 2-RDM
    D1_ = ambit::BlockedTensor::build(tensor_type, "1RDM", {"aa"});
    D2_ = ambit::BlockedTensor::build(tensor_type, "2RDM", {"aaaa"});
    rdm1_ = std::make_shared<psi::Matrix>("1RDM", nactvpi_, nactvpi_);

    // orbital gradients related
    A_ = ambit::BlockedTensor::build(tensor_type, "A", {"GG"});

    std::vector<std::string> g_blocks{"ac", "vo"};
    if (internal_rot_) {
        g_blocks.push_back("aa");
        Guu_ = ambit::BlockedTensor::build(CoreTensor, "Guu", {"aa"});
        Guv_ = ambit::BlockedTensor::build(CoreTensor, "Guv", {"aa"});
        jk_internal_ = ambit::BlockedTensor::build(CoreTensor, "tei_internal", {"aaa"});
        d2_internal_ = ambit::BlockedTensor::build(CoreTensor, "rdm_internal", {"aaa"});
    }
    g_ = ambit::BlockedTensor::build(tensor_type, "g", g_blocks);
    h_diag_ = ambit::BlockedTensor::build(tensor_type, "h_diag", g_blocks);

    grad_ = std::make_shared<psi::Vector>("Gradient Vector", nrot_);
    hess_diag_ = std::make_shared<psi::Vector>("Diagonal Hessian", nrot_);

    // orbital rotation related
    R_ = std::make_shared<psi::Matrix>("Skew-Symmetric Orbital Rotation", nmopi_, nmopi_);
    U_ = std::make_shared<psi::Matrix>("Orthogonal Transformation", nmopi_, nmopi_);
    U_->identity();
}

void CASSCF_ORB_GRAD::build_mo_integrals() {
    // form closed-shell Fock matrix
    build_fock_inactive();

    // form the MO 2e-integrals
    build_tei_from_ao();
}

void CASSCF_ORB_GRAD::build_tei_from_ao() {
    // This function will do an integral transformation using the JK builder,
    // and return the integrals of type <px|uy> = (pu|xy).
    timer_on("Build (pu|xy) integrals");

    // Transform C matrix to C1 symmetry (JK will do this anyway)
    psi::SharedMatrix aotoso = ints_->wfn()->aotoso();
    auto C_nosym = std::make_shared<psi::Matrix>(nso_, nmo_);

    // Transform from the SO to the AO basis for the C matrix
    // MO in Pitzer ordering and only keep the non-frozen MOs
    for (int h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi_[h]; ++i) {
            int nao = nso_, nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &C_->pointer(h)[0][i],
                    nmopi_[h], 0.0, &C_nosym->pointer()[0][index], nmo_);

            index += 1;
        }
    }

    // set up the active part of the C matrix
    auto Cact = std::make_shared<psi::Matrix>("Cact", nso_, nactv_);
    std::vector<std::shared_ptr<psi::Matrix>> Cact_vec(nactv_);

    for (size_t x = 0; x < nactv_; ++x) {
        psi::SharedVector Ca_nosym_vec = C_nosym->get_column(0, actv_mos_[x]);
        Cact->set_column(0, x, Ca_nosym_vec);

        std::string name = "Cact slice " + std::to_string(x);
        auto temp = std::make_shared<psi::Matrix>(name, nso_, 1);
        temp->set_column(0, 0, Ca_nosym_vec);
        Cact_vec[x] = temp;
    }

    // The following type of integrals are needed:
    // (pu|xy) = C_{Mp}^T C_{Nu} C_{Rx}^T C_{Sy} (MN|RS)
    //         = C_{Mp}^T C_{Nu} J_{MN}^{xy}
    //         = C_{Mp}^T J_{MN}^{xy} C_{Nu}

    JK_->set_do_K(false);
    std::vector<psi::SharedMatrix>& Cl = JK_->C_left();
    std::vector<psi::SharedMatrix>& Cr = JK_->C_right();
    Cl.clear();
    Cr.clear();

    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = x; y < nactv_; ++y) {
            Cl.push_back(Cact_vec[x]);
            Cr.push_back(Cact_vec[y]);
        }
    }

    JK_->compute();

    // second-half transformation and fill in to BlockedTensor
    size_t nactv2 = nactv_ * nactv_;
    size_t nactv3 = nactv2 * nactv_;

    for (size_t x = 0, offset = 0; x < nactv_; ++x) {
        offset += x;
        for (size_t y = x; y < nactv_; ++y) {
            std::shared_ptr<psi::Matrix> J = JK_->J()[x * nactv_ + y - offset];
            auto half_trans = psi::linalg::triplet(C_nosym, J, Cact, true, false, false);

            for (size_t p = 0; p < nmo_; ++p) {
                // grab the block data
                std::string p_space = mos_rel_space_[p].first;
                std::string block = p_space + "aaa";

                size_t np = mos_rel_space_[p].second;
                auto& data = V_.block(block).data();

                for (size_t u = 0; u < nactv_; ++u) {
                    double value = half_trans->get(p, u);

                    data[np * nactv3 + u * nactv2 + x * nactv_ + y] = value;
                    data[np * nactv3 + u * nactv2 + y * nactv_ + x] = value;
                }
            }
        }
    }

    timer_off("Build (pu|xy) integrals");
}

void CASSCF_ORB_GRAD::build_fock_inactive() {
    // Implementation Notes (in AO basis)
    // F_frozen = D_{uv}^{frozen} * (2 * (uv|rs) - (us|rv))
    // F_restricted = D_{uv}^{restricted} * (2 * (uv|rs) - (us|rv))
    // F_inactive = Hcore + F_frozen + F_restricted
    // D_{uv}^{frozen} = \sum_{i}^{frozen} C_{ui} * C_{vi}
    // D_{uv}^{restricted} = \sum_{i}^{restricted} C_{ui} * C_{vi}

    // grab part of Ca for inactive docc
    auto Cdocc = std::make_shared<psi::Matrix>("C_INACTIVE", nirrep_, nsopi_, ndoccpi_);
    for (int h = 0; h < nirrep_; h++) {
        for (int i = 0; i < ndoccpi_[h]; i++) {
            Cdocc->set_column(h, i, C_->get_column(h, i));
        }
    }

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    JK_->set_do_K(true);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cdocc); // Cr is the same as Cl
    JK_->compute();

    auto J = JK_->J()[0];
    J->scale(2.0);
    J->subtract(JK_->K()[0]);
    J->add(H_ao_);

    F_closed_->copy(J);
    F_closed_->transform(C_);

    // put it in Ambit BlockedTensor format
    format_fock(F_closed_, Fc_);

    // compute closed-shell energy
    J->add(H_ao_);
    e_closed_ = J->vector_dot(psi::linalg::doublet(Cdocc, Cdocc, false, true));

    if (debug_print_) {
        F_closed_->print();
        outfile->Printf("\n  Frozen-core energy   %20.15f", ints_->frozen_core_energy());
        outfile->Printf("\n  Closed-shell energy  %20.15f", e_closed_);
    }
}

void CASSCF_ORB_GRAD::build_fock(bool rebuild_inactive) {
    if (rebuild_inactive) {
        build_fock_inactive();
    }

    build_fock_active();

    Fock_->add(F_closed_);
    Fock_->set_name("Fock_MO");

    format_fock(Fock_, F_);

    // fill in diagonal Fock in Pitzer ordering
    for (const std::string& space : {"f", "c", "a", "v"}) {
        std::string block = space + space;
        auto mos = label_to_mos_[space];
        for (size_t i = 0, size = mos.size(); i < size; ++i) {
            Fd_[mos[i]] = F_.block(block).data()[i * size + i];
        }
    }

    if (debug_print_) {
        Fock_->print();
    }
}

void CASSCF_ORB_GRAD::build_fock_active() {
    // Implementation Notes (in AO basis)
    // F_active = D_{uv}^{active} * ( (uv|rs) - 0.5 * (us|rv) )
    // D_{uv}^{active} = \sum_{xy}^{active} C_{ux} * C_{vy} * Gamma1_{xy}

    // grab part of Ca for active
    auto Cactv = std::make_shared<psi::Matrix>("C_ACTIVE", nirrep_, nsopi_, nactvpi_);
    for (int h = 0; h < nirrep_; h++) {
        for (int i = 0, offset = ndoccpi_[h]; i < nactvpi_[h]; i++) {
            Cactv->set_column(h, i, C_->get_column(h, i + offset));
        }
    }

    // dress Cactv by one-density, which will the C_right for JK
    auto Cactv_dressed = linalg::doublet(Cactv, rdm1_, false, false);

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    JK_->set_do_K(true);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cactv);
    Cr.push_back(Cactv_dressed);
    JK_->compute();

    Fock_->copy(JK_->K()[0]);
    Fock_->scale(-0.5);
    Fock_->add(JK_->J()[0]);

    // transform to MO
    Fock_->transform(C_);

    if (debug_print_) {
        Fock_->set_name("Fock_active");
        Fock_->print();
    }
}

void CASSCF_ORB_GRAD::format_fock(psi::SharedMatrix Fock, ambit::BlockedTensor F) {
    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        auto irrep_index_pair1 = mos_rel_[i[0]];
        auto irrep_index_pair2 = mos_rel_[i[1]];

        int h1 = irrep_index_pair1.first;

        if (h1 == irrep_index_pair2.first) {
            auto p = irrep_index_pair1.second;
            auto q = irrep_index_pair2.second;
            value = Fock->get(h1, p, q);
        } else {
            value = 0.0;
        }
    });
}

double CASSCF_ORB_GRAD::evaluate(psi::SharedVector x, psi::SharedVector g, bool do_g) {
    // if need to update orbitals and integrals
    if (update_orbitals(x)) {
        build_mo_integrals();
    }

    compute_reference_energy();

    // if need to compute gradient
    if (do_g) {
        build_fock();
        compute_orbital_grad();
        g->copy(*grad_);
    }

    return energy_;
}

bool CASSCF_ORB_GRAD::update_orbitals(psi::SharedVector x) {
    // test if need to update orbitals
    auto dR = std::make_shared<psi::Matrix>("Delta Orbital Rotation", nmopi_, nmopi_);

    for (size_t n = 0; n < nrot_; ++n) {
        int h, i, j;
        std::tie(h, i, j) = rot_mos_irrep_[n];

        dR->set(h, i, j, x->get(n));
        dR->set(h, j, i, -x->get(n));
    }

    dR->subtract(R_);

    // incoming x consistent with R_, no need to update orbitals
    if (dR->absmax() < 1.0e-13)
        return false;

    // officially save progress of dR
    R_->add(dR);

    // U_new = U_old * exp(dR)
    U_ = psi::linalg::doublet(U_, matrix_exponential(dR, 3), false, false);

    // update orbitals
    C_ = psi::linalg::doublet(C0_, U_, false, false);
    C_->set_name(C0_->name());

    // printing
    if (debug_print_) {
        dR->print();
        R_->print();
        U_->print();
        C_->print();
    }

    return true;
}

psi::SharedMatrix CASSCF_ORB_GRAD::matrix_exponential(psi::SharedMatrix A, int n) {
    auto U = std::make_shared<psi::Matrix>("U", A->rowspi(), A->colspi());
    U->identity();
    U->add(A);

    if (n > 1) {
        auto M = A->clone();
        M->set_name("M");

        for (int i = 1; i < n; ++i) {
            auto B = psi::linalg::doublet(M, A, false, false);
            B->scale(1.0 / (i + 1));
            U->add(B);
            M->copy(B);
        }
    }

    U->schmidt();
    return U;
}

void CASSCF_ORB_GRAD::compute_reference_energy() {
    // compute energy given that all useful MO integrals are available
    energy_ = e_closed_ + ints_->nuclear_repulsion_energy();
    energy_ += Fc_["uv"] * D1_["uv"];
    energy_ += 0.5 * V_["uvxy"] * D2_["uvxy"];

    if (debug_print_) {
        outfile->Printf("\n  Reference energy     %20.15f", energy_);
    }
}

void CASSCF_ORB_GRAD::compute_orbital_grad() {
    // build orbital Lagrangian
    A_["Ri"] = 2.0 * F_["Ri"];
    A_["Ru"] = Fc_["Rt"] * D1_["tu"];
    A_["Ru"] += V_["Rtvw"] * D2_["tuvw"];

    // build orbital gradients
    g_["pq"] = 2.0 * A_["pq"];
    g_["pq"] -= 2.0 * A_["qp"];

    // reshape and format to SharedVector
    reshape_rot_ambit(g_, grad_);

    if (debug_print_) {
        grad_->print();
    }
}

void CASSCF_ORB_GRAD::hess_diag(psi::SharedVector, psi::SharedVector h0) {
    compute_orbital_hess_diag();
    h0->copy(*hess_diag_);
}

void CASSCF_ORB_GRAD::compute_orbital_hess_diag() {
    // modified diagonal Hessian from Theor. Chem. Acc. 97, 88-95 (1997)

    // virtual-core block
    h_diag_.block("vc").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
    });

    // virtual-active block
    auto& d1_data = D1_.block("aa").data();
    auto& a_data = A_.block("aa").data();

    h_diag_.block("va").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = i[1] * nactv_ + i[1];
        value = 2.0 * (Fd_[i0] * d1_data[i1] - a_data[i1]);
    });

    // active-core block
    h_diag_.block("ac").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["a"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        auto i1p = i[0] * nactv_ + i[0];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
        value += 2.0 * (Fd_[i1] * d1_data[i1p] - a_data[i1p]);
    });

    // active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
    if (internal_rot_) {
        size_t nactv2 = nactv_ * nactv_;
        size_t nactv3 = nactv2 * nactv_;

        auto& fc_data = Fc_.block("aa").data();
        auto& v_data = V_.block("aaaa").data();
        auto& d2_data = D2_.block("aaaa").data();

        // G^{uu}_{vv}

        // (uu|xy)
        jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            value = v_data[idx];
        });

        // D_{vv,xy}
        d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            value = d2_data[idx];
        });

        Guu_["uv"] = jk_internal_["uxy"] * d2_internal_["vxy"];

        // (ux|uy)
        jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            value = v_data[idx];
        });

        // D_{vx,vy}
        d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            value = d2_data[idx];
        });

        Guu_["uv"] += 2.0 * jk_internal_["uxy"] * d2_internal_["vxy"];

        Guu_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto i0 = i[0] * nactv_ + i[0];
            auto i1 = i[1] * nactv_ + i[1];
            value += fc_data[i0] * d1_data[i1];
        });

        // G^{uv}_{vu}
        Guv_["uv"] = Fc_["uv"] * D1_["vu"];
        Guv_["uv"] += V_["uvxy"] * D2_["vuxy"];
        Guv_["uv"] += 2.0 * V_["uxvy"] * D2_["vxuy"];

        // build diagonal Hessian
        h_diag_["uv"] = 2.0 * Guu_["uv"];
        h_diag_["uv"] += 2.0 * Guu_["vu"];
        h_diag_["uv"] -= 2.0 * Guv_["uv"];
        h_diag_["uv"] -= 2.0 * Guv_["vu"];

        h_diag_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto i0 = i[0] * nactv_ + i[0];
            auto i1 = i[1] * nactv_ + i[1];
            value -= 2.0 * (a_data[i0] + a_data[i1]);
        });
    }

    // reshape and format to SharedVector
    reshape_rot_ambit(h_diag_, hess_diag_);

    if (debug_print_) {
        hess_diag_->print();
    }
}

void CASSCF_ORB_GRAD::reshape_rot_ambit(ambit::BlockedTensor bt, psi::SharedVector sv) {
    size_t vec_size = sv->dimpi().sum();
    if (vec_size != nrot_) {
        std::runtime_error("Inconsistent size between SharedVector and number of rotaitons");
    }

    for (size_t n = 0; n < nrot_; ++n) {
        std::string block;
        int i, j;
        std::tie(block, i, j) = rot_mos_block_[n];
        auto tensor = bt.block(block);
        auto dim1 = tensor.dim(1);

        sv->set(n, tensor.data()[i * dim1 + j]);
    }
}

void CASSCF_ORB_GRAD::set_rdms(RDMs& rdms) {
    // form spin-summed densities
    D1_.block("aa").copy(rdms.g1a());
    D1_.block("aa")("pq") += rdms.g1b()("pq");
    format_1rdm();

    // change to chemists' notation
    D2_.block("aaaa")("pqrs") = rdms.SFg2()("prqs");
    D2_.block("aaaa")("pqrs") += rdms.SFg2()("qrps");
    D2_.scale(0.5);
}

void CASSCF_ORB_GRAD::format_1rdm() {
    const auto& d1_data = D1_.block("aa").data();

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < nactvpi_[h]; ++u) {
            size_t nu = u + offset;
            for (int v = 0; v < nactvpi_[h]; ++v) {
                rdm1_->set(h, u, v, d1_data[nu * nactv_ + v + offset]);
            }
        }
        offset += nactvpi_[h];
    }

    if (debug_print_) {
        rdm1_->print();
    }
}

std::shared_ptr<ActiveSpaceIntegrals> CASSCF_ORB_GRAD::active_space_ints() {
    auto actv_sym = mo_space_info_->symmetry("ACTIVE");
    auto fci_ints = std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_sym, core_mos_);

    // build antisymmetrized TEI in physists' notation
    auto actv_ab = ambit::Tensor::build(CoreTensor, "tei_actv_aa", std::vector<size_t>(4, nactv_));
    actv_ab("pqrs") = V_.block("aaaa")("prqs");

    auto actv_aa = ambit::Tensor::build(CoreTensor, "tei_actv_aa", std::vector<size_t>(4, nactv_));
    actv_aa.copy(actv_ab);
    actv_aa("uvxy") -= actv_ab("uvyx");

    fci_ints->set_active_integrals(actv_aa, actv_ab, actv_aa);

    auto& oei = Fc_.block("aa").data();
    fci_ints->set_restricted_one_body_operator(oei, oei);

    fci_ints->set_scalar_energy(e_closed_ - ints_->frozen_core_energy());

    return fci_ints;
}

void CASSCF_ORB_GRAD::canonicalize_final() {
    auto U = canonicalize();

    C_ = psi::linalg::doublet(C_, U, false, false);
    C_->set_name(C0_->name());

    build_mo_integrals();
}

std::shared_ptr<psi::Matrix> CASSCF_ORB_GRAD::canonicalize() {
    print_h2("Canonicalize Orbitals (" + orb_type_redundant_ + ")");

    // unitary rotation matrix for output
    auto U = std::make_shared<psi::Matrix>("U_redundant", nmopi_, nmopi_);
    U->identity();

    // diagonalize sub-blocks of Fock
    auto ncorepi = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto nvirtpi = mo_space_info_->dimension("RESTRICTED_UOCC");
    std::vector<psi::Dimension> mos_dim{ncorepi, nvirtpi};
    std::vector<psi::Dimension> mos_offsets{nfrzcpi_, ndoccpi_ + nactvpi_};
    std::vector<std::string> names{"RESTRICTED_DOCC", "RESTRICTED_UOCC"};

    if (orb_type_redundant_ == "CANONICAL") {
        mos_dim.push_back(nactvpi_);
        mos_offsets.push_back(ndoccpi_);
        names.push_back("ACTIVE");
    }

    for (int i = 0, size = mos_dim.size(); i < size; ++i) {
        std::string block_name = "Diagonalizing Fock block " + names[i] + " ...";
        outfile->Printf("\n    %-44s", block_name.c_str());

        auto dim = mos_dim[i];
        auto offset_dim = mos_offsets[i];

        auto Fsub = std::make_shared<psi::Matrix>("Fsub_" + names[i], dim, dim);

        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < dim[h]; ++p) {
                size_t np = p + offset_dim[h];
                for (int q = 0; q < dim[h]; ++q) {
                    size_t nq = q + offset_dim[h];
                    Fsub->set(h, p, q, Fock_->get(h, np, nq));
                }
            }
        }

        // test off-diagonal elements to decide if need to diagonalize this block
        auto Fsub_od = Fsub->clone();
        Fsub_od->zero_diagonal();

        double Fsub_max = Fsub_od->absmax();
        double Fsub_norm = std::sqrt(Fsub_od->sum_of_squares());

        double threshold_max = 0.1 * g_conv_;
        if (int_type_ == "CHOLESKY") {
            double cd_tlr = options_->get_double("CHOLESKY_TOLERANCE");
            threshold_max = (threshold_max < 0.5 * cd_tlr) ? 0.5 * cd_tlr : threshold_max;
        }
        double threshold_rms = std::sqrt(dim.sum() * (dim.sum() - 1) / 2.0) * threshold_max;

        // diagonalize
        if (Fsub_max > threshold_max or Fsub_norm > threshold_rms) {
            auto Usub = std::make_shared<psi::Matrix>("Usub_" + names[i], dim, dim);
            auto Esub = std::make_shared<psi::Vector>("Esub_" + names[i], dim);
            Fsub->diagonalize(Usub, Esub);

            // fill in data
            for (int h = 0; h < nirrep_; ++h) {
                for (int p = 0; p < dim[h]; ++p) {
                    size_t np = p + offset_dim[h];
                    for (int q = 0; q < dim[h]; ++q) {
                        size_t nq = q + offset_dim[h];
                        U->set(h, np, nq, Usub->get(h, p, q));
                    }
                }
            }
        } // end if need to diagonalize
        outfile->Printf(" Done.");
    } // end sub block

    // natural orbitals
    if (orb_type_redundant_ == "NATURAL") {
        std::string block_name = "Diagonalizing 1-RDM ...";
        outfile->Printf("\n    %-44s", block_name.c_str());

        auto Usub = std::make_shared<psi::Matrix>("Usub_ACTIVE", nactvpi_, nactvpi_);
        auto Esub = std::make_shared<psi::Vector>("Esub_ACTIVE", nactvpi_);
        rdm1_->diagonalize(Usub, Esub, descending);

        // fill in data
        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < nactvpi_[h]; ++p) {
                size_t np = p + ndoccpi_[h];
                for (int q = 0; q < nactvpi_[h]; ++q) {
                    size_t nq = q + ndoccpi_[h];
                    U->set(h, np, nq, Usub->get(h, p, q));
                }
            }
        }

        outfile->Printf(" Done.");
    }

    if (debug_print_)
        U->print();

    return U;
}

psi::SharedMatrix CASSCF_ORB_GRAD::Lagrangian() {
    // format A matrix
    auto L = std::make_shared<psi::Matrix>("Lagrangian AO Back-Transformed", nmopi_, nmopi_);

    A_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            auto irrep_index_pair1 = mos_rel_[i[0]];
            auto irrep_index_pair2 = mos_rel_[i[1]];

            int h1 = irrep_index_pair1.first;

            if (h1 == irrep_index_pair2.first) {
                auto p = irrep_index_pair1.second;
                auto q = irrep_index_pair2.second;
                L->set(h1, p, q, value);
            }
        });

    return L;
}

psi::SharedMatrix CASSCF_ORB_GRAD::opdm() {
    auto D1 = std::make_shared<psi::Matrix>("OPDM AO Back-Transformed", nmopi_, nmopi_);

    // inactive docc part
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            D1->set(h, i, i, 2.0);
        }
    }

    // active part
    for (int h = 0; h < nirrep_; ++h) {
        auto offset = ndoccpi_[h];
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset;
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset;
                D1->set(h, nu, nv, rdm1_->get(h, u, v));
                D1->set(h, nv, nu, rdm1_->get(h, v, u));
            }
        }
    }

    return D1;
}

void CASSCF_ORB_GRAD::dump_tpdm_iwl() {
    auto psio = _default_psio_lib_;
    IWL d2(psio.get(), PSIF_MO_TPDM, 1.0e-15, 0, 0);
    std::string name = "outfile";
    int print = debug_print_ ? 1 : 0;

    // inactive docc part
    auto docc_mos = mo_space_info_->absolute_mo("INACTIVE_DOCC");
    for (int i = 0, ndocc = docc_mos.size(); i < ndocc; ++i) {
        auto ni = docc_mos[i];
        d2.write_value(ni, ni, ni, ni, 1.0, print, name, 0);
        for (int j = 0; j < i; ++j) {
            auto nj = docc_mos[j];
            d2.write_value(ni, ni, nj, nj, 2.0, print, name, 0);
            d2.write_value(nj, nj, ni, ni, 2.0, print, name, 0);
            d2.write_value(ni, nj, nj, ni, -1.0, print, name, 0);
            d2.write_value(nj, ni, ni, nj, -1.0, print, name, 0);
        }
    }

    // 1-rdm part
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset + ndoccpi_[h];
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset + ndoccpi_[h];

                double d_uv = rdm1_->get(h, u, v);
                double d_vu = rdm1_->get(h, v, u);

                for (int i = 0, ndocc = docc_mos.size(); i < ndocc; ++i) {
                    auto ni = docc_mos[i];

                    d2.write_value(nu, nv, ni, ni, 2.0 * d_uv, print, name, 0);
                    d2.write_value(nu, ni, ni, nv, -d_uv, print, name, 0);

                    if (u != v) {
                        d2.write_value(nv, nu, ni, ni, 2.0 * d_vu, print, name, 0);
                        d2.write_value(nv, ni, ni, nu, -d_vu, print, name, 0);
                    }
                }
            }
        }
        offset += nmopi_[h];
    }

    // 2-rdm part
    auto& d2_data = D2_.block("aaaa").data();
    auto na2 = nactv_ * nactv_;
    auto na3 = nactv_ * na2;

    for (size_t u = 0; u < nactv_; ++u) {
        auto nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            auto nv = actv_mos_[v];
            for (size_t x = 0; x < nactv_; ++x) {
                auto nx = actv_mos_[x];
                for (size_t y = 0; y < nactv_; ++y) {
                    auto ny = actv_mos_[y];

                    double value = d2_data[u * na3 + v * na2 + x * nactv_ + y];
                    d2.write_value(nu, nv, nx, ny, 0.5 * value, print, name, 0);
                }
            }
        }
    }

    d2.flush(1);
    d2.set_keep_flag(1);
    d2.close();
}
} // namespace forte
