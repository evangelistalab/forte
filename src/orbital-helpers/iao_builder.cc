/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2016 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "iao_builder.h"

using namespace psi;

namespace forte {

IAOBuilder::IAOBuilder(std::shared_ptr<psi::BasisSet> primary, std::shared_ptr<psi::BasisSet> minao,
                       psi::SharedMatrix C)
    : C_(C), primary_(primary), minao_(minao) {
    if (C->nirrep() != 1) {
        throw psi::PSIEXCEPTION("Localizer: C matrix is not C1");
    }
    if (C->rowspi()[0] != primary->nbf()) {
        throw psi::PSIEXCEPTION("Localizer: C matrix does not match basis");
    }
    common_init();
}
IAOBuilder::~IAOBuilder() {}
void IAOBuilder::common_init() {
    print_ = 0;
    debug_ = 0;
    bench_ = 0;
    convergence_ = 1.0E-12;
    maxiter_ = 50;
    use_ghosts_ = false;
    power_ = 4;
    condition_ = 1.0E-7;
    use_stars_ = false;
    stars_completeness_ = 0.9;
    stars_.clear();
}
std::shared_ptr<IAOBuilder> IAOBuilder::build(std::shared_ptr<psi::BasisSet> primary,
                                              std::shared_ptr<psi::BasisSet> minao, psi::SharedMatrix C,
                                              psi::Options& options) {
    //    psi::Options& options = psi::Process::environment.options;

    //  std::shared_ptr<psi::BasisSet> minao = psi::BasisSet::pyconstruct_orbital(primary->molecule(),
    //      "BASIS", options.get_str("MINAO_BASIS"));

    std::shared_ptr<IAOBuilder> local(new IAOBuilder(primary, minao, C));

    local->set_print(options.get_int("PRINT"));
    local->set_debug(options.get_int("DEBUG"));
    local->set_bench(options.get_int("BENCH"));
    local->set_convergence(options.get_double("LOCAL_CONVERGENCE"));
    local->set_maxiter(options.get_int("LOCAL_MAXITER"));
    local->set_use_ghosts(options.get_bool("LOCAL_USE_GHOSTS"));
    local->set_condition(options.get_double("LOCAL_IBO_CONDITION"));
    local->set_power(options.get_double("LOCAL_IBO_POWER"));
    local->set_use_stars(options.get_bool("LOCAL_IBO_USE_STARS"));
    local->set_stars_completeness(options.get_double("LOCAL_IBO_STARS_COMPLETENESS"));

    std::vector<int> stars;
    for (size_t ind = 0; ind < options["LOCAL_IBO_STARS"].size(); ind++) {
        stars.push_back(options["LOCAL_IBO_STARS"][ind].to_integer() - 1);
    }
    local->set_stars(stars);

    return local;
}

std::map<std::string, psi::SharedMatrix> IAOBuilder::build_iaos() {
    // => Ghosting <= //
    std::shared_ptr<psi::Molecule> mol = minao_->molecule();
    true_atoms_.clear();
    true_iaos_.clear();
    iaos_to_atoms_.clear();
    for (int A = 0; A < mol->natom(); A++) {
        if (!use_ghosts_ && mol->Z(A) == 0.0)
            continue;
        size_t Atrue = true_atoms_.size();
        int nPshells = minao_->nshell_on_center(A);
        int sPshells = minao_->shell_on_center(A, 0);
        for (int P = sPshells; P < sPshells + nPshells; P++) {
            int nP = minao_->shell(P).nfunction();
            int oP = minao_->shell(P).function_index();
            for (int p = 0; p < nP; p++) {
                true_iaos_.push_back(p + oP);
                iaos_to_atoms_.push_back(Atrue);
            }
        }
        true_atoms_.push_back(A);
    }

    // => Overlap Integrals <= //

    std::shared_ptr<IntegralFactory> fact11(
        new IntegralFactory(primary_, primary_, primary_, primary_));
    std::shared_ptr<IntegralFactory> fact12(
        new IntegralFactory(primary_, minao_, primary_, minao_));
    std::shared_ptr<IntegralFactory> fact22(new IntegralFactory(minao_, minao_, minao_, minao_));

    std::shared_ptr<OneBodyAOInt> ints11(fact11->ao_overlap());
    std::shared_ptr<OneBodyAOInt> ints12(fact12->ao_overlap());
    std::shared_ptr<OneBodyAOInt> ints22(fact22->ao_overlap());

    psi::SharedMatrix S11(new psi::Matrix("S11", primary_->nbf(), primary_->nbf()));
    psi::SharedMatrix S12f(new psi::Matrix("S12f", primary_->nbf(), minao_->nbf()));
    psi::SharedMatrix S22f(new psi::Matrix("S22f", minao_->nbf(), minao_->nbf()));

    ints11->compute(S11);
    ints12->compute(S12f);
    ints22->compute(S22f);

    ints11.reset();
    ints12.reset();
    ints22.reset();

    fact11.reset();
    fact12.reset();
    fact22.reset();

    // => Ghosted Overlap Integrals <= //

    psi::SharedMatrix S12(new psi::Matrix("S12", primary_->nbf(), true_iaos_.size()));
    psi::SharedMatrix S22(new psi::Matrix("S22", true_iaos_.size(), true_iaos_.size()));

    double** S12p = S12->pointer();
    double** S12fp = S12f->pointer();
    for (int m = 0; m < primary_->nbf(); m++) {
        for (size_t p = 0; p < true_iaos_.size(); p++) {
            S12p[m][p] = S12fp[m][true_iaos_[p]];
        }
    }

    double** S22p = S22->pointer();
    double** S22fp = S22f->pointer();
    for (size_t p = 0; p < true_iaos_.size(); p++) {
        for (size_t q = 0; q < true_iaos_.size(); q++) {
            S22p[p][q] = S22fp[true_iaos_[p]][true_iaos_[q]];
        }
    }

    // => Metric Inverses <= //

    psi::SharedMatrix S11_m12(S11->clone());
    psi::SharedMatrix S22_m12(S22->clone());
    S11_m12->copy(S11);
    S22_m12->copy(S22);
    S11_m12->power(-1.0 / 2.0, condition_);
    S22_m12->power(-1.0 / 2.0, condition_);

    // => Tilde C <= //

    psi::SharedMatrix C = C_;
    psi::SharedMatrix T1 = psi::Matrix::doublet(S22_m12, S12, false, true);
    psi::SharedMatrix T2 =
        psi::Matrix::doublet(S11_m12, psi::Matrix::triplet(T1, T1, C, true, false, false), false, false);
    psi::SharedMatrix T3 = psi::Matrix::doublet(T2, T2, true, false);
    T3->power(-1.0 / 2.0, condition_);
    psi::SharedMatrix Ctilde = psi::Matrix::triplet(S11_m12, T2, T3, false, false, false);

    // => D and Tilde D <= //

    psi::SharedMatrix D = psi::Matrix::doublet(C, C, false, true);
    psi::SharedMatrix Dtilde = psi::Matrix::doublet(Ctilde, Ctilde, false, true);

    // => A (Before Orthogonalization) <= //

    psi::SharedMatrix DSDtilde = psi::Matrix::triplet(D, S11, Dtilde, false, false, false);
    DSDtilde->scale(2.0);

    psi::SharedMatrix L = psi::Matrix::doublet(S11_m12, S11_m12, false, false); // TODO: Possibly Unstable
    L->add(DSDtilde);
    L->subtract(D);
    L->subtract(Dtilde);

    psi::SharedMatrix AN = psi::Matrix::doublet(L, S12, false, false);

    // => A (After Orthogonalization) <= //

    psi::SharedMatrix V = psi::Matrix::triplet(AN, S11, AN, true, false, false);
    V->power(-1.0 / 2.0, condition_);

    psi::SharedMatrix A = psi::Matrix::doublet(AN, V, false, false);

    // => Assignment <= //

    S_ = S11;
    A_ = A;

    psi::SharedMatrix Acoeff(A->clone());
    psi::SharedMatrix S_min(S22->clone());
    psi::SharedMatrix L_clone(L->clone());

    std::vector<std::vector<int>> minao_inds;
    for (size_t A = 0; A < true_atoms_.size(); A++) {
        std::vector<int> vec;
        for (size_t m = 0; m < iaos_to_atoms_.size(); m++) {
            if (iaos_to_atoms_[m] == static_cast<int>(A)) {
                vec.push_back(m);
            }
        }
        minao_inds.push_back(vec);
    }

    int nocc = L->rowspi()[0];

    std::vector<int> ranges;
    ranges.push_back(0);
    ranges.push_back(nocc);

    std::vector<std::pair<int, int>> rot_inds;
    for (size_t ind = 0; ind < ranges.size() - 1; ind++) {
        int start = ranges[ind];
        int stop = ranges[ind + 1];
        for (int i = start; i < stop; i++) {
            for (int j = start; j < i; j++) {
                rot_inds.push_back(std::pair<int, int>(i, j));
            }
        }
    }

    // Build projection matrix U

    psi::SharedMatrix Cinv(C->clone());
    Cinv->invert();
    psi::SharedMatrix U = psi::Matrix::doublet(Cinv, Ctilde, false, false);

    std::map<std::string, psi::SharedMatrix> ret;
    ret["A"] = Acoeff;
    ret["S_min"] = S_min;
    ret["U"] = U;
    // print_IAO(Acoeff,nmin,primary_->nbf()); Function I envision
    // ret["A"] = set_name("A")
    // ret["S_min"] = set_name("S_min")

    return ret;
}

std::vector<std::string> IAOBuilder::print_IAO(psi::SharedMatrix A_, int nmin, int nbf,
                                               psi::SharedWavefunction wfn_) {
    CubeProperties cube = CubeProperties(wfn_);
    std::shared_ptr<psi::Molecule> mol = minao_->molecule();
    std::vector<int> iao_inds;
    psi::SharedMatrix A_nbf =
        std::make_shared<psi::Matrix>("IAO coefficient matrix in nbf dimensions", nbf, nbf);
    for (int i = 0; i < nbf; ++i) {
        for (int j = 0; j < minao_->nbf(); ++j) {
            A_nbf->set(i, j, A_->get(i, j));
        }
    }
    // Form a map that lists all functions on a given atom and with a given ang. momentum
    std::map<std::tuple<int, int, int, int, int>, std::vector<int>> atom_am_to_f_minao;
    std::map<std::tuple<int, int, int, int, int>, std::vector<int>> atom_am_to_f_primary;
    std::vector<std::tuple<std::string, double>> all_basis_conts;
    int sum = 0;
    int count_iao = 0;
    for (int A = 0; A < mol->natom(); A++) {
        int principal_qn = 0;
        int n_shell = minao_->nshell_on_center(A);
        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = minao_->shell(A, Q);
            int nfunction = shell.nfunction();
            int am = shell.am();
            if (am == 0) {
                principal_qn = principal_qn + 1;
            }
            for (int p = sum; p < sum + nfunction; ++p) {
                std::tuple<int, int, int, int, int> atom_am;
                atom_am = std::make_tuple(A, am, (principal_qn), count_iao, p - sum);
                atom_am_to_f_minao[atom_am].push_back(p);
            }
            count_iao++;
            sum += nfunction;
        }
    }

    std::vector<std::string> l_to_symbol{"s", "p", "d", "f", "g", "h"};
    std::vector<std::vector<std::string>> m_to_symbol{
        {""}, {"z", "x", "y"}, {"Z2", "XZ", "YZ", "X2Y2", "XY"}};

    std::vector<std::tuple<int, int, int, int, int>> keys;
    for (auto& kv : atom_am_to_f_minao) {
        keys.push_back(kv.first);
    }

    int sum2 = 0;
    int count_nbf = 0;
    for (int A = 0; A < mol->natom(); A++) {
        int principal_qn = 0;
        int n_shell = primary_->nshell_on_center(A);
        for (int Q = 0; Q < n_shell; Q++) {
            const GaussianShell& shell = primary_->shell(A, Q);
            int nfunction = shell.nfunction();
            int am = shell.am();
            if (am == 0) {
                principal_qn = principal_qn + 1;
            }
            for (int p = sum2; p < sum2 + nfunction; ++p) {
                std::tuple<int, int, int, int, int> atom_am;
                atom_am = std::make_tuple(A, am, (principal_qn), count_nbf, p - sum2);
                atom_am_to_f_primary[atom_am].push_back(p);
            }
            count_nbf++;
            sum2 += nfunction;
        }
    }

    std::vector<std::tuple<int, int, int, int, int>> keys_primary;
    std::vector<std::tuple<int, double, std::string>> all_iao_contributions;
    std::vector<std::string> duplicates_iao;
    for (auto& kv_primary : atom_am_to_f_primary) {
        keys_primary.push_back(kv_primary.first);
    }
    std::vector<std::string> iao_labs;
    for (auto& i : keys) {
        auto& ifn = atom_am_to_f_minao[i];
        for (auto& k : keys_primary) {
            for (auto& iao : ifn) {
                auto& ifn_primary = atom_am_to_f_primary[k];
                for (auto& nbf_primary : ifn_primary) {
                    int num = iao;
                    std::string outstr_primary = boost::str(
                        boost::format("%d%s%s%s_%d") % (std::get<0>(k) + 1) %
                        mol->symbol(std::get<0>(k)).c_str() % l_to_symbol[std::get<1>(k)].c_str() %
                        m_to_symbol[std::get<1>(k)][std::get<4>(k)].c_str() % num);
                    // iao_labs.push_back(outstr);
                    double a = 0.0;
                    a = A_nbf->get(nbf_primary, iao);
                    // outfile->Printf("%s: %.5f \n", outstr_primary.c_str(), a);
                    std::tuple<std::string, double> base;
                    std::tuple<int, double, std::string> iao_cont;
                    base = std::make_tuple(outstr_primary.c_str(), a);
                    iao_cont = std::make_tuple(num, a, outstr_primary.c_str());
                    all_basis_conts.push_back(base);
                    all_iao_contributions.push_back(iao_cont);
                }

                std::string outstr = boost::str(
                    boost::format("%d%s%s%s_%d") % (std::get<0>(i) + 1) %
                    mol->symbol(std::get<0>(i)).c_str() % l_to_symbol[std::get<1>(i)].c_str() %
                    m_to_symbol[std::get<1>(i)][std::get<4>(i)].c_str() % iao);
                std::string istring = outstr;
                if (std::find(duplicates_iao.begin(), duplicates_iao.end(), istring.c_str()) !=
                    duplicates_iao.end()) {
                } else {
                    //  outfile->Printf("%s\n", outstr.c_str());
                    // iao_labs.push_back(outstr);
                }
                duplicates_iao.push_back(istring.c_str());
            }
        }
    }

    // for (int ind = 0; ind < nmin ; ++ind){
    //    iao_inds.push_back(ind);
    // }
    // cube.compute_orbitals(A_nbf, iao_inds,iao_labs, "iao");

    std::vector<std::string> duplicates;
    int basis_conts_size = all_basis_conts.size();
    int all_iao_size = all_iao_contributions.size();
    std::vector<std::tuple<int, double, std::string>> iao_sum;
    std::vector<std::tuple<int, std::string>> iao_sum_final;
    for (int i = 0; i < all_iao_size; ++i) {
        double total_basis_cont = 0.0;
        std::string istring = std::get<2>(all_iao_contributions[i]);
        if (std::find(duplicates.begin(), duplicates.end(), istring.c_str()) != duplicates.end()) {
        } else {
            for (int j = 0; j < basis_conts_size; ++j) {
                std::string jstring = std::get<0>(all_basis_conts[j]);
                if (istring == jstring) {
                    total_basis_cont += std::abs(std::get<1>(all_iao_contributions[j]));
                }
            }
        }
        duplicates.push_back(istring.c_str());
        if (total_basis_cont > 0.001) {
            //      outfile->Printf("SUM(%s): %.2f \n",istring.c_str(),total_basis_cont);
            std::tuple<int, double, std::string> iao_sum_cont;
            iao_sum_cont = std::make_tuple(std::get<0>(all_iao_contributions[i]), total_basis_cont,
                                           std::get<2>(all_iao_contributions[i]).c_str());
            iao_sum.push_back(iao_sum_cont);
            // outfile->Printf("Saved Tuple:
            // (%d,%.2f,%s)\n",all_iao_contributions[i].get<0>(),total_basis_cont,all_iao_contributions[i].get<2>().c_str());
        }
    }

    std::sort(iao_sum.begin(), iao_sum.end());
    int iao_sum_size = iao_sum.size();
    for (int i = 0; i < nmin; ++i) {
        std::vector<std::tuple<double, int, std::string>> iao_max_contributions;
        for (int j = 0; j < iao_sum_size; ++j) {
            std::vector<double> max_candidates;
            if (i == std::get<0>(iao_sum[j])) {
                // outfile->Printf("for i=%d -> (%s)\n", iao_sum[j].get<0>(),
                // iao_sum[j].get<2>().c_str());
                std::tuple<double, int, std::string> iao_max_candidate;
                iao_max_candidate =
                    std::make_tuple(std::get<1>(iao_sum[j]), std::get<0>(iao_sum[j]),
                                    std::get<2>(iao_sum[j]).c_str());
                iao_max_contributions.push_back(iao_max_candidate);
            } else {
            }
            // outfile->Printf("Saved Tuple:
            // (%d,%.2f,%s)\n",iao_sum[i].get<0>(),iao_sum[i].get<1>(),iao_sum[i].get<2>().c_str());
        }
        std::sort(iao_max_contributions.begin(), iao_max_contributions.end());
        std::reverse(iao_max_contributions.begin(), iao_max_contributions.end());
        outfile->Printf("IAO%d -> %s(%.2f) \n", std::get<1>(iao_max_contributions[0]),
                        std::get<2>(iao_max_contributions[0]).c_str(),
                        std::get<0>(iao_max_contributions[0]));
        iao_labs.push_back(std::get<2>(iao_max_contributions[0]).c_str());
    }

    for (int ind = 0; ind < nmin; ++ind) {
        iao_inds.push_back(ind);
    }
    cube.compute_orbitals(A_nbf, iao_inds, iao_labs, "iao");

    // A_->print();
    // A_nbf->print();
    return iao_labs;
}

std::map<std::string, psi::SharedMatrix>
IAOBuilder::ibo_localizer(psi::SharedMatrix L, const std::vector<std::vector<int>>& minao_inds,
                          const std::vector<std::pair<int, int>>& rot_inds, double convergence,
                          int maxiter, int power) {
    int nmin = L->colspi()[0];
    int nocc = L->rowspi()[0];

    psi::SharedMatrix L2(L->clone());
    L2->copy(L);
    double** Lp = L2->pointer();

    psi::SharedMatrix U(new psi::Matrix("U", nocc, nocc));
    U->identity();
    double** Up = U->pointer();

    bool converged = false;

    if (power != 2 && power != 4)
        throw psi::PSIEXCEPTION("IAO: Invalid metric power.");

    outfile->Printf("    @IBO %4s: %24s %14s\n", "Iter", "Metric", "Gradient");

    for (int iter = 1; iter <= maxiter; iter++) {

        double metric = 0.0;
        for (int i = 0; i < nocc; i++) {
            for (size_t A = 0; A < minao_inds.size(); A++) {
                double Lval = 0.0;
                for (size_t m = 0; m < minao_inds[A].size(); m++) {
                    int mind = minao_inds[A][m];
                    Lval += Lp[i][mind] * Lp[i][mind];
                }
                metric += pow(Lval, power);
            }
        }
        metric = pow(metric, 1.0 / power);

        double gradient = 0.0;
        for (size_t ind = 0; ind < rot_inds.size(); ind++) {
            int i = rot_inds[ind].first;
            int j = rot_inds[ind].second;

            double Aij = 0.0;
            double Bij = 0.0;
            for (size_t A = 0; A < minao_inds.size(); A++) {
                double Qii = 0.0;
                double Qij = 0.0;
                double Qjj = 0.0;
                for (size_t m = 0; m < minao_inds[A].size(); m++) {
                    int mind = minao_inds[A][m];
                    Qii += Lp[i][mind] * Lp[i][mind];
                    Qij += Lp[i][mind] * Lp[j][mind];
                    Qjj += Lp[j][mind] * Lp[j][mind];
                }
                if (power == 2) {
                    Aij += 4.0 * Qij * Qij - (Qii - Qjj) * (Qii - Qjj);
                    Bij += 4.0 * Qij * (Qii - Qjj);
                } else {
                    Aij += (-1.0) * Qii * Qii * Qii * Qii - Qjj * Qjj * Qjj * Qjj +
                           6.0 * (Qii * Qii + Qjj * Qjj) * Qij * Qij + Qii * Qii * Qii * Qjj +
                           Qii * Qjj * Qjj * Qjj;
                    Bij += 4.0 * Qij * (Qii * Qii * Qii - Qjj * Qjj * Qjj);
                }
            }
            double phi = 0.25 * atan2(Bij, -Aij);
            double c = cos(phi);
            double s = sin(phi);

            C_DROT(nmin, Lp[i], 1, Lp[j], 1, c, s);
            C_DROT(nocc, Up[i], 1, Up[j], 1, c, s);

            gradient += Bij * Bij;
        }
        gradient = sqrt(gradient);

        outfile->Printf("    @IBO %4d: %24.16E %14.6E\n", iter, metric, gradient);

        if (gradient < convergence) {
            converged = true;
            break;
        }
    }
    outfile->Printf("\n");
    if (converged) {
        outfile->Printf("    IBO Localizer converged.\n\n");
    } else {
        outfile->Printf("    IBO Localizer failed.\n\n");
    }

    U->transpose_this();
    // L2->transpose_this();
    std::map<std::string, psi::SharedMatrix> ret;
    ret["U"] = U;
    ret["L"] = L2;
    psi::SharedMatrix L_local(L2->clone());
    // ret["U"]->set_name("U");
    // ret["L"]->set_name("L");

    return ret;
}

std::map<std::string, psi::SharedMatrix> IAOBuilder::localize(psi::SharedMatrix Cocc, psi::SharedMatrix Focc,
                                                         const std::vector<int>& ranges2) {
    if (!A_)
        build_iaos();

    std::vector<int> ranges = ranges2;
    if (!ranges.size()) {
        ranges.push_back(0);
        ranges.push_back(Cocc->colspi()[0]);
    }

    std::vector<std::vector<int>> minao_inds;
    for (size_t A = 0; A < true_atoms_.size(); A++) {
        std::vector<int> vec;
        for (size_t m = 0; m < iaos_to_atoms_.size(); m++) {
            if (iaos_to_atoms_[m] == static_cast<int>(A)) {
                vec.push_back(m);
            }
        }
        minao_inds.push_back(vec);
    }

    std::vector<std::pair<int, int>> rot_inds;
    for (size_t ind = 0; ind < ranges.size() - 1; ind++) {
        int start = ranges[ind];
        int stop = ranges[ind + 1];
        for (int i = start; i < stop; i++) {
            for (int j = start; j < i; j++) {
                rot_inds.push_back(std::pair<int, int>(i, j));
            }
        }
    }

    psi::SharedMatrix L = psi::Matrix::triplet(Cocc, S_, A_, true, false, false);
    // L->set_name("L");

    std::map<std::string, psi::SharedMatrix> ret1 =
        IAOBuilder::ibo_localizer(L, minao_inds, rot_inds, convergence_, maxiter_, power_);
    L = ret1["L"];
    psi::SharedMatrix L_local(L->clone());
    outfile->Printf("Localized Matrix from ibo code! \n");
    // L_local->print();
    psi::SharedMatrix U = ret1["U"];

    if (use_stars_) {
        psi::SharedMatrix Q = orbital_charges(L);
        double** Qp = Q->pointer();
        int nocc = Q->colspi()[0];
        int natom = Q->rowspi()[0];

        std::vector<int> pi_orbs;
        for (int i = 0; i < nocc; i++) {
            std::vector<double> Qs;
            for (int A = 0; A < natom; A++) {
                Qs.push_back(fabs(Qp[A][i]));
            }
            std::sort(Qs.begin(), Qs.end(), std::greater<double>());
            double Qtot = 0.0;
            for (int A = 0; A < natom && A < 2; A++) {
                Qtot += Qs[A];
            }
            if (Qtot < stars_completeness_) {
                pi_orbs.push_back(i);
            }
        }
        std::vector<std::pair<int, int>> rot_inds2;
        for (size_t iind = 0; iind < pi_orbs.size(); iind++) {
            for (size_t jind = 0; jind < iind; jind++) {
                rot_inds2.push_back(std::pair<int, int>(pi_orbs[iind], pi_orbs[jind]));
            }
        }

        std::vector<std::vector<int>> minao_inds2;
        for (size_t Aind = 0; Aind < stars_.size(); Aind++) {
            int A = -1;
            for (size_t A2 = 0; A2 < true_atoms_.size(); A2++) {
                if (stars_[Aind] == true_atoms_[A2]) {
                    A = A2;
                    break;
                }
            }
            if (A == -1)
                continue;
            std::vector<int> vec;
            for (size_t m = 0; m < iaos_to_atoms_.size(); m++) {
                if (iaos_to_atoms_[m] == A) {
                    vec.push_back(m);
                }
            }
            minao_inds2.push_back(vec);
        }

        outfile->Printf("    *** Stars Procedure ***\n\n");
        outfile->Printf("    Pi Completeness = %11.3f\n", stars_completeness_);
        outfile->Printf("    Number of Pis   = %11zu\n", pi_orbs.size());
        outfile->Printf("    Number of Stars = %11zu\n", stars_.size());
        outfile->Printf("    Star Centers: ");
        for (size_t ind = 0; ind < stars_.size(); ind++) {
            outfile->Printf("%3d ", stars_[ind] + 1);
        }
        outfile->Printf("\n\n");

        std::map<std::string, psi::SharedMatrix> ret2 =
            IAOBuilder::ibo_localizer(L, minao_inds2, rot_inds2, convergence_, maxiter_, power_);
        L = ret2["L"];
        psi::SharedMatrix U3 = ret2["U"];
        U = psi::Matrix::doublet(U, U3, false, false);

        std::map<std::string, psi::SharedMatrix> ret3 =
            IAOBuilder::ibo_localizer(L, minao_inds, rot_inds, convergence_, maxiter_, power_);
        L = ret3["L"];
        psi::SharedMatrix U4 = ret3["U"];
        U = psi::Matrix::doublet(U, U4, false, false);

        // => Analysis <= //

        Q = orbital_charges(L);
        Qp = Q->pointer();

        pi_orbs.clear();
        for (int i = 0; i < nocc; i++) {
            std::vector<double> Qs;
            for (int A = 0; A < natom; A++) {
                Qs.push_back(fabs(Qp[A][i]));
            }
            std::sort(Qs.begin(), Qs.end(), std::greater<double>());
            double Qtot = 0.0;
            for (int A = 0; A < natom && A < 2; A++) {
                Qtot += Qs[A];
            }
            if (Qtot < stars_completeness_) {
                pi_orbs.push_back(i);
            }
        }

        std::vector<int> centers;
        for (size_t i2 = 0; i2 < pi_orbs.size(); i2++) {
            int i = pi_orbs[i2];
            int ind = 0;
            for (int A = 0; A < natom; A++) {
                if (fabs(Qp[A][i]) >= fabs(Qp[ind][i])) {
                    ind = A;
                }
            }
            centers.push_back(ind);
        }
        std::sort(centers.begin(), centers.end());

        outfile->Printf("    *** Stars Analysis ***\n\n");
        outfile->Printf("    Pi Centers: ");
        for (size_t ind = 0; ind < centers.size(); ind++) {
            outfile->Printf("%3d ", centers[ind] + 1);
        }
        outfile->Printf("\n\n");
    }

    psi::SharedMatrix Focc2 = psi::Matrix::triplet(U, Focc, U, true, false, false);
    psi::SharedMatrix U2 = IAOBuilder::reorder_orbitals(Focc2, ranges);

    psi::SharedMatrix Uocc3 = psi::Matrix::doublet(U, U2, false, false);
    psi::SharedMatrix Focc3 = psi::Matrix::triplet(Uocc3, Focc, Uocc3, true, false, false);
    psi::SharedMatrix Locc3 = psi::Matrix::doublet(Cocc, Uocc3, false, false);
    L = psi::Matrix::doublet(U2, L, true, false);
    psi::SharedMatrix Q = orbital_charges(L);

    std::map<std::string, psi::SharedMatrix> ret;
    ret["L"] = Locc3;
    ret["L_local"] = L_local;
    ret["U"] = Uocc3;
    ret["F"] = Focc3;
    ret["Q"] = Q;

    // ret["L"]->set_name("L");
    // ret["U"]->set_name("U");
    // ret["F"]->set_name("F");
    // ret["Q"]->set_name("Q");

    return ret;
}

psi::SharedMatrix IAOBuilder::reorder_orbitals(psi::SharedMatrix F, const std::vector<int>& ranges) {
    int nmo = F->rowspi()[0];
    double** Fp = F->pointer();

    psi::SharedMatrix U(new psi::Matrix("U", nmo, nmo));
    double** Up = U->pointer();

    for (size_t ind = 0; ind < ranges.size() - 1; ind++) {
        int start = ranges[ind];
        int stop = ranges[ind + 1];
        std::vector<std::pair<double, int>> fvals;
        for (int i = start; i < stop; i++) {
            fvals.push_back(std::pair<double, int>(Fp[i][i], i));
        }
        std::sort(fvals.begin(), fvals.end());
        for (int i = start; i < stop; i++) {
            Up[i][fvals[i - start].second] = 1.0;
        }
    }

    return U;
}

psi::SharedMatrix IAOBuilder::orbital_charges(psi::SharedMatrix L) {
    double** Lp = L->pointer();
    int nocc = L->rowspi()[0];
    int nmin = L->colspi()[0];
    int natom = true_atoms_.size();

    psi::SharedMatrix Q(new psi::Matrix("Q", natom, nocc));
    double** Qp = Q->pointer();

    for (int i = 0; i < nocc; i++) {
        for (int m = 0; m < nmin; m++) {
            Qp[iaos_to_atoms_[m]][i] += Lp[i][m] * Lp[i][m];
        }
    }

    return Q;
}
}
