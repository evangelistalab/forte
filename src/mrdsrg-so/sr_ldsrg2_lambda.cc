/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <algorithm>
#include <map>
#include <vector>
#include <unistd.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "mrdsrg_so.h"

using namespace psi;

namespace forte {

void MRDSRG_SO::compute_lambda() {
    Tbar1 = BTF_->build(tensor_type_, "Tbar1", {"cv"});
    Tbar2 = BTF_->build(tensor_type_, "Tbar2", {"ccvv"});

    // lambda equation
    BlockedTensor C1 = BTF_->build(tensor_type_, "C1 Lambda", {"cv"});
    BlockedTensor C2 = BTF_->build(tensor_type_, "C2 Lambda", {"ccvv"});

    // start iteration
    outfile->Printf("\n\n  ==> Lambda Iterations <==\n");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n           Cycle     |L1|    |L2|  |L1 - T1|  |L2 - T2|  max(L1) max(L2) ");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    int maxiter = foptions_->get_int("MAXITER");
    int cycle = 0;
    T1norm = 0.0;
    T2norm = 0.0;
    T1max = 0.0;
    T2max = 0.0;
    Tbar1_diff = 0.0;
    Tbar2_diff = 0.0;

    for (int i = 0; i < maxiter; ++i) {
        C1.zero();
        C2.zero();

        build_lambda_numerical(C1, C2, i);

        //        compute_lambda_comm1(F, V, Tbar1, Tbar2, C1, C2);
        //        compute_lambda_comm2(F, V, T1, T2, Tbar1, Tbar2, C1, C2);
        //        compute_lambda_comm3(F, V, T1, T2, Tbar1, Tbar2, C1, C2);
        //        compute_lambda_comm4(F, V, T1, T2, Tbar1, Tbar2, C1, C2);
        //        compute_lambda_comm5(F, V, T1, T2, C1, C2);

        outfile->Printf("\n      @CT %4d %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e", cycle, T1norm,
                        T2norm, Tbar1_diff, Tbar2_diff, T1max, T2max);

        update_lambda(C1, C2);

        double rms = std::max(rms_t1, rms_t2);
        if (rms < foptions_->get_double("R_CONVERGENCE")) {
            break;
        }
        cycle++;
    }

    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    if (cycle == maxiter) {
        outfile->Printf("\n\n\tThe calculation did not converge in %d "
                        "cycles\n\tQuitting.\n",
                        maxiter);
    }

    // revmove all lambda files on disk
    for (const std::string& filename : std::unordered_set<std::string> (lambda_files_)) {
        if (remove(filename.c_str()) != 0) {
            std::stringstream ss;
            ss << "Error deleting file " << filename << ": No such file or directory";
            throw psi::PSIEXCEPTION(ss.str());
        }
        lambda_files_.erase(filename);
    }
}

void MRDSRG_SO::build_lambda_numerical(BlockedTensor& C1, BlockedTensor& C2, int iter) {
    // 4-point formula: f'(x) = [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / 12h
    double h = foptions_->get_double("DSRG_LAMBDA_FINDIFF_STEPSIZE");
    int npt = foptions_->get_int("DSRG_LAMBDA_FINDIFF_FORM");
    std::vector<int> factors {1, -1};
    if (npt >= 4) {
        factors = std::vector<int> {2, 1, -1, -2};
    }

    BlockedTensor O1 = ambit::BlockedTensor::build(tensor_type_, "O1", {"cv"});
    BlockedTensor O2 = ambit::BlockedTensor::build(tensor_type_, "O2", {"ccvv"});

    auto ST1 = ambit::BlockedTensor::build(ambit::CoreTensor, "scaled t1", {"cv"});
    auto ST2 = ambit::BlockedTensor::build(ambit::CoreTensor, "scaled t2", {"ccvv"});

    size_t nc_nmo = acore_sos.size();
    size_t nv_nmo = avirt_sos.size();

    auto acore_irrep = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    auto avirt_irrep = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

    std::string path0 = PSIOManager::shared_object()->get_default_path() + "psi." +
                        std::to_string(getpid()) + "." +
                        psi::Process::environment.molecule()->name();

    for (size_t i = 0; i < nc_nmo; ++i) {
        auto h_i = std::get<0>(acore_irrep[i]);
        for (size_t a = 0; a < nv_nmo; ++a) {
            auto h_a = std::get<0>(avirt_irrep[a]);
            if(h_i ^ h_a) {
                continue;
            }

            outfile->Printf("\n  working on %2zu -> %2zu:", i, a);

            size_t idx = i * nv_ + a;
            double t = T1.block("cv").data()[idx];

            std::vector<double> pts;
            std::vector<double> diag;
            for (int factor: factors) {
                // filenames for dump and read
                std::string step = "step" + std::to_string(factor);
                std::string key_Hbar0 = std::to_string(i) + "." + std::to_string(a) + "." + step;
                std::string filename1 = path0 + ".T1L1." + key_Hbar0 + ".bin";
                std::string filename2 = path0 + ".T1L2." + key_Hbar0 + ".bin";

                if (iter != 0) {
                    // read from saved data or disk
                    Hbar0 = lambda_Hbar0_[key_Hbar0];
                    read_disk_vector_double(filename1, O1.block("cv").data());
                    read_disk_vector_double(filename2, O2.block("ccvv").data());
                } else {
                    T1.block("cv").data()[idx] = t + factor * h;
                    compute_lhbar();

                    O1["ia"] = Hbar1["ia"];
                    O2["ijab"] = Hbar2["ijab"];

                    O1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                        double D = Fd[i[0]] - Fd[i[1]];
                        value *= 1.0 - std::exp(-s_ * D * D);
                    });

                    O2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                        double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
                        value *= 1.0 - std::exp(-s_ * D * D);
                    });

                    ST1["ia"] = T1["ia"];
                    ST2["ijab"] = T2["ijab"];
                    ST1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                        double D = Fd[i[0]] - Fd[i[1]];
                        value *= D * std::exp(-s_ * D * D);
                    });
                    ST2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                        double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
                        value *= D * std::exp(-s_ * D * D);
                    });

                    O1["ia"] -= ST1["ia"];
                    O2["ijab"] -= ST2["ijab"];

                    // save data or dump to disk
                    lambda_Hbar0_[key_Hbar0] = Hbar0;

                    lambda_files_.insert(filename1);
                    write_disk_vector_double(filename1, O1.block("cv").data());

                    lambda_files_.insert(filename2);
                    write_disk_vector_double(filename2, O2.block("ccvv").data());
                }

                // zero diagonal element
                double diag_elem = O1.block("cv").data()[idx];
                O1.block("cv").data()[idx] = 0.0;
                diag.push_back(diag_elem);

                // add lambda contribution
                Hbar0 += O1["ia"] * Tbar1["ia"];
                Hbar0 += 0.25 * O2["ijab"] * Tbar2["ijab"];

                pts.push_back(Hbar0);
                outfile->Printf(" X");
            }
            T1.block("cv").data()[idx] = t;

            double grad, diag_grad;
            if (npt >= 4) {
                grad = (-pts[0] + 8 * pts[1] - 8 * pts[2] + pts[3]) / (12 * h);
                diag_grad = (-diag[0] + 8 * diag[1] - 8 * diag[2] + diag[3]) / (12 * h);
            } else {
                grad = (pts[0] - pts[1]) / (2 * h);
                diag_grad = (diag[0] - diag[1]) / (2 * h);
            }

            C1.block("cv").data()[idx] = grad / (-diag_grad);
            C1.block("cv").data()[(i + nc_nmo) * nv_ + (a + nv_nmo)] = grad / (-diag_grad);

            double d = Fd[acore_sos[i]] - Fd[avirt_sos[a]];
            double ref_nu = 2.0 * (F.block("cv").data()[idx] - d * T1.block("cv").data()[idx]);
            double ref_de = -d;
            double ref = ref_nu / (-ref_de);
            outfile->Printf("\n ref = %20.15f = %20.15f / %20.15f", ref, ref_nu, ref_de);
            outfile->Printf("\n val = %20.15f = %20.15f / %20.15f", C1.block("cv").data()[idx], grad, diag_grad);
        }
    }

    for (size_t i = 0; i < nc_nmo; ++i) {
        auto h_i = std::get<0>(acore_irrep[i]);
        for (size_t j = 0; j < nc_nmo; ++j) {
            auto h_j = std::get<0>(acore_irrep[j]);
            for (size_t a = 0; a < nv_nmo; ++a) {
                auto h_a = std::get<0>(avirt_irrep[a]);
                for (size_t b = 0; b < nv_nmo; ++b) {
                    auto h_b = std::get<0>(avirt_irrep[b]);
                    if(h_i ^ h_j ^ h_a ^ h_b) {
                        continue;
                    }

                    outfile->Printf("\n  working on (%2zu,%2zu) -> (%2zu,%2zu):", i, (j + nc_nmo), a, (b + nv_nmo));

                    size_t idxx = i * nc_ * nv_ * nv_ + (j + nc_nmo) * nv_ * nv_ + a * nv_ + (b + nv_nmo);

                    double numerator = 0.0;
                    double denominator = 0.0;

                    std::vector<std::vector<size_t>> indices;
                    indices.push_back(std::vector<size_t> {i, j + nc_nmo, a, b + nv_nmo});
                    indices.push_back(std::vector<size_t> {j + nc_nmo, i, a, b + nv_nmo});
                    indices.push_back(std::vector<size_t> {j + nc_nmo, i, b + nv_nmo, a});
                    indices.push_back(std::vector<size_t> {i, j + nc_nmo, b + nv_nmo, a});

                    // need to consider permutations
                    for (int x = 0; x < 4; ++x) {
                        size_t idx = indices[x][0] * nc_ * nv_ * nv_ + indices[x][1] * nv_ * nv_ + indices[x][2] * nv_ + indices[x][3];
                        double t = T2.block("ccvv").data()[idx];

                        std::vector<double> pts;
                        std::vector<double> diag;
                        for (int factor: factors) {
                            // filenames for dump and read
                            std::string perm = "perm" + std::to_string(x);
                            std::string step = "step" + std::to_string(factor);
                            std::string key_Hbar0 = std::to_string(i) + "." + std::to_string(j)
                                    + "." + std::to_string(a) + "." + std::to_string(b)
                                    + "." + step + "." + perm;
                            std::string filename1 = path0 + ".T2L1." + key_Hbar0 + ".bin";
                            std::string filename2 = path0 + ".T2L2." + key_Hbar0 + ".bin";

                            if (iter != 0) {
                                // read from saved data or disk
                                Hbar0 = lambda_Hbar0_[key_Hbar0];
                                read_disk_vector_double(filename1, O1.block("cv").data());
                                read_disk_vector_double(filename2, O2.block("ccvv").data());
                            } else {
                                T2.block("ccvv").data()[idx] = t + factor * h;
                                compute_lhbar();

                                O1["ia"] = Hbar1["ia"];
                                O2["ijab"] = Hbar2["ijab"];

                                O1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                                    double D = Fd[i[0]] - Fd[i[1]];
                                    value *= 1.0 - std::exp(-s_ * D * D);
                                });

                                O2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                                    double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
                                    value *= 1.0 - std::exp(-s_ * D * D);
                                });

                                ST1["ia"] = T1["ia"];
                                ST2["ijab"] = T2["ijab"];
                                ST1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                                    double D = Fd[i[0]] - Fd[i[1]];
                                    value *= D * std::exp(-s_ * D * D);
                                });
                                ST2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
                                    double D = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
                                    value *= D * std::exp(-s_ * D * D);
                                });

                                O1["ia"] -= ST1["ia"];
                                O2["ijab"] -= ST2["ijab"];

                                // save data or dump to disk
                                lambda_Hbar0_[key_Hbar0] = Hbar0;

                                lambda_files_.insert(filename1);
                                write_disk_vector_double(filename1, O1.block("cv").data());

                                lambda_files_.insert(filename2);
                                write_disk_vector_double(filename2, O2.block("ccvv").data());
                            }

                            // zero diagonal element
                            double diag_elem = O2.block("ccvv").data()[idx];
                            diag_elem -= O2.block("ccvv").data()[indices[x][1] * nc_ * nv_ * nv_ + indices[x][0] * nv_ * nv_ + indices[x][2] * nv_ + indices[x][3]];
                            diag_elem -= O2.block("ccvv").data()[indices[x][0] * nc_ * nv_ * nv_ + indices[x][1] * nv_ * nv_ + indices[x][3] * nv_ + indices[x][2]];
                            diag_elem += O2.block("ccvv").data()[indices[x][1] * nc_ * nv_ * nv_ + indices[x][0] * nv_ * nv_ + indices[x][3] * nv_ + indices[x][2]];
                            diag.push_back(diag_elem);

                            O2.block("ccvv").data()[idx] = 0.0;
                            O2.block("ccvv").data()[indices[x][1] * nc_ * nv_ * nv_ + indices[x][0] * nv_ * nv_ + indices[x][2] * nv_ + indices[x][3]] = 0.0;
                            O2.block("ccvv").data()[indices[x][0] * nc_ * nv_ * nv_ + indices[x][1] * nv_ * nv_ + indices[x][3] * nv_ + indices[x][2]] = 0.0;
                            O2.block("ccvv").data()[indices[x][1] * nc_ * nv_ * nv_ + indices[x][0] * nv_ * nv_ + indices[x][3] * nv_ + indices[x][2]] = 0.0;

                            // add lambda contribution
                            Hbar0 += O1["ia"] * Tbar1["ia"];
                            Hbar0 += 0.25 * O2["ijab"] * Tbar2["ijab"];

                            pts.push_back(Hbar0);
                            outfile->Printf(" X");
                        }
                        T2.block("ccvv").data()[idx] = t;

                        double grad, diag_grad;
                        if (npt >= 4) {
                            grad = (-pts[0] + 8 * pts[1] - 8 * pts[2] + pts[3]) / (12 * h);
                            diag_grad = (-diag[0] + 8 * diag[1] - 8 * diag[2] + diag[3]) / (12 * h);
                        } else {
                            grad = (pts[0] - pts[1]) / (2 * h);
                            diag_grad = (diag[0] - diag[1]) / (2 * h);
                        }

                        numerator += std::pow(-1, x) * grad;
                        denominator += diag_grad;
                    }

                    C2.block("ccvv").data()[idxx] += numerator / (-0.25 * denominator);
                    C2.block("ccvv").data()[i * nc_ * nv_ * nv_ + (j + nc_nmo) * nv_ * nv_ + (b + nv_nmo) * nv_ + a] = -C2.block("ccvv").data()[idxx];
                    C2.block("ccvv").data()[(j + nc_nmo) * nc_ * nv_ * nv_ + i * nv_ * nv_ + a * nv_ + (b + nv_nmo)] = -C2.block("ccvv").data()[idxx];
                    C2.block("ccvv").data()[(j + nc_nmo) * nc_ * nv_ * nv_ + i * nv_ * nv_ + (b + nv_nmo) * nv_ + a] = C2.block("ccvv").data()[idxx];

                    double d = Fd[core_sos_[i]] + Fd[core_sos_[j]] - Fd[virt_sos_[a]] - Fd[virt_sos_[b]];
                    double ref_nu = 0.5 * (V.block("ccvv").data()[idxx] - d * T2.block("ccvv").data()[idxx]);
                    double ref_de = -d;
                    double ref = ref_nu / (-0.25 * ref_de);
                    outfile->Printf("\n ref = %20.15f = -4 * %20.15f / %20.15f", ref, ref_nu, ref_de);
                    outfile->Printf("\n val = %20.15f = -4 * %20.15f / %20.15f", C2.block("ccvv").data()[idxx], numerator, denominator);
                }
            }
        }
    }

    for (size_t i = 0; i < nc_nmo; ++i) {
        for (size_t j = 0; j < nc_nmo; ++j) {
            for (size_t a = 0; a < nv_nmo; ++a) {
                for (size_t b = 0; b < nv_nmo; ++b) {

                    auto ijab = C2.block("ccvv").data()[i * nc_ * nv_ * nv_ + (j + nc_nmo) * nv_ * nv_ + a * nv_ + (b + nv_nmo)];
                    auto ijba = C2.block("ccvv").data()[i * nc_ * nv_ * nv_ + (j + nc_nmo) * nv_ * nv_ + b * nv_ + (a + nv_nmo)];

                    C2.block("ccvv").data()[i * nc_ * nv_ * nv_ + j * nv_ * nv_ + a * nv_ + b] = ijab - ijba;
                    C2.block("ccvv").data()[(i + nc_nmo) * nc_ * nv_ * nv_ + (j + nc_nmo) * nv_ * nv_ + (a + nv_nmo) * nv_ + (b + nv_nmo)] = ijab - ijba;
                }
            }
        }
    }
}

void MRDSRG_SO::compute_lhbar_lambda_test(){
    Hbar0 = 0.0;
    Hbar1["pq"] = F["pq"];
    Hbar2["pqrs"] = V["pqrs"];

    Hbar1["c0,v0"] += 1.0 * F["v1,v0"] * T1["c0,v1"];
    Hbar1["c0,v0"] += -1.0 * F["c1,c0"] * T1["c1,v0"];
    Hbar1["v0,c0"] += 1.0 * F["v1,v0"] * T1["c0,v1"];
    Hbar1["v0,c0"] += -1.0 * F["c1,c0"] * T1["c1,v0"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * F["v2,v0"] * T2["c0,c1,v1,v2"];
    Hbar2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    Hbar2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    Hbar2["v0,v1,c0,c1"] += temp["c0,c1,v0,v1"];
    Hbar2["v1,v0,c0,c1"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * F["c2,c0"] * T2["c1,c2,v0,v1"];
    Hbar2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    Hbar2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    Hbar2["v0,v1,c0,c1"] += temp["c0,c1,v0,v1"];
    Hbar2["v0,v1,c1,c0"] -= temp["c0,c1,v0,v1"];

    Hbar0 += 2.0 * F["c0,v0"] * T1["c0,v0"];
    Hbar0 += (1.0 / 2.0) * V["c0,c1,v0,v1"] * T2["c0,c1,v0,v1"];
    Hbar0 += 1.0 * F["v1,v0"] * T1["c0,v0"] * T1["c0,v1"];
    Hbar0 += (1.0 / 2.0) * F["v1,v0"] * T2["c0,c1,v0,v2"] * T2["c0,c1,v1,v2"];
    Hbar0 += -1.0 * F["c1,c0"] * T1["c0,v0"] * T1["c1,v0"];
    Hbar0 += (-1.0 / 2.0) * F["c1,c0"] * T2["c0,c2,v0,v1"] * T2["c1,c2,v0,v1"];
}

void MRDSRG_SO::update_lambda(BlockedTensor& C1, BlockedTensor& C2) {
    BlockedTensor X1 = ambit::BlockedTensor::build(tensor_type_, "X1", {"cv"});
    BlockedTensor X2 = ambit::BlockedTensor::build(tensor_type_, "X2", {"ccvv"});
    X1["ia"] = C1["ia"];
    X2["ijab"] = C2["ijab"];

//    X1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
//        double delta = Fd[i[0]] - Fd[i[1]];
//        value /= delta;
//    });

//    X2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
//        double delta = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
//        value /= delta;
//    });

    T1max = X1.norm(0);
    T2max = X2.norm(0);
    T1norm = X1.norm();
    T2norm = X2.norm();

    C1["ia"] = X1["ia"];
    C2["ijab"] = X2["ijab"];
    C1["ia"] -= Tbar1["ia"];
    C2["ijab"] -= Tbar2["ijab"];
    rms_t1 = C1.norm();
    rms_t2 = C2.norm();

    C1["ia"] = X1["ia"];
    C2["ijab"] = X2["ijab"];
    C1["ia"] -= T1["ia"];
    C2["ijab"] -= T2["ijab"];
    Tbar1_diff = C1.norm();
    Tbar2_diff = C2.norm();

    Tbar1["ia"] = X1["ia"];
    Tbar2["ijab"] = X2["ijab"];
}

void MRDSRG_SO::compute_lambda_comm4(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                     BlockedTensor& T2, BlockedTensor& L1, BlockedTensor& L2,
                                     BlockedTensor& C1, BlockedTensor& C2) {
    compute_lambda_comm4_part1(H1, H2, T1, T2, L1, L2, C1);
    compute_lambda_comm4_part2(H1, H2, T1, T2, L1, L2, C2);
    compute_lambda_comm4_part3(H1, H2, T1, T2, L1, L2, C2);
}

void MRDSRG_SO::compute_lambda_comm1(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& L1,
                                     BlockedTensor& L2, BlockedTensor& C1, BlockedTensor& C2) {
    BlockedTensor temp;

    C1["c0,v0"] += 2.0 * H1["c0,v0"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v1,v2,v0,c1"] * L2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * L1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["c0,c1,v0,v1"] * L1["c1,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,c0"] * L2["c1,c2,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];

    C2["c0,c1,v0,v1"] += 2.0 * H2["c0,c1,v0,v1"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * L2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * L2["c2,c3,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * L1["c2,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,v0"] * L1["c1,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * L2["c1,c2,v1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];
}

void MRDSRG_SO::compute_lambda_comm2(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                     BlockedTensor& T2, BlockedTensor& L1, BlockedTensor& L2,
                                     BlockedTensor& C1, BlockedTensor& C2) {
    C1["c0,v0"] += 2.0 * H1["v1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += 1.0 * H1["c1,v1"] * T2["c0,c1,v0,v1"];
    C1["c0,v0"] += -2.0 * H1["c1,c0"] * T1["c1,v0"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,v0,c1"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -2.0 * H2["v1,c0,v0,c1"] * T1["c1,v1"];
    C1["c0,v0"] += 2.0 * H2["c0,c1,v0,v1"] * T1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += 1.0 * H1["v1,v0"] * L1["c1,v2"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H1["v2,v1"] * L1["c1,v1"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += -1.0 * H1["c0,v1"] * L1["c1,v0"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c0,v1"] * L1["c1,v1"] * T1["c1,v0"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c0,v1"] * L2["c1,c2,v0,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H1["c0,v1"] * L2["c1,c2,v1,v2"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += -1.0 * H1["c1,v0"] * L1["c0,v1"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v0"] * L1["c1,v1"] * T1["c0,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v0"] * L2["c0,c2,v1,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H1["c1,v0"] * L2["c1,c2,v1,v2"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v1"] * L1["c0,v1"] * T1["c1,v0"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c1,v1"] * L1["c1,v0"] * T1["c0,v1"];
    C1["c0,v0"] += -1.0 * H1["c1,c0"] * L1["c2,v1"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H1["c2,c1"] * L1["c1,v1"] * T2["c0,c2,v0,v1"];
    C1["c0,v0"] += 1.0 * H2["v1,v2,v0,c1"] * L1["c0,v1"] * T1["c1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,v2,v0,c1"] * L1["c1,v1"] * T1["c0,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c0,v0,c1"] * L1["c2,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c2,v0,c1"] * L1["c2,v2"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += -1.0 * H2["v1,c2,v0,c1"] * L2["c0,c1,v1,v2"] * T1["c2,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * L1["c1,v1"] * T2["c0,c1,v2,v3"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["v2,v3,v0,v1"] * L2["c0,c1,v2,v3"] * T1["c1,v1"];
    C1["c0,v0"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c1,v1"] * T1["c1,v2"];
    C1["c0,v0"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c1,v2"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v2,c0,v0,v1"] * L2["c1,c2,v1,v3"] * T2["c1,c2,v2,v3"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v2,c0,v0,v1"] * L2["c1,c2,v2,v3"] * T2["c1,c2,v1,v3"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L1["c0,v2"] * T1["c1,v1"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L1["c1,v1"] * T1["c0,v2"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L2["c0,c2,v2,v3"] * T2["c1,c2,v1,v3"];
    C1["c0,v0"] += 1.0 * H2["v2,c1,v0,v1"] * L2["c1,c2,v1,v3"] * T2["c0,c2,v2,v3"];
    C1["c0,v0"] += -1.0 * H2["v2,c1,v1,c0"] * L1["c2,v2"] * T2["c1,c2,v0,v1"];
    C1["c0,v0"] += -1.0 * H2["v2,c1,v1,c0"] * L2["c1,c2,v0,v1"] * T1["c2,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["v2,c2,v1,c1"] * L1["c2,v1"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["v3,c0,v1,v2"] * L2["c1,c2,v0,v3"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["v3,c0,v1,v2"] * L2["c1,c2,v1,v2"] * T2["c1,c2,v0,v3"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["v3,c1,v1,v2"] * L2["c1,c2,v1,v2"] * T2["c0,c2,v0,v3"];
    C1["c0,v0"] += 1.0 * H2["c0,c1,v0,v1"] * L1["c2,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c0,c1,v1,v2"] * L1["c2,v0"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c0,c1,v1,v2"] * L2["c1,c2,v1,v2"] * T1["c2,v0"];
    C1["c0,v0"] += -1.0 * H2["c0,c2,v0,c1"] * L1["c1,v1"] * T1["c2,v1"];
    C1["c0,v0"] += -1.0 * H2["c0,c2,v0,c1"] * L1["c2,v1"] * T1["c1,v1"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c0,c2,v0,c1"] * L2["c1,c3,v1,v2"] * T2["c2,c3,v1,v2"];
    C1["c0,v0"] += (-1.0 / 2.0) * H2["c0,c2,v0,c1"] * L2["c2,c3,v1,v2"] * T2["c1,c3,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L1["c2,v1"] * T1["c1,v0"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L2["c1,c3,v0,v2"] * T2["c2,c3,v1,v2"];
    C1["c0,v0"] += 1.0 * H2["c0,c2,v1,c1"] * L2["c2,c3,v1,v2"] * T2["c1,c3,v0,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v0,v1"] * L1["c0,v2"] * T2["c1,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v0,v1"] * L2["c1,c2,v1,v2"] * T1["c0,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["c1,c2,v1,v2"] * L1["c0,v1"] * T2["c1,c2,v0,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["c1,c2,v1,v2"] * L1["c1,v0"] * T2["c0,c2,v1,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c1,c2,v1,v2"] * L1["c1,v1"] * T2["c0,c2,v0,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * L2["c0,c1,v1,v2"] * T1["c2,v0"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c1,c2,v1,v2"] * L2["c1,c2,v0,v1"] * T1["c0,v2"];
    C1["c0,v0"] += 1.0 * H2["c1,c2,v1,c0"] * L1["c1,v0"] * T1["c2,v1"];
    C1["c0,v0"] += -1.0 * H2["c1,c2,v1,c0"] * L1["c1,v1"] * T1["c2,v0"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c2,c3,v0,c1"] * L2["c0,c1,v1,v2"] * T2["c2,c3,v1,v2"];
    C1["c0,v0"] += (1.0 / 4.0) * H2["c2,c3,v0,c1"] * L2["c2,c3,v1,v2"] * T2["c0,c1,v1,v2"];
    C1["c0,v0"] += (-1.0 / 4.0) * H2["c2,c3,v1,c1"] * L2["c2,c3,v1,v2"] * T2["c0,c1,v0,v2"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * L1["c1,v1"] * T2["c2,c3,v0,v1"];
    C1["c0,v0"] += (1.0 / 2.0) * H2["c2,c3,c0,c1"] * L2["c2,c3,v0,v1"] * T1["c1,v1"];

    C2["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,v1"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,c0,c1"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c2,v0,v1"] * L1["c2,v3"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["v2,c2,v0,v1"] * L2["c0,c1,v2,v3"] * T1["c2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c0,c1,v2,v3"] * L2["c2,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c0,c1,v2,v3"] * L2["c2,c3,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v2,c2"] * L1["c3,v2"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v2,c2"] * L2["c2,c3,v0,v1"] * T1["c3,v2"];
    C2["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c2,c3,v0,v1"] * L2["c0,c1,v2,v3"] * T2["c2,c3,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c2,c3,v0,v1"] * L2["c2,c3,v2,v3"] * T2["c0,c1,v2,v3"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c2,c3,v2,v3"] * L2["c0,c1,v2,v3"] * T2["c2,c3,v0,v1"];
    C2["c0,c1,v0,v1"] += (1.0 / 8.0) * H2["c2,c3,v2,v3"] * L2["c2,c3,v0,v1"] * T2["c0,c1,v2,v3"];

    auto temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += -2.0 * H1["v2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += -2.0 * H2["c0,c1,v0,c2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v0"] * L1["c2,v2"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,v0"] * L2["c0,c1,v1,v2"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v2"] * L1["c2,v0"] * T2["c0,c1,v1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H1["c2,v2"] * L2["c0,c1,v0,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * L1["c2,v1"] * T2["c0,c1,v2,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * L1["c2,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,v3,v0,c2"] * L2["c0,c1,v1,v2"] * T1["c2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * L2["c0,c1,v2,v3"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c2,v0,v2"] * L1["c2,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c2,v0,v2"] * L2["c0,c1,v1,v3"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,v2"] * L1["c2,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,v0,v2"] * L1["c2,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c1,v0,v2"] * L2["c2,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c0,c1,v0,v2"] * L2["c2,c3,v2,v3"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c1,v0,c2"] * L1["c3,v2"] * T2["c2,c3,v1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v0,v2"] * L2["c0,c1,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v0,v2"] * L2["c2,c3,v2,v3"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c0,c1,v0,v2"] * T2["c2,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c2,c3,v0,v2"] * T2["c0,c1,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,c0,c1"] * L1["c2,v0"] * T1["c3,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,v0"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += -2.0 * H2["v2,c0,v0,c2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H1["v2,v0"] * L1["c0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["v2,v0"] * L1["c0,v2"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c0,v0"] * L1["c2,v2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v2"] * L1["c0,v0"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c2,c0"] * L1["c1,v0"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H1["c2,c0"] * L1["c2,v0"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,v3,v0,c2"] * L1["c0,v1"] * T2["c1,c2,v2,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * L1["c0,v2"] * T2["c1,c2,v1,v3"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,c2"] * L2["c0,c2,v1,v2"] * T1["c1,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["v2,v3,v0,c2"] * L2["c0,c2,v2,v3"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,c2"] * L1["c1,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,v0,c2"] * L1["c1,v2"] * T1["c2,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,c0,v0,c2"] * L1["c2,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["v2,c0,v0,c2"] * L1["c2,v2"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c0,v0,v2"] * L1["c2,v2"] * T2["c1,c2,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v3,c0,v0,v2"] * L2["c1,c2,v1,v3"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c0,c2,v0,v2"] * L1["c1,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c0,c2,v0,v2"] * L1["c2,v2"] * T1["c1,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c0,c2,v0,v2"] * L2["c1,c3,v1,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c0,c2,v0,v2"] * L2["c2,c3,v2,v3"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v0,c2"] * L1["c3,v2"] * T2["c1,c2,v1,v2"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v0,c2"] * L2["c1,c2,v1,v2"] * T1["c3,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H2["c2,c3,v2,v3"] * L2["c0,c2,v0,v2"] * T2["c1,c3,v1,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * L1["c1,v0"] * T2["c2,c3,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * L1["c2,v0"] * T2["c1,c3,v1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * L2["c1,c2,v0,v2"] * T1["c3,v1"];
    temp["c0,c1,v0,v1"] += (1.0 / 4.0) * H2["c2,c3,v2,c0"] * L2["c2,c3,v0,v2"] * T1["c1,v1"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c0,c1,v1,v0"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
    C2["c1,c0,v1,v0"] += temp["c0,c1,v0,v1"];

    temp = ambit::BlockedTensor::build(ambit::CoreTensor, "temp", {"ccvv"});
    temp["c0,c1,v0,v1"] += 2.0 * H1["c2,c0"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -2.0 * H2["v2,c0,v0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c0,v2"] * L1["c2,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H1["c0,v2"] * L2["c1,c2,v0,v1"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (1.0 / 2.0) * H1["c2,v2"] * L1["c0,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H1["c2,v2"] * L2["c0,c2,v0,v1"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["v2,v3,v0,v1"] * L1["c0,v2"] * T1["c1,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["v2,c0,v0,v1"] * L1["c2,v3"] * T2["c1,c2,v2,v3"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c2,v0,v1"] * L1["c1,v2"] * T1["c2,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c2,v0,v1"] * L1["c2,v2"] * T1["c1,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c2,v0,v1"] * L2["c1,c3,v2,v3"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c0,c2,v0,v1"] * L2["c2,c3,v2,v3"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c0,c2,v2,v3"] * L2["c1,c3,v0,v1"] * T2["c2,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c0,c2,v2,v3"] * L2["c2,c3,v2,v3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v2,c2"] * L1["c3,v2"] * T2["c1,c2,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c0,c3,v2,c2"] * L2["c1,c2,v0,v1"] * T1["c3,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c0,c2,v0,v1"] * T2["c1,c3,v2,v3"];
    temp["c0,c1,v0,v1"] += (-1.0 / 4.0) * H2["c2,c3,v2,v3"] * L2["c0,c2,v2,v3"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * L1["c1,v2"] * T2["c2,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += 1.0 * H2["c2,c3,v2,c0"] * L1["c2,v2"] * T2["c1,c3,v0,v1"];
    temp["c0,c1,v0,v1"] += -1.0 * H2["c2,c3,v2,c0"] * L2["c1,c2,v0,v1"] * T1["c3,v2"];
    temp["c0,c1,v0,v1"] += (-1.0 / 2.0) * H2["c2,c3,v2,c0"] * L2["c2,c3,v0,v1"] * T1["c1,v2"];
    C2["c0,c1,v0,v1"] += temp["c0,c1,v0,v1"];
    C2["c1,c0,v0,v1"] -= temp["c0,c1,v0,v1"];
}

} // namespace forte
