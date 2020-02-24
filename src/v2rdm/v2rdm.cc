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

#include <fstream>
#include <iostream>

#include "boost/format.hpp"

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"

#include "v2rdm.h"

using namespace psi;

namespace forte {

struct tpdm {
    int i, j, k, l;
    double val;
};

struct dm3 {
    int i, j, k, l, m, n;
    double val;
};

V2RDM::V2RDM(psi::SharedWavefunction ref_wfn, psi::Options& options, std::shared_ptr<ForteIntegrals> ints,
             std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    print_method_banner({"V2RDM-CASSCF Interface"});
    startup();
}

V2RDM::~V2RDM() {}

void V2RDM::startup() {
    // number of MO per irrep
    nmopi_ = this->nmopi();
    nirrep_ = this->nirrep();
    fdoccpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    rdoccpi_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    active_ = mo_space_info_->dimension("ACTIVE");

    // map active absolute index to relative index
    for (size_t h = 0, offset_abs = 0, offset_rel = 0; h < nirrep_; ++h) {
        size_t nact_h = active_[h];
        for (size_t u = 0; u < nact_h; ++u) {
            size_t abs = fdoccpi_[h] + rdoccpi_[h] + u + offset_abs;
            size_t rel = u + offset_rel;
            abs_to_rel_[abs] = rel;
        }
        offset_rel += active_[h];
        offset_abs += nmopi_[h];
    }

    // read 2-pdm
    read_2pdm();

    // build opdm
    build_opdm();

    // read 3-pdm
    if (options_.get_str("THREEPDC") != "ZERO") {
        read_3pdm();
    }

    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // orbital spaces
    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");

    // write density to files
    if (options_.get_str("WRITE_DENSITY_TYPE") == "DENSITY") {
        write_density_to_file();
    }
}

void V2RDM::read_2pdm() {
    // map file names
    std::map<unsigned int, std::string> filename;
    filename[PSIF_V2RDM_D2AA] = "D2aa";
    filename[PSIF_V2RDM_D2AB] = "D2ab";
    filename[PSIF_V2RDM_D2BB] = "D2bb";

    // test if files exist
    std::string str = "Testing if 2RDM files exist";
    outfile->Printf("\n  %-45s ...", str.c_str());
    std::shared_ptr<PSIO> psio(new PSIO());
    for (const auto& file : {PSIF_V2RDM_D2AA, PSIF_V2RDM_D2AB, PSIF_V2RDM_D2BB}) {
        if (!psio->exists(file)) {
            std::string error = "V2RDM file for " + filename[file] + " does not exist";
            throw psi::PSIEXCEPTION(error);
        }
    }
    outfile->Printf("    OK.");

    // initialization of 2PDM
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nactv2 = nactv * nactv;
    size_t nactv3 = nactv * nactv2;

    // test if active orbitals are consistent in forte and v2rdm-casscf
    long int nab;
    psio_address addr_ab = PSIO_ZERO;
    psio->open(PSIF_V2RDM_D2AB, PSIO_OPEN_OLD);
    psio->read_entry(PSIF_V2RDM_D2AB, "length", (char*)&nab, sizeof(long int));

    size_t nsymgem = 0; // number of totally symmetric geminals
    for (size_t h = 0; h < nirrep_; ++h) {
        nsymgem += active_[h] * active_[h];
    }
    for (size_t n = 0; n < nsymgem; ++n) {
        tpdm d2;
        psio->read(PSIF_V2RDM_D2AB, "D2ab", (char*)&d2, sizeof(tpdm), addr_ab, &addr_ab);
        size_t l = static_cast<size_t>(d2.l);
        if (abs_to_rel_.find(l) == abs_to_rel_.end()) {
            outfile->Printf("\n  The active block of FORTE is different from "
                            "V2RDM-CASSCF.");
            outfile->Printf("\n  Please check the input file and make the "
                            "active block consistent.");
            throw psi::PSIEXCEPTION("The active block of FORTE is different from V2RDM-CASSCF.");
        }
    }
    psio->close(PSIF_V2RDM_D2AB, 1);

    // Read 2RDM
    str = "Reading 2RDMs";
    outfile->Printf("\n  %-45s ...", str.c_str());
    for (const auto& file : {PSIF_V2RDM_D2AA, PSIF_V2RDM_D2AB, PSIF_V2RDM_D2BB}) {
        ambit::Tensor D2 =
            ambit::Tensor::build(ambit::CoreTensor, filename[file], {nactv, nactv, nactv, nactv});

        long int nline;
        psio_address addr = PSIO_ZERO;
        psio->open(file, PSIO_OPEN_OLD);
        psio->read_entry(file, "length", (char*)&nline, sizeof(long int));

        for (int n = 0; n < nline; ++n) {
            tpdm d2;
            psio->read(file, filename[file].c_str(), (char*)&d2, sizeof(tpdm), addr, &addr);
            size_t i = abs_to_rel_[static_cast<size_t>(d2.i)];
            size_t j = abs_to_rel_[static_cast<size_t>(d2.j)];
            size_t k = abs_to_rel_[static_cast<size_t>(d2.k)];
            size_t l = abs_to_rel_[static_cast<size_t>(d2.l)];

            size_t idx = i * nactv3 + j * nactv2 + k * nactv + l;
            D2.data()[idx] = d2.val;
        }
        psio->close(file, 1);

        D2_.push_back(D2);
    }
    outfile->Printf("    Done.");

    // average Daa and Dbb
    if (options_.get_bool("AVG_DENS_SPIN")) {
        // reference D2aa, D2ab, D2bb to D2_ element
        ambit::Tensor& D2aa = D2_[0];
        ambit::Tensor& D2bb = D2_[2];

        str = "Averaging 2RDM AA and BB blocks";
        outfile->Printf("\n  %-45s ...", str.c_str());
        ambit::Tensor D2 =
            ambit::Tensor::build(ambit::CoreTensor, "D2avg_aa", {nactv, nactv, nactv, nactv});
        D2("pqrs") = 0.5 * D2aa("pqrs");
        D2("pqrs") += 0.5 * D2bb("pqrs");

        D2aa("pqrs") = D2("pqrs");
        D2bb("pqrs") = D2("pqrs");
        outfile->Printf("    Done.");
    }
}

void V2RDM::build_opdm() {
    std::string str = "Computing 1RDM";
    outfile->Printf("\n  %-45s ...", str.c_str());

    // initialization of OPDM
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nactv2 = nactv * nactv;
    size_t nactv3 = nactv * nactv2;
    D1a_ = ambit::Tensor::build(ambit::CoreTensor, "D1a", {nactv, nactv});
    D1b_ = ambit::Tensor::build(ambit::CoreTensor, "D1b", {nactv, nactv});

    // number of active electrons
    size_t nalfa = this->nalpha() - mo_space_info_->size("FROZEN_DOCC") -
                   mo_space_info_->size("RESTRICTED_DOCC");
    size_t nbeta = this->nbeta() - mo_space_info_->size("FROZEN_DOCC") -
                   mo_space_info_->size("RESTRICTED_DOCC");

    // reference D2aa, D2ab, D2bb to D2_ element
    ambit::Tensor& D2aa = D2_[0];
    ambit::Tensor& D2ab = D2_[1];
    ambit::Tensor& D2bb = D2_[2];

    // compute OPDM
    for (size_t u = 0; u < nactv; ++u) {
        for (size_t v = 0; v < nactv; ++v) {

            double va = 0, vb = 0;
            for (size_t x = 0; x < nactv; ++x) {
                va += D2aa.data()[u * nactv3 + x * nactv2 + v * nactv + x];
                va += D2ab.data()[u * nactv3 + x * nactv2 + v * nactv + x];

                vb += D2bb.data()[u * nactv3 + x * nactv2 + v * nactv + x];
                vb += D2ab.data()[x * nactv3 + u * nactv2 + x * nactv + v];
            }

            D1a_.data()[u * nactv + v] = va / (nalfa + nbeta - 1.0);
            D1b_.data()[u * nactv + v] = vb / (nalfa + nbeta - 1.0);
        }
    }
    outfile->Printf("    Done.");

    // average Da and Db
    if (options_.get_bool("AVG_DENS_SPIN")) {
        str = "Averaging 1RDM A and B blocks";
        outfile->Printf("\n  %-45s ...", str.c_str());
        ambit::Tensor D = ambit::Tensor::build(ambit::CoreTensor, "D1avg", {nactv, nactv});
        D("pq") = 0.5 * D1a_("pq");
        D("pq") += 0.5 * D1b_("pq");

        D1a_("pq") = D("pq");
        D1b_("pq") = D("pq");
        outfile->Printf("    Done.");
    }
}

void V2RDM::read_3pdm() {
    // map file names
    std::map<unsigned int, std::string> filename;
    filename[PSIF_V2RDM_D3AAA] = "D3aaa";
    filename[PSIF_V2RDM_D3AAB] = "D3aab";
    filename[PSIF_V2RDM_D3BBA] = "D3bba";
    filename[PSIF_V2RDM_D3BBB] = "D3bbb";

    // test if files exist
    std::string str = "Testing if 3RDM files exist";
    outfile->Printf("\n  %-45s ...", str.c_str());
    std::shared_ptr<PSIO> psio(new PSIO());
    for (const auto& file :
         {PSIF_V2RDM_D3AAA, PSIF_V2RDM_D3AAB, PSIF_V2RDM_D3BBA, PSIF_V2RDM_D3BBB}) {
        if (!psio->exists(file)) {
            std::string error = "V2RDM file for " + filename[file] + " does not exist";
            throw psi::PSIEXCEPTION(error);
        }
    }
    outfile->Printf("    OK.");

    // initialization of 3PDM
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nactv2 = nactv * nactv;
    size_t nactv3 = nactv * nactv2;
    size_t nactv4 = nactv * nactv3;
    size_t nactv5 = nactv * nactv4;

    // Read 3RDM
    str = "Reading 3RDMs";
    outfile->Printf("\n  %-45s ...", str.c_str());
    for (const auto& file :
         {PSIF_V2RDM_D3AAA, PSIF_V2RDM_D3AAB, PSIF_V2RDM_D3BBA, PSIF_V2RDM_D3BBB}) {
        ambit::Tensor D3 = ambit::Tensor::build(ambit::CoreTensor, filename[file],
                                                {nactv, nactv, nactv, nactv, nactv, nactv});

        long int nline;
        psio_address addr = PSIO_ZERO;
        psio->open(file, PSIO_OPEN_OLD);
        psio->read_entry(file, "length", (char*)&nline, sizeof(long int));

        if (file != PSIF_V2RDM_D3BBA) {
            for (int nl = 0; nl < nline; ++nl) {
                dm3 d3;
                psio->read(file, filename[file].c_str(), (char*)&d3, sizeof(dm3), addr, &addr);
                size_t i = abs_to_rel_[static_cast<size_t>(d3.i)];
                size_t j = abs_to_rel_[static_cast<size_t>(d3.j)];
                size_t k = abs_to_rel_[static_cast<size_t>(d3.k)];
                size_t l = abs_to_rel_[static_cast<size_t>(d3.l)];
                size_t m = abs_to_rel_[static_cast<size_t>(d3.m)];
                size_t n = abs_to_rel_[static_cast<size_t>(d3.n)];

                size_t idx = i * nactv5 + j * nactv4 + k * nactv3 + l * nactv2 + m * nactv + n;
                D3.data()[idx] = d3.val;
            }
        } else {
            for (int nl = 0; nl < nline; ++nl) {
                dm3 d3;
                psio->read(file, filename[file].c_str(), (char*)&d3, sizeof(dm3), addr, &addr);
                size_t i = abs_to_rel_[static_cast<size_t>(d3.i)];
                size_t j = abs_to_rel_[static_cast<size_t>(d3.j)];
                size_t k = abs_to_rel_[static_cast<size_t>(d3.k)];
                size_t l = abs_to_rel_[static_cast<size_t>(d3.l)];
                size_t m = abs_to_rel_[static_cast<size_t>(d3.m)];
                size_t n = abs_to_rel_[static_cast<size_t>(d3.n)];

                size_t idx = k * nactv5 + i * nactv4 + j * nactv3 + n * nactv2 + l * nactv + m;
                D3.data()[idx] = d3.val;
            }
        }
        psio->close(file, 1);

        D3_.push_back(D3);
    }
    outfile->Printf("    Done.");

    // average Daaa and Dbbb, Daab and Dabb
    if (options_.get_bool("AVG_DENS_SPIN")) {
        // reference D3aaa, D3aab, D3abb, D3bbb to D3_ element
        ambit::Tensor& D3aaa = D3_[0];
        ambit::Tensor& D3aab = D3_[1];
        ambit::Tensor& D3abb = D3_[2];
        ambit::Tensor& D3bbb = D3_[3];

        str = "Averaging 3RDM AAA & BBB, AAB & ABB blocks";
        outfile->Printf("\n  %-45s ...", str.c_str());
        ambit::Tensor D3 = ambit::Tensor::build(ambit::CoreTensor, "D3avg_aa",
                                                {nactv, nactv, nactv, nactv, nactv, nactv});
        D3("pqrstu") = 0.5 * D3aaa("pqrstu");
        D3("pqrstu") += 0.5 * D3bbb("pqrstu");
        D3aaa("pqrstu") = D3("pqrstu");
        D3bbb("pqrstu") = D3("pqrstu");

        D3("pqrstu") = 0.5 * D3aab("pqrstu");
        D3("pqrstu") += 0.5 * D3abb("rpqust");
        D3aab("pqrstu") = D3("pqrstu");
        D3abb("rpqust") = D3("pqrstu");
        outfile->Printf("    Done.");
    }
}

double V2RDM::compute_ref_energy() {
    std::string str = "Computing reference energy";
    outfile->Printf("\n  %-45s ...", str.c_str());

    /* Eref = sum_{m} h^{m}_{m} + 0.5 * sum_{mn} v^{mn}_{mn}
              + \sum_{uv} ( h^{u}_{v} + \sum_{m} v^{mu}_{mv} ) * D^{v}_{u}
              + 0.25 * \sum_{uvxy} v^{xy}_{uv} * D^{uv}_{xy} */
    double Eref =
        frozen_core_energy_ +
        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());
    size_t ncore = core_mos_.size();
    size_t nactv = actv_mos_.size();

    // sum_{m} h^{m}_{m} + 0.5 * sum_{mn} v^{mn}_{mn}
    for (size_t m = 0; m < ncore; ++m) {
        size_t nm = core_mos_[m];
        Eref += ints_->oei_a(nm, nm);
        Eref += ints_->oei_b(nm, nm);

        for (size_t n = 0; n < ncore; ++n) {
            size_t nn = core_mos_[n];
            Eref += 0.5 * ints_->aptei_aa(nm, nn, nm, nn);
            Eref += 0.5 * ints_->aptei_bb(nm, nn, nm, nn);

            Eref += 0.5 * ints_->aptei_ab(nm, nn, nm, nn);
            Eref += 0.5 * ints_->aptei_ab(nn, nm, nn, nm);
        }
    }

    // \sum_{uv} ( h^{u}_{v} + \sum_{m} v^{mu}_{mv} ) * D^{v}_{u}
    ambit::Tensor sum = ambit::Tensor::build(ambit::CoreTensor, "sum_a", {nactv, nactv});
    sum.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        value = ints_->oei_a(nu, nv);

        for (size_t m = 0; m < ncore; ++m) {
            size_t nm = core_mos_[m];
            value += ints_->aptei_aa(nm, nu, nm, nv);
            value += ints_->aptei_ab(nu, nm, nv, nm);
        }
    });
    Eref += sum("uv") * D1a_("uv");

    sum.zero();
    sum.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        value = ints_->oei_b(nu, nv);

        for (size_t m = 0; m < ncore; ++m) {
            size_t nm = core_mos_[m];
            value += ints_->aptei_bb(nm, nu, nm, nv);
            value += ints_->aptei_ab(nm, nu, nm, nv);
        }
    });
    Eref += sum("uv") * D1b_("uv");

    // 0.25 * \sum_{uvxy} v^{xy}_{uv} * D^{uv}_{xy}
    ambit::Tensor& D2aa = D2_[0];
    ambit::Tensor& D2ab = D2_[1];
    ambit::Tensor& D2bb = D2_[2];
    sum = ambit::Tensor::build(ambit::CoreTensor, "sum_aa", {nactv, nactv, nactv, nactv});
    sum.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        size_t nx = actv_mos_[i[2]];
        size_t ny = actv_mos_[i[3]];
        value = ints_->aptei_aa(nu, nv, nx, ny);
    });
    Eref += 0.25 * sum("uvxy") * D2aa("uvxy");

    sum.zero();
    sum.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        size_t nx = actv_mos_[i[2]];
        size_t ny = actv_mos_[i[3]];
        value = ints_->aptei_bb(nu, nv, nx, ny);
    });
    Eref += 0.25 * sum("uvxy") * D2bb("uvxy");

    sum.zero();
    sum.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        size_t nx = actv_mos_[i[2]];
        size_t ny = actv_mos_[i[3]];
        value = ints_->aptei_ab(nu, nv, nx, ny);
    });
    Eref += sum("uvxy") * D2ab("uvxy");

    outfile->Printf("    Done.");
    return Eref;
}

RDMs V2RDM::reference() {
    std::string str = "Converting to RDMs";
    outfile->Printf("\n  %-45s ...", str.c_str());
    // if 3-RDMs are needed
    if (options_.get_str("THREEPDC") != "ZERO") {
        RDMs return_ref(D1a_, D1b_, D2_[0], D2_[1], D2_[3], D3_[0], D3_[1], D3_[2], D3_[3]);   
        if (options_.get_str("WRITE_DENSITY_TYPE") == "CUMULANT") {
            write_density_to_file();
        }

        outfile->Printf("    Done.");
        return return_ref;
    } else {
        
        RDMs return_ref(D1a_, D1b_, D2_[0], D2_[1], D2_[3]);   
        if (options_.get_str("WRITE_DENSITY_TYPE") == "CUMULANT") {
            write_density_to_file();
        }
        outfile->Printf("    Done.");
        return return_ref;
    }
}

void V2RDM::write_density_to_file() {
    std::string str = "Writing density matrices to files";
    outfile->Printf("\n  %-45s ...", str.c_str());

    std::vector<std::string> filenames;
    if (options_.get_str("WRITE_DENSITY_TYPE") == "DENSITY") {
        for (const std::string& spin : {"a", "b"}) {
            filenames.push_back("file_opdm_" + spin);
        }
        for (const std::string& spin : {"aa", "ab", "bb"}) {
            filenames.push_back("file_2pdm_" + spin);
        }
        for (const std::string& spin : {"aaa", "aab", "abb", "bbb"}) {
            filenames.push_back("file_3pdm_" + spin);
        }
    } else if (options_.get_str("WRITE_DENSITY_TYPE") == "CUMULANT") {
        for (const std::string& spin : {"a", "b"}) {
            filenames.push_back("file_opdc_" + spin);
        }
        for (const std::string& spin : {"aa", "ab", "bb"}) {
            filenames.push_back("file_2pdc_" + spin);
        }
        for (const std::string& spin : {"aaa", "aab", "abb", "bbb"}) {
            filenames.push_back("file_3pdc_" + spin);
        }
    }

    std::ofstream outfstr;
    outfstr.open(filenames[0]);
    D1a_.iterate([&](const std::vector<size_t>& i, double& value) {
        outfstr << boost::format("%4d %4d  %20.15f\n") % i[0] % i[1] % value;
    });
    outfstr.close();
    outfstr.clear();
    outfstr.open(filenames[1]);
    D1b_.iterate([&](const std::vector<size_t>& i, double& value) {
        outfstr << boost::format("%4d %4d  %20.15f\n") % i[0] % i[1] % value;
    });
    outfstr.close();
    outfstr.clear();

    for (int m = 0; m < 3; ++m) {
        outfstr.open(filenames[m + 2]);

        ambit::Tensor& D2 = D2_[m];
        D2.iterate([&](const std::vector<size_t>& i, double& value) {
            outfstr << boost::format("%4d %4d %4d %4d  %20.15f\n") % i[0] % i[1] % i[2] % i[3] %
                           value;
        });

        outfstr.close();
        outfstr.clear();
    }

    if (options_.get_str("THREEPDC") != "ZERO") {
        for (int m = 0; m < 4; ++m) {
            outfstr.open(filenames[m + 5]);

            ambit::Tensor& D3 = D3_[m];
            D3.iterate([&](const std::vector<size_t>& i, double& value) {
                outfstr << boost::format("%4d %4d %4d %4d %4d %4d  %20.15f\n") % i[0] % i[1] %
                               i[2] % i[3] % i[4] % i[5] % value;
            });

            outfstr.close();
            outfstr.clear();
        }
    }

    outfile->Printf("    Done.");
}
}
