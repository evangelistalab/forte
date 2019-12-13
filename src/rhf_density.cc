#include <tuple>

#include "ambit/tensor.h"

#include "rhf_density.h"

//For debug only:
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"
#include "helpers/helpers.h"
#include "helpers/printing.h"

namespace forte {

RHF_DENSITY::RHF_DENSITY(std::shared_ptr<SCFInfo> scf_info,
                         std::shared_ptr<MOSpaceInfo> mo_space_info)
    : scf_info_(scf_info), mo_space_info_(mo_space_info) {

    start_up();
}

void RHF_DENSITY::start_up() {
    auto doccpi = scf_info_->doccpi();
    auto rdoccpi = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    auto actvpi = mo_space_info_->get_dimension("ACTIVE");
    auto actvopi = doccpi - rdoccpi - mo_space_info_->get_dimension("FROZEN_DOCC");

    psi::outfile->Printf("\n docc: %d, rdocc: %d, actv: %d, actvo: %d", doccpi[0], rdoccpi[0], actvpi[0], actvopi[0]);

    mos_actv_o_.clear();
    mos_actv_o_.reserve(actvopi.sum());
    for (int h = 0, shift = 0, nirrep = mo_space_info_->nirrep(); h < nirrep; ++h) {
        for (int i = 0; i < actvopi[h]; ++i) {
            mos_actv_o_.push_back(shift + i);
        }
        shift += actvpi[h];
    }
}

RDMs RHF_DENSITY::rhf_rdms() {
    size_t na = mo_space_info_->size("ACTIVE");
    psi::outfile->Printf("\n na: %d", na);

    // 1-RDM
    auto D1a = ambit::Tensor::build(ambit::CoreTensor, "D1a", std::vector<size_t>(2, na));
    auto D1b = ambit::Tensor::build(ambit::CoreTensor, "D1b", std::vector<size_t>(2, na));

    for (auto i : mos_actv_o_) {
        psi::outfile->Printf("\n rdm(%d, %d)", i, i);
        D1a.data()[i * na + i] = 1.0;
        D1b.data()[i * na + i] = 1.0;
    }

    psi::outfile->Printf("\n D1 formed.");

    // 2-RDM
    auto D2aa = ambit::Tensor::build(ambit::CoreTensor, "D2aa", std::vector<size_t>(4, na));
    auto D2ab = ambit::Tensor::build(ambit::CoreTensor, "D2ab", std::vector<size_t>(4, na));
    auto D2bb = ambit::Tensor::build(ambit::CoreTensor, "D2bb", std::vector<size_t>(4, na));

    D2aa("pqrs") += D1a("pr") * D1a("qs");
    D2aa("pqrs") -= D1a("ps") * D1a("qr");

    D2bb("pqrs") += D1b("pr") * D1b("qs");
    D2bb("pqrs") -= D1b("ps") * D1b("qr");

    D2ab("pqrs") = D1a("pr") * D1b("qs");

    psi::outfile->Printf("\n D2 formed.");

    // 3-RDM
    if (0) {
    auto D3aaa = ambit::Tensor::build(ambit::CoreTensor, "D3aaa", std::vector<size_t>(6, na));
    auto D3aab = ambit::Tensor::build(ambit::CoreTensor, "D3aab", std::vector<size_t>(6, na));
    auto D3abb = ambit::Tensor::build(ambit::CoreTensor, "D3abb", std::vector<size_t>(6, na));
    auto D3bbb = ambit::Tensor::build(ambit::CoreTensor, "D3bbb", std::vector<size_t>(6, na));


    D3aaa("pqrstu") += D1a("ps") * D1a("qt") * D1a("ru");
    D3aaa("pqrstu") -= D1a("ps") * D1a("rt") * D1a("qu");
    D3aaa("pqrstu") -= D1a("qs") * D1a("pt") * D1a("ru");
    D3aaa("pqrstu") += D1a("qs") * D1a("rt") * D1a("pu");
    D3aaa("pqrstu") -= D1a("rs") * D1a("qt") * D1a("pu");
    D3aaa("pqrstu") += D1a("rs") * D1a("pt") * D1a("qu");

    D3bbb("pqrstu") += D1b("ps") * D1b("qt") * D1b("ru");
    D3bbb("pqrstu") -= D1b("ps") * D1b("rt") * D1b("qu");
    D3bbb("pqrstu") -= D1b("qs") * D1b("pt") * D1b("ru");
    D3bbb("pqrstu") += D1b("qs") * D1b("rt") * D1b("pu");
    D3bbb("pqrstu") -= D1b("rs") * D1b("qt") * D1b("pu");
    D3bbb("pqrstu") += D1b("rs") * D1b("pt") * D1b("qu");

    D3aab("pqrstu") += D1a("ps") * D1a("qt") * D1b("ru");
    D3aab("pqrstu") -= D1a("qs") * D1a("pt") * D1b("ru");

    D3abb("pqrstu") += D1a("ps") * D1b("qt") * D1b("ru");
    D3abb("pqrstu") -= D1a("ps") * D1b("rt") * D1b("qu");

        psi::outfile->Printf("\n D3 formed.");

        return RDMs(D1a, D1b, D2aa, D2ab, D2bb, D3aaa, D3aab, D3abb, D3bbb);
    }
    else {
        psi::outfile->Printf("\n D3 skipped.");

        return RDMs(D1a, D1b, D2aa, D2ab, D2bb);
    }
}

} // namespace forte
