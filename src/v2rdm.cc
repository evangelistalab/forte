#include <libmints/molecule.h>
#include "v2rdm.h"

namespace psi{ namespace forte {

struct tpdm {
    int i, j, k, l;
    double val;
};

V2RDM::V2RDM(SharedWavefunction ref_wfn, Options &options,
             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info)
{
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    print_method_banner({"V2RDM-CASSCF Interface"});
    startup();
}

V2RDM::~V2RDM()
{}

void V2RDM::startup(){
    // number of MO per irrep
    nmopi_ = this->nmopi();
    nirrep_ = this->nirrep();
    fdoccpi_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    rdoccpi_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    active_ = mo_space_info_->get_dimension("ACTIVE");

    // map active absolute index to relative index
    for(size_t h = 0, offset_abs = 0, offset_rel = 0; h < nirrep_; ++h){
        for(size_t u = 0; u < active_[h]; ++u){
            size_t abs = fdoccpi_[h] + rdoccpi_[h] + u + offset_abs;
            size_t rel = u + offset_rel;
            abs_to_rel_[abs] = rel;
            ++offset_rel;
        }
        offset_abs += nmopi_[h];
    }

    // read 2-pdm
    read_2pdm();

    // build opdm
    build_opdm();

    // read 3-pdm
    if(options_.get_str("THREEPDC") != "ZERO"){
        read_3pdm();
    }

    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // orbital spaces
    core_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
}

void V2RDM::read_2pdm(){
    std::string str = "Testing if 2RDM files exist";
    outfile->Printf("\n  %-45s ...", str.c_str());
    boost::shared_ptr<PSIO> psio (new PSIO());
    if ( !psio->exists(PSIF_V2RDM_D2AB) ) {
        throw PSIEXCEPTION("V2RDM_D2AB does not exist.");
    }
    if ( !psio->exists(PSIF_V2RDM_D2AA) ) {
        throw PSIEXCEPTION("V2RDM_D2AA does not exist.");
    }
    if ( !psio->exists(PSIF_V2RDM_D2BB) ) {
        throw PSIEXCEPTION("V2RDM_D2BB does not exist.");
    }
    outfile->Printf("    OK.");

    // initialization of 2PDM
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nactv2 = nactv * nactv;
    size_t nactv3 = nactv * nactv2;
    D2ab_ = ambit::Tensor::build(ambit::CoreTensor,"D2ab",{nactv,nactv,nactv,nactv});
    D2aa_ = ambit::Tensor::build(ambit::CoreTensor,"D2aa",{nactv,nactv,nactv,nactv});
    D2bb_ = ambit::Tensor::build(ambit::CoreTensor,"D2bb",{nactv,nactv,nactv,nactv});

    // 2PDM AB spin
    str = "Reading 2RDM AB block";
    outfile->Printf("\n  %-45s ...", str.c_str());
    long int nab;
    psio_address addr_ab = PSIO_ZERO;
    psio->open(PSIF_V2RDM_D2AB,PSIO_OPEN_OLD);
    psio->read_entry(PSIF_V2RDM_D2AB,"length",(char*)&nab,sizeof(long int));

    // test if active orbitals are consistent in forte and v2rdm-casscf
    size_t nsymgem = 0; // number of totally symmetric geminals
    for(int h = 0; h < nirrep_; ++h){
        nsymgem += active_[h] * active_[h];
    }
    for(size_t n = 0; n < nsymgem; ++n){
        tpdm d2;
        psio->read(PSIF_V2RDM_D2AB,"D2ab",(char*)&d2,sizeof(tpdm),addr_ab,&addr_ab);
        size_t l = static_cast<size_t> (d2.l);
        if(abs_to_rel_.find(l) == abs_to_rel_.end()){
            outfile->Printf("\n  The active block of FORTE is different from V2RDM-CASSCF.");
            outfile->Printf("\n  Please check the input file and make the active block consistent.");
            throw PSIEXCEPTION("The active block of FORTE is different from V2RDM-CASSCF.");
        }
    }
    addr_ab = PSIO_ZERO; // reset address to the beginning of the file

    for (int n = 0; n < nab; ++n) {
        tpdm d2;
        psio->read(PSIF_V2RDM_D2AB,"D2ab",(char*)&d2,sizeof(tpdm),addr_ab,&addr_ab);
        size_t i = abs_to_rel_[static_cast<size_t> (d2.i)];
        size_t j = abs_to_rel_[static_cast<size_t> (d2.j)];
        size_t k = abs_to_rel_[static_cast<size_t> (d2.k)];
        size_t l = abs_to_rel_[static_cast<size_t> (d2.l)];

        size_t idx = i * nactv3 + j * nactv2 + k * nactv + l;
        D2ab_.data()[idx] = d2.val;
    }
    psio->close(PSIF_V2RDM_D2AB,1);
    outfile->Printf("    Done.");

    // 2PDM AA spin
    str = "Reading 2RDM AA block";
    outfile->Printf("\n  %-45s ...", str.c_str());
    long int naa;
    psio_address addr_aa = PSIO_ZERO;
    psio->open(PSIF_V2RDM_D2AA,PSIO_OPEN_OLD);
    psio->read_entry(PSIF_V2RDM_D2AA,"length",(char*)&naa,sizeof(long int));
    for (int n = 0; n < naa; ++n) {
        tpdm d2;
        psio->read(PSIF_V2RDM_D2AA,"D2aa",(char*)&d2,sizeof(tpdm),addr_aa,&addr_aa);
        size_t i = abs_to_rel_[static_cast<size_t> (d2.i)];
        size_t j = abs_to_rel_[static_cast<size_t> (d2.j)];
        size_t k = abs_to_rel_[static_cast<size_t> (d2.k)];
        size_t l = abs_to_rel_[static_cast<size_t> (d2.l)];

        size_t idx = i * nactv3 + j * nactv2 + k * nactv + l;
        D2aa_.data()[idx] = d2.val;
    }
    psio->close(PSIF_V2RDM_D2AA,1);
    outfile->Printf("    Done.");

    // 2PDM BB spin
    str = "Reading 2RDM BB block";
    outfile->Printf("\n  %-45s ...", str.c_str());
    long int nbb;
    psio_address addr_bb = PSIO_ZERO;
    psio->open(PSIF_V2RDM_D2BB,PSIO_OPEN_OLD);
    psio->read_entry(PSIF_V2RDM_D2BB,"length",(char*)&nbb,sizeof(long int));
    for (int n = 0; n < nbb; ++n) {
        tpdm d2;
        psio->read(PSIF_V2RDM_D2BB,"D2bb",(char*)&d2,sizeof(tpdm),addr_bb,&addr_bb);
        size_t i = abs_to_rel_[static_cast<size_t> (d2.i)];
        size_t j = abs_to_rel_[static_cast<size_t> (d2.j)];
        size_t k = abs_to_rel_[static_cast<size_t> (d2.k)];
        size_t l = abs_to_rel_[static_cast<size_t> (d2.l)];

        size_t idx = i * nactv3 + j * nactv2 + k * nactv + l;
        D2bb_.data()[idx] = d2.val;
    }
    psio->close(PSIF_V2RDM_D2BB,1);
    outfile->Printf("    Done.");

    // average Daa and Dbb
    if(options_.get_bool("AVG_DENS_SPIN")){
        str = "Averaging 2RDM AA and BB blocks";
        outfile->Printf("\n  %-45s ...", str.c_str());
        ambit::Tensor D2 = ambit::Tensor::build(ambit::CoreTensor,"D2avg_aa",{nactv,nactv,nactv,nactv});
        D2("pqrs")  = D2aa_("pqrs");
        D2("pqrs") += D2bb_("pqrs");
        D2("pqrs") += D2ab_("pqrs");
        D2("pqrs") -= D2ab_("pqsr");
        D2.scale(1.0/3.0);

        D2aa_("pqrs")  = D2("pqrs");
        D2bb_("pqrs")  = D2("pqrs");
        outfile->Printf("    Done.");
    }
}

void V2RDM::build_opdm(){
    std::string str = "Computing 1RDM";
    outfile->Printf("\n  %-45s ...", str.c_str());

    // initialization of OPDM
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nactv2 = nactv * nactv;
    size_t nactv3 = nactv * nactv2;
    D1a_ = ambit::Tensor::build(ambit::CoreTensor,"D1a",{nactv,nactv});
    D1b_ = ambit::Tensor::build(ambit::CoreTensor,"D1b",{nactv,nactv});

    // number of active electrons
    size_t nalfa = this->nalpha() - mo_space_info_->size("FROZEN_DOCC")
            - mo_space_info_->size("RESTRICTED_DOCC");
    size_t nbeta = this->nbeta() - mo_space_info_->size("FROZEN_DOCC")
            - mo_space_info_->size("RESTRICTED_DOCC");

    // compute OPDM
    for(size_t u = 0; u < nactv; ++u){
        for(size_t v = 0; v < nactv; ++v){

            double va = 0, vb = 0;
            for(size_t x = 0; x < nactv; ++x){
                va += D2aa_.data()[u * nactv3 + x * nactv2 + v * nactv + x];
                va += D2ab_.data()[u * nactv3 + x * nactv2 + v * nactv + x];

                vb += D2bb_.data()[u * nactv3 + x * nactv2 + v * nactv + x];
                vb += D2ab_.data()[x * nactv3 + u * nactv2 + x * nactv + v];
            }

            D1a_.data()[u * nactv + v] = va / (nalfa + nbeta - 1.0);
            D1b_.data()[u * nactv + v] = vb / (nalfa + nbeta - 1.0);
        }
    }
    outfile->Printf("    Done.");

    // average Da and Db
    if(options_.get_bool("AVG_DENS_SPIN")){
        str = "Averaging 1RDM A and B blocks";
        outfile->Printf("\n  %-45s ...", str.c_str());
        ambit::Tensor D = ambit::Tensor::build(ambit::CoreTensor,"D1avg",{nactv,nactv});
        D("pq")  = 0.5 * D1a_("pq");
        D("pq") += 0.5 * D1b_("pq");

        D1a_("pq") = D("pq");
        D1b_("pq") = D("pq");
        outfile->Printf("    Done.");
    }
}

void V2RDM::read_3pdm(){
    std::string str = "Testing if 3RDM files exist";
    outfile->Printf("\n  %-45s ...", str.c_str());
    boost::shared_ptr<PSIO> psio (new PSIO());
//    if ( !psio->exists(PSIF_V2RDM_D2AB) ) {
//        throw PSIEXCEPTION("V2RDM_D2AB does not exist.");
//    }
    outfile->Printf("    OK.");

    // initialization of 3PDM
    size_t nactv = mo_space_info_->size("ACTIVE");
    size_t nactv2 = nactv * nactv;
    size_t nactv3 = nactv * nactv2;
    size_t nactv4 = nactv * nactv3;
    size_t nactv5 = nactv * nactv4;
    D3aaa_ = ambit::Tensor::build(ambit::CoreTensor,"D3aaa",{nactv,nactv,nactv,nactv,nactv,nactv});
    D3aab_ = ambit::Tensor::build(ambit::CoreTensor,"D3aab",{nactv,nactv,nactv,nactv,nactv,nactv});
    D3abb_ = ambit::Tensor::build(ambit::CoreTensor,"D3abb",{nactv,nactv,nactv,nactv,nactv,nactv});
    D3bbb_ = ambit::Tensor::build(ambit::CoreTensor,"D3bbb",{nactv,nactv,nactv,nactv,nactv,nactv});

    // 3PDM AAA spin
    str = "Reading 3RDM AAA block";
    outfile->Printf("\n  %-45s ...", str.c_str());
//    long int naa;
//    psio_address addr_aa = PSIO_ZERO;
//    psio->open(PSIF_V2RDM_D2AA,PSIO_OPEN_OLD);
//    psio->read_entry(PSIF_V2RDM_D2AA,"length",(char*)&naa,sizeof(long int));
//    for (int n = 0; n < naa; ++n) {
//        tpdm d2;
//        psio->read(PSIF_V2RDM_D2AA,"D2aa",(char*)&d2,sizeof(tpdm),addr_aa,&addr_aa);
//        size_t i = abs_to_rel_[static_cast<size_t> (d2.i)];
//        size_t j = abs_to_rel_[static_cast<size_t> (d2.j)];
//        size_t k = abs_to_rel_[static_cast<size_t> (d2.k)];
//        size_t l = abs_to_rel_[static_cast<size_t> (d2.l)];

//        size_t idx = i * nactv3 + j * nactv2 + k * nactv + l;
//        D2aa_.data()[idx] = d2.val;
//    }
//    psio->close(PSIF_V2RDM_D2AA,1);
    outfile->Printf("    Done.");

    // 3PDM AAB spin
    str = "Reading 3RDM AAB block";
    outfile->Printf("\n  %-45s ...", str.c_str());

    outfile->Printf("    Done.");

    // 3PDM ABB spin
    str = "Reading 3RDM ABB block";
    outfile->Printf("\n  %-45s ...", str.c_str());

    outfile->Printf("    Done.");

    // 3PDM BBB spin
    str = "Reading 3RDM BBB block";
    outfile->Printf("\n  %-45s ...", str.c_str());

    outfile->Printf("    Done.");

    // average Daaa and Dbbb, Daab and Dabb
    if(options_.get_bool("AVG_DENS_SPIN")){
        str = "Averaging 3RDM AAA & BBB, AAB & ABB blocks";
        outfile->Printf("\n  %-45s ...", str.c_str());
//        ambit::Tensor D2 = ambit::Tensor::build(ambit::CoreTensor,"D2avg_aa",{nactv,nactv,nactv,nactv});
//        D2("pqrs")  = D2aa_("pqrs");
//        D2("pqrs") += D2bb_("pqrs");
//        D2("pqrs") += D2ab_("pqrs");
//        D2("pqrs") -= D2ab_("pqsr");
//        D2.scale(1.0/3.0);

//        D2aa_("pqrs")  = D2("pqrs");
//        D2bb_("pqrs")  = D2("pqrs");
        outfile->Printf("    Done.");
    }
}

double V2RDM::compute_ref_energy(){
    std::string str = "Computing reference energy";
    outfile->Printf("\n  %-45s ...", str.c_str());

    /* Eref = sum_{m} h^{m}_{m} + 0.5 * sum_{mn} v^{mn}_{mn}
              + \sum_{uv} ( h^{u}_{v} + \sum_{m} v^{mu}_{mv} ) * D^{v}_{u}
              + 0.25 * \sum_{uvxy} v^{xy}_{uv} * D^{uv}_{xy} */
    double Eref = frozen_core_energy_ + molecule_->nuclear_repulsion_energy();
    size_t ncore = core_mos_.size();
    size_t nactv = actv_mos_.size();

    // sum_{m} h^{m}_{m} + 0.5 * sum_{mn} v^{mn}_{mn}
    for(size_t m = 0; m < ncore; ++m){
        size_t nm = core_mos_[m];
        Eref += ints_->oei_a(nm,nm);
        Eref += ints_->oei_b(nm,nm);

        for(size_t n = 0; n < ncore; ++n){
            size_t nn = core_mos_[n];
            Eref += 0.5 * ints_->aptei_aa(nm,nn,nm,nn);
            Eref += 0.5 * ints_->aptei_bb(nm,nn,nm,nn);

            Eref += 0.5 * ints_->aptei_ab(nm,nn,nm,nn);
            Eref += 0.5 * ints_->aptei_ab(nn,nm,nn,nm);
        }
    }

    // \sum_{uv} ( h^{u}_{v} + \sum_{m} v^{mu}_{mv} ) * D^{v}_{u}
    ambit::Tensor sum = ambit::Tensor::build(ambit::CoreTensor,"sum_a",{nactv,nactv});
    sum.iterate([&](const std::vector<size_t>& i,double& value){
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        value = ints_->oei_a(nu,nv);

        for(size_t m = 0; m < ncore; ++m){
            size_t nm = core_mos_[m];
            value += ints_->aptei_aa(nm,nu,nm,nv);
            value += ints_->aptei_ab(nu,nm,nv,nm);
        }
    });
    Eref += sum("uv") * D1a_("uv");

    sum.zero();
    sum.iterate([&](const std::vector<size_t>& i,double& value){
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        value = ints_->oei_b(nu,nv);

        for(size_t m = 0; m < ncore; ++m){
            size_t nm = core_mos_[m];
            value += ints_->aptei_bb(nm,nu,nm,nv);
            value += ints_->aptei_ab(nm,nu,nm,nv);
        }
    });
    Eref += sum("uv") * D1b_("uv");

    // 0.25 * \sum_{uvxy} v^{xy}_{uv} * D^{uv}_{xy}
    sum = ambit::Tensor::build(ambit::CoreTensor,"sum_aa",{nactv,nactv,nactv,nactv});
    sum.iterate([&](const std::vector<size_t>& i,double& value){
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        size_t nx = actv_mos_[i[2]];
        size_t ny = actv_mos_[i[3]];
        value = ints_->aptei_aa(nu,nv,nx,ny);
    });
    Eref += 0.25 * sum("uvxy") * D2aa_("uvxy");

    sum.zero();
    sum.iterate([&](const std::vector<size_t>& i,double& value){
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        size_t nx = actv_mos_[i[2]];
        size_t ny = actv_mos_[i[3]];
        value = ints_->aptei_bb(nu,nv,nx,ny);
    });
    Eref += 0.25 * sum("uvxy") * D2bb_("uvxy");

    sum.zero();
    sum.iterate([&](const std::vector<size_t>& i,double& value){
        size_t nu = actv_mos_[i[0]];
        size_t nv = actv_mos_[i[1]];
        size_t nx = actv_mos_[i[2]];
        size_t ny = actv_mos_[i[3]];
        value = ints_->aptei_ab(nu,nv,nx,ny);
    });
    Eref += sum("uvxy") * D2ab_("uvxy");

    outfile->Printf("    Done.");
    return Eref;
}

Reference V2RDM::reference(){
    Reference return_ref;
    double Eref = compute_ref_energy();

    std::string str = "Converting to Reference";
    outfile->Printf("\n  %-45s ...", str.c_str());

    // compute 2-cumulants
    D2aa_("pqrs") -= D1a_("pr") * D1a_("qs");
    D2aa_("pqrs") += D1a_("ps") * D1a_("qr");

    D2bb_("pqrs") -= D1b_("pr") * D1b_("qs");
    D2bb_("pqrs") += D1b_("ps") * D1b_("qr");

    D2ab_("pqrs") -= D1a_("pr") * D1b_("qs");

    // compute 3-cumulants


    // fill out values
    return_ref.set_Eref(Eref);
    return_ref.set_L1a(D1a_);
    return_ref.set_L1b(D1b_);
    return_ref.set_L2aa(D2aa_);
    return_ref.set_L2ab(D2ab_);
    return_ref.set_L2bb(D2bb_);
    if(options_.get_str("THREEPDC") != "ZERO"){
        return_ref.set_L3aaa(D3aaa_);
        return_ref.set_L3aab(D3aab_);
        return_ref.set_L3abb(D3abb_);
        return_ref.set_L3bbb(D3bbb_);
    }

    outfile->Printf("    Done.");
    return return_ref;
}

}}
