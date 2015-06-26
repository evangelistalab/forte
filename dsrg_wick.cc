#include "ambit/tensor.h"
#include "dsrg_wick.h"

namespace psi{ namespace libadaptive {

DSRG_WICK::DSRG_WICK(std::shared_ptr<MOSpaceInfo> mo_space_info,
                     ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI,
                     ambit::BlockedTensor T1, ambit::BlockedTensor T2)
    : mo_space_info_(mo_space_info)
{
    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("RESTRICTED_DOCC")
            + mo_space_info_->size("ACTIVE")
            + mo_space_info_->size("RESTRICTED_UOCC");

    acore_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    for(size_t idx: acore_mos){
        bcore_mos.push_back(idx + mo_shift);
    }

    aactv_mos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    for(size_t idx: aactv_mos){
        bactv_mos.push_back(idx + mo_shift);
    }

    avirt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    for(size_t idx: avirt_mos){
        bvirt_mos.push_back(idx + mo_shift);
    }

    ncso_ = 2 * mo_shift;
    nc_ = 2 * acore_mos.size();
    na_ = 2 * aactv_mos.size();
    nv_ = 2 * avirt_mos.size();
    nh_ = nc_ + na_;
    np_ = nv_ + na_;

    // block indices: c for core, a for active, v for virtual
    // lowercase for alpha and uppercase for beta
    label_to_spacemo['c'] = acore_mos;
    label_to_spacemo['C'] = bcore_mos;
    label_to_spacemo['a'] = aactv_mos;
    label_to_spacemo['A'] = bactv_mos;
    label_to_spacemo['v'] = avirt_mos;
    label_to_spacemo['V'] = bvirt_mos;

    // setup Fock operator
    setup(Fock, RTEI, T1, T2);
}

void DSRG_WICK::setup(ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI,
                      ambit::BlockedTensor T1, ambit::BlockedTensor T2){
    // Setup Fock operator

//    d2 F = d2(ncso_, d1(ncso_));
//    for(std::string block: Fock.block_labels()){
//        Fock.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
//            size_t idx0 = label_to_spacemo[block[0]][i[0]];
//            size_t idx1 = label_to_spacemo[block[1]][i[1]];
//            F[idx0][idx1] = value;
//        });
//    }

//    SharedMatrix M (new Matrix("Fock", ncso_, ncso_));
//    for(size_t i = 0; i != ncso_; ++i){
//        for(size_t j = 0; j != ncso_; ++j){
//            M->pointer()[i][j] = F[i][j];
//        }
//    }
//    M->print();

//    for(int p = 0; p < ncso_; ++p){
//        for(int q = 0; q < ncso_; ++q){
//            SqOperator op_pq({p}, {q});
//            F_.add(F[p][q], op_pq);
//        }
//    }

    for(std::string block: Fock.block_labels()){
        Fock.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            SqOperator op_pq({idx0}, {idx1});
//            op_pq.test_sort();
            F_.add(value, op_pq);
        });
    }
//    outfile->Printf("\n  %s", F_.str().c_str());

    // Setup two-electron integral
    for(std::string block: RTEI.block_labels()){
        RTEI.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            int idx2 = label_to_spacemo[block[2]][i[2]];
            int idx3 = label_to_spacemo[block[3]][i[3]];
            if(islower(block[0]) && isupper(block[1])){
                SqOperator abab({idx0, idx1}, {idx2, idx3});
                SqOperator abba({idx0, idx1}, {idx3, idx2});
                SqOperator baab({idx1, idx0}, {idx2, idx3});
                SqOperator baba({idx1, idx0}, {idx3, idx2});
                V_.add(value, abab);
                V_.add(value, baba);
                V_.add(-value, abba);
                V_.add(-value, baab);
            }else{
                SqOperator op_pqrs({idx0, idx1}, {idx2, idx3});
                V_.add(value, op_pqrs);
            }
//            outfile->Printf("\n  [%d][%d][%d][%d] = %.15f", idx0, idx1, idx2, idx3, value);
        });
    }

//    d4 V = d4(ncso_, d3(ncso_, d2(ncso_, d1(ncso_))));
//    for(std::string block: RTEI.block_labels()){
//        RTEI.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
//            size_t idx0 = label_to_spacemo[block[0]][i[0]];
//            size_t idx1 = label_to_spacemo[block[1]][i[1]];
//            size_t idx2 = label_to_spacemo[block[2]][i[2]];
//            size_t idx3 = label_to_spacemo[block[3]][i[3]];
    // TODO add different spin
//            V[idx0][idx1][idx2][idx3] = value;
//        });
//    }
//    for(int p = 0; p != ncso_; ++p){
//        for(int q = 0; q != ncso_; ++q){
//            for(int r = 0; r != ncso_; ++r){
//                for(int s = 0; s != ncso_; ++s){
//                    SqOperator op_pqrs({p,q}, {r,s});
//                    V_.add(V[p][q][r][s], op_pqrs);
//                }
//            }
//        }
//    }

    // Setup T1
    for(std::string block: T1.block_labels()){
        T1.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            SqOperator op_pq({idx0}, {idx1});
//            op_pq.test_sort();
            T1_.add(value, op_pq);
        });
    }
//    outfile->Printf("\n  %s", T1_.str().c_str());

    // Setup T2
    for(std::string block: T2.block_labels()){
        T2.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            int idx2 = label_to_spacemo[block[2]][i[2]];
            int idx3 = label_to_spacemo[block[3]][i[3]];
            if(islower(block[0]) && isupper(block[1])){
                SqOperator abab({idx0, idx1}, {idx2, idx3});
                SqOperator abba({idx0, idx1}, {idx3, idx2});
                SqOperator baab({idx1, idx0}, {idx2, idx3});
                SqOperator baba({idx1, idx0}, {idx3, idx2});
                T2_.add(value, abab);
                T2_.add(value, baba);
                T2_.add(-value, abba);
                T2_.add(-value, baab);
            }else{
                SqOperator op_pqrs({idx0, idx1}, {idx2, idx3});
                T2_.add(value, op_pqrs);
            }
            outfile->Printf("\n  [%d][%d][%d][%d] = %.15f", idx0, idx1, idx2, idx3, value);
        });
    }

//    SqTest sqtest;
}



}}
