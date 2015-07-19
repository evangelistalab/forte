#include <cmath>
#include "uno.h"
#include "helpers.h"

namespace psi{
namespace libadaptive{

UNO::UNO(Options &options){

    print_method_banner({"Unrestricted Natural Orbitals (UNO)","Chenyang (York) Li"});

    // wavefunction from psi
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    // number of irrep
    int nirrep = wfn->nirrep();
    Dimension nsopi = wfn->nsopi();

    // total density
    SharedMatrix Dt = wfn->Da();
    Dt->add(wfn->Db());

    // AO overlap
    SharedMatrix overlap = wfn->S();
    SharedMatrix Lvector (new Matrix("Overlap Eigen Vectors", nsopi, nsopi));
    SharedVector Lvalues (new Vector("Overlap Eigen Values", nsopi));
    overlap->diagonalize(Lvector, Lvalues);

    // AO overlap one half and minus one half
    SharedMatrix Lvalue_onehalf (new Matrix("Overlap Eigen Values One Half", nsopi, nsopi));
    SharedMatrix Lvalue_minus_onehalf (new Matrix("Overlap Eigen Values Minus One Half", nsopi, nsopi));
    for(int h = 0; h != nirrep; ++h){
        size_t m = nsopi[h];
        for (size_t i = 0; i != m; ++i){
            Lvalue_onehalf->set(h, i, i, sqrt(Lvalues->get(h, i)));
            Lvalue_minus_onehalf->set(h, i, i, 1 / sqrt(Lvalues->get(h, i)));
        }
    }
    SharedMatrix S_onehalf = Matrix::triplet(Lvector, Lvalue_onehalf, Lvector, false, false, true);
    SharedMatrix S_minus_onehalf = Matrix::triplet(Lvector, Lvalue_minus_onehalf, Lvector, false, false, true);

    // diagonalize S^(1/2) * Dt * S^(1/2)
    SharedMatrix X = Matrix::triplet(S_onehalf, Dt, S_onehalf, false, false, false);
    SharedMatrix Xvector (new Matrix("S*Dt Eigen Vectors", nsopi, nsopi));
    SharedVector occ (new Vector("Occupation Numbers", nsopi));
    X->diagonalize(Xvector, occ, descending); // NOT sure if this always works, should be the orbitals ordered according to orbital energies?
//    Xvector->eivprint(occ);

    // print occupation number
    std::vector<size_t> closed, active;
    double unomin = options.get_double("UNOMIN");
    double unomax = options.get_double("UNOMAX");
    outfile->Printf("\n  UNO Orbital Spaces for CASSCF/CASCI (Min. Occ.: %.3f, Max. Occ.: %.3f)", unomin, unomax);
    outfile->Printf("\n");
    for(int h = 0; h != nirrep; ++h){
        size_t closedpi = 0, activepi = 0;
        for(size_t i = 0; i != nsopi[h]; ++i){
            double occ_num = occ->get(h, i);
            if(occ_num < unomin){
                continue;
            }else if(occ_num >= unomax){
                ++closedpi;
            }else{
                ++activepi;
            }
        }
        closed.push_back(closedpi);
        active.push_back(activepi);
    }
    outfile->Printf("\n  %-25s ", "CLOSED:");
    for(auto &rocc: closed){
        outfile->Printf("%5zu", rocc);
    }
    outfile->Printf("\n  %-25s ", "ACTIVE:");
    for(auto &aocc: active){
        outfile->Printf("%5zu", aocc);
    }
    outfile->Printf("\n");

    // UNO coefficients
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();
    SharedMatrix Cnew (new Matrix("New MO Coefficients", nsopi, nsopi));
    Cnew->gemm(false, false, 1.0, S_minus_onehalf, Xvector, 0.0);
    Ca->copy(Cnew);
    Cb->copy(Cnew);

    // print UNO details
    if(options.get_bool("UNO_PRINT")){
        outfile->Printf("\n  UNO Occupation Number:\n");
        occ->print();
        outfile->Printf("\n  UNO Coefficients:\n");
        Ca->print();
    }

    // write molden
    if(options.get_bool("MOLDEN_WRITE")){
        boost::shared_ptr<MoldenWriter> molden(new MoldenWriter(wfn));
        std::string filename = get_writer_file_prefix() + ".molden";

        SharedVector dummy(new Vector("Dummy Vector of Orbital Energy", nirrep, nsopi));

        SharedVector occ_a (new Vector("Occ. Alpha", nsopi));
        SharedVector occ_b (new Vector("Occ. Beta", nsopi));
        occ_a->copy(*occ);
        occ_b->copy(*occ);
        occ_a->scale(0.5);
        occ_b->scale(0.5);

        molden->write(filename, Ca, Ca, dummy, dummy, occ_a, occ_b);
    }
}

UNO::~UNO(){}

}}
