#ifndef _localize_h_
#define _localize_h_

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/local.h"

#include "helpers.h"
#include "integrals.h"
#include "reference.h"

namespace psi{

namespace forte{

class LOCALIZE
{
public:
    LOCALIZE(std::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~LOCALIZE();
    
    void localize_orbitals();

private:

    std::shared_ptr<Wavefunction> wfn_;

    std::shared_ptr<ForteIntegrals> ints_;

    size_t nfrz_;
    size_t nrst_;
    size_t namo_;

    int naocc_;
    int navir_;
    int multiplicity_;

    std::vector<size_t> abs_act_;
        
    std::string local_type_;
    std::string local_method_;

};


}} // End Namespaces

#endif
