#ifndef _localize_h_
#define _localize_h_

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include <libmints/local.h>

#include "helpers.h"
#include "integrals.h"
#include "reference.h"

namespace psi{

namespace forte{

class LOCALIZE
{
public:
    LOCALIZE(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~LOCALIZE();
    

private:
    
    std::string local_type_;
    void update_cmat(boost::shared_ptr<BasisSet> primary, SharedMatrix Ca, SharedMatrix Cb);

};


}} // End Namespaces

#endif
