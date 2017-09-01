#ifndef _run_dsrg_h_
#define _run_dsrg_h_

#include "../forte_options.h"
#include "../fci_mo.h"
#include "../fci/fci.h"
#include "../mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "../mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "../mrdsrg-spin-integrated/master_mrdsrg.h"
#include "../mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "../mrdsrg-spin-integrated/mrdsrg.h"

namespace psi {
namespace forte {

/// Set the DSRG options
void set_DSRG_options(ForteOptions& foptions);
}
}
#endif // RUN_DSRG_H
