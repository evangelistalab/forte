#include "dsrg_source.h"

namespace psi{ namespace forte{

DSRG_SOURCE::DSRG_SOURCE(double s, double taylor_threshold)
    : s_(s), taylor_threshold_(taylor_threshold)
{}

STD_SOURCE::STD_SOURCE(double s, double taylor_threshold)
    : DSRG_SOURCE(s, taylor_threshold)
{}

LABS_SOURCE::LABS_SOURCE(double s, double taylor_threshold)
    : DSRG_SOURCE(s, taylor_threshold)
{}

DYSON_SOURCE::DYSON_SOURCE(double s, double taylor_threshold)
    : DSRG_SOURCE(s, taylor_threshold)
{}
MP2_SOURCE::MP2_SOURCE(double s, double taylor_threshold)
    : DSRG_SOURCE(s, taylor_threshold)
{}

}}
