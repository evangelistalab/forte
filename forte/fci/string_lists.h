#pragma once

#define USE_GAS_LISTS 1

#if USE_GAS_LISTS
#include "gas_string_lists.h"
namespace forte {
using StringLists = GASStringLists;
}
#else
#include "fci_string_lists.h"
namespace forte {
using StringLists = FCIStringLists;
}
#endif
