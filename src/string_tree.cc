#include "string_tree.h"

string_tree::string_tree(size_t ndets)
    : sorted_ab_(ndets), sorted_ba_(ndets), C_ab_(ndets), C_ba_(ndets)
{
}
