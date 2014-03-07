/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _tensorsrg_h_
#define _tensorsrg_h_

#include <fstream>

#include "methodbase.h"

namespace psi{ namespace libadaptive{

/**
 * @brief The TensorSRG class
 * This class implements Canonical Transformation (CT) theory
 * and the Similarity Renormalization Group (SRG) method using
 * the Tensor classes
 */
class TensorSRG : public MethodBase
{
public:
    TensorSRG(Options &options, ExplorerIntegrals* ints/*, TwoIndex G1aa, TwoIndex G1bb*/);
    ~TensorSRG();
private:
    void startup();
    void cleanup();

    BlockedTensor T2;
};

}} // End Namespaces

#endif // _tensorsrg_h_
