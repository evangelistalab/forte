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

#ifndef _reference_h_
#define _reference_h_

#include <libmints/wavefunction.h>

#include "tensor_basic.h"

namespace psi{
namespace libadaptive{

class Reference // : public Wavefunction
{
protected:

    // => Class data <= //

//    Dimension rdoccpi_;
//    Dimension actvpi_;
//    Dimension udoccpi_;

    double Eref_;

    SharedTensor L1a_;
    SharedTensor L1b_;
    SharedTensor L2aa_;
    SharedTensor L2ab_;
    SharedTensor L2bb_;
    SharedTensor L3aaa_;
    SharedTensor L3aab_;
    SharedTensor L3abb_;
    SharedTensor L3bbb_;

public:

    // => Constructors <= //

    Reference();
    Reference(double Eref,SharedTensor L1a,SharedTensor L1b,SharedTensor L2aa,SharedTensor L2ab,SharedTensor L2bb);
    Reference(double Eref,SharedTensor L1a,SharedTensor L1b,SharedTensor L2aa,SharedTensor L2ab,SharedTensor L2bb,
              SharedTensor L3aaa,SharedTensor L3aab,SharedTensor L3abb,SharedTensor L3bbb);
    ~Reference();

    double get_Eref() {return Eref_;}

    void set_L1a(SharedTensor L1a) {L1a_ = L1a;}
    void set_L1b(SharedTensor L1b) {L1b_ = L1b;}

    void set_L2aa(SharedTensor L2aa) {L2aa_ = L2aa;}
    void set_L2ab(SharedTensor L2ab) {L2ab_ = L2ab;}
    void set_L2bb(SharedTensor L2bb) {L2bb_ = L2bb;}

    void set_L3aaa(SharedTensor L3aaa) {L3aaa_ = L3aaa;}
    void set_L3aab(SharedTensor L3aab) {L3aab_ = L3aab;}
    void set_L3abb(SharedTensor L3abb) {L3abb_ = L3abb;}
    void set_L3bbb(SharedTensor L3bbb) {L3bbb_ = L3bbb;}

    SharedTensor L1a() {return L1a_;}
    SharedTensor L1b() {return L1b_;}
    SharedTensor L2aa() {return L2aa_;}
    SharedTensor L2ab() {return L2ab_;}
    SharedTensor L2bb() {return L2bb_;}
    SharedTensor L3aaa() {return L3aaa_;}
    SharedTensor L3aab() {return L3aab_;}
    SharedTensor L3abb() {return L3abb_;}
    SharedTensor L3bbb() {return L3bbb_;}
};

}} // End Namespaces

#endif // _reference_h_
