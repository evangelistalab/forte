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

#include <ambit/tensor.h>

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

    ambit::Tensor L1a_;
    ambit::Tensor L1b_;
    ambit::Tensor L2aa_;
    ambit::Tensor L2ab_;
    ambit::Tensor L2bb_;
    ambit::Tensor L3aaa_;
    ambit::Tensor L3aab_;
    ambit::Tensor L3abb_;
    ambit::Tensor L3bbb_;

public:

    // => Constructors <= //

    Reference();
    Reference(double Eref,ambit::Tensor L1a,ambit::Tensor L1b,ambit::Tensor L2aa,ambit::Tensor L2ab,ambit::Tensor L2bb);
    Reference(double Eref,ambit::Tensor L1a,ambit::Tensor L1b,ambit::Tensor L2aa,ambit::Tensor L2ab,ambit::Tensor L2bb,
              ambit::Tensor L3aaa,ambit::Tensor L3aab,ambit::Tensor L3abb,ambit::Tensor L3bbb);
    //Constructor for DMRG based reference.
//    Reference(std::string);
    ~Reference();

    double get_Eref() {return Eref_;}

    void set_L1a(ambit::Tensor L1a) {L1a_ = L1a;}
    void set_L1b(ambit::Tensor L1b) {L1b_ = L1b;}

    void set_L2aa(ambit::Tensor L2aa) {L2aa_ = L2aa;}
    void set_L2ab(ambit::Tensor L2ab) {L2ab_ = L2ab;}
    void set_L2bb(ambit::Tensor L2bb) {L2bb_ = L2bb;}

    void set_L3aaa(ambit::Tensor L3aaa) {L3aaa_ = L3aaa;}
    void set_L3aab(ambit::Tensor L3aab) {L3aab_ = L3aab;}
    void set_L3abb(ambit::Tensor L3abb) {L3abb_ = L3abb;}
    void set_L3bbb(ambit::Tensor L3bbb) {L3bbb_ = L3bbb;}

    ambit::Tensor L1a() {return L1a_;}
    ambit::Tensor L1b() {return L1b_;}
    ambit::Tensor L2aa() {return L2aa_;}
    ambit::Tensor L2ab() {return L2ab_;}
    ambit::Tensor L2bb() {return L2bb_;}
    ambit::Tensor L3aaa() {return L3aaa_;}
    ambit::Tensor L3aab() {return L3aab_;}
    ambit::Tensor L3abb() {return L3abb_;}
    ambit::Tensor L3bbb() {return L3bbb_;}
};

}} // End Namespaces

#endif // _reference_h_
