/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <cmath>

#include "mini-boost/boost/numeric/odeint.hpp"
#include "psi4/libmints/matrix.h"

#include "psi4/libdiis/diismanager.h"

#include "tensorsrg.h"

using namespace psi;

namespace psi {
namespace forte {

typedef std::vector<std::pair<double, std::pair<size_t, size_t>>> pair_t;
typedef std::map<std::pair<size_t, size_t>, size_t> pair_map_t;

void set_elements(Matrix& M, Tensor& T, pair_map_t& A_map, pair_map_t& B_map);

void TensorSRG::save_hbar() {
    throw std::runtime_error("TensorSRG::save_hbar() is not implemented yet!");

    /*
        std::string filename;
        outfile->Printf("\n  Saving Hbar\n");

        pair_t oO;
        pair_t oV;
        pair_t vV;
        pair_map_t oO_map;
        pair_map_t oV_map;
        pair_map_t vV_map;

    //    Tensor& Fa_oo = *F.block("oo");
    //    Tensor& Fa_vv = *F.block("vv");
    //    Tensor& Fb_OO = *F.block("OO");
    //    Tensor& Fb_VV = *F.block("VV");

        size_t naocc = a_occ_mos.size();
        size_t nbocc = b_occ_mos.size();
        size_t navir = a_vir_mos.size();
        size_t nbvir = b_vir_mos.size();

        size_t offset = 0;
        for (int i = 0; i < naocc; ++i){
            for (int J = 0; J < nbocc; ++J){
                double e = Fa_oo(i,i) + Fb_OO(J,J);
                oO.push_back(std::make_pair(e,std::make_pair(i,J)));
            }
        }
        std::sort(oO.begin(),oO.end());
        for (size_t n = 0; n < oO.size(); ++n){
            oO_map[oO[n].second] = n;
        }

        offset += oO.size();
        for (int i = 0; i < naocc; ++i){
            for (int B = 0; B < nbvir; ++B){
                double e = Fa_oo(i,i) + Fb_VV(B,B);
                oV.push_back(std::make_pair(e,std::make_pair(i,B)));
            }
        }
        std::sort(oV.begin(),oV.end());
        for (size_t n = 0; n < oV.size(); ++n){
            oV_map[oV[n].second] = n + offset + 1;
        }

        offset += oV.size();
        for (int a = 0; a < navir; ++a){
            for (int B = 0; B < nbvir; ++B){
                double e = Fa_vv(a,a) + Fb_VV(B,B);
                vV.push_back(std::make_pair(e,std::make_pair(a,B)));
            }
        }
        std::sort(vV.begin(),vV.end());
        for (size_t n = 0; n < vV.size(); ++n){
            vV_map[vV[n].second] = n + offset + 2;
        }

        for (size_t n = 0; n < oO.size(); ++n){
            outfile->Printf("%f %zu
    %zu\n",oO[n].first,oO[n].second.first,oO[n].second.second);
        }

        int dim = oO.size() + oV.size() + vV.size() + 2;
        Matrix hb("Hbar",dim,dim);


    //    {
    //    Tensor& Hbar_oOoO = *Hbar2.block("oOoO");
    //    Tensor::iterator it = Hbar_oOoO.begin();
    //    Tensor::iterator endit = Hbar_oOoO.end();
    //    for (; it != endit; ++it){
    //        std::vector<size_t>& i = it.address();
    //        std::pair<size_t,size_t> xx(i[0],i[1]);
    //        std::pair<size_t,size_t> yy(i[2],i[3]);
    //        int x = oO_map[xx];
    //        int y = oO_map[yy];
    //        hb.set(x,y,std::fabs(*it));
    //    }
    //    }
        set_elements(hb,*Hbar2.block("oOoO"),oO_map,oO_map);
        set_elements(hb,*Hbar2.block("oOoV"),oO_map,oV_map);
        set_elements(hb,*Hbar2.block("oOvV"),oO_map,vV_map);

        set_elements(hb,*Hbar2.block("oVoO"),oV_map,oO_map);
        set_elements(hb,*Hbar2.block("oVoV"),oV_map,oV_map);
        set_elements(hb,*Hbar2.block("oVvV"),oV_map,vV_map);

        set_elements(hb,*Hbar2.block("vVoO"),vV_map,oO_map);
        set_elements(hb,*Hbar2.block("vVoV"),vV_map,oV_map);
        set_elements(hb,*Hbar2.block("vVvV"),vV_map,vV_map);

        ofstream myfile;
        myfile.open ("hb.txt");
        for (int x = 0; x < dim; ++x){
            for (int y = 0; y < dim; ++y){
                myfile << std::max(std::log(hb.get(dim-x-1,y)),-20.0) << " ";
            }
            myfile << "\n";
        }
        myfile.close();
        */
}

// void set_elements(Matrix& M,Tensor& T, pair_map_t &A_map, pair_map_t &B_map){
//    Tensor::iterator it = T.begin();
//    Tensor::iterator endit = T.end();
//    for (; it != endit; ++it){
//        std::vector<size_t>& i = it.address();
//        std::pair<size_t,size_t> xx(i[0],i[1]);
//        std::pair<size_t,size_t> yy(i[2],i[3]);
//        int x = A_map[xx];
//        int y = B_map[yy];
//        M.set(x,y,std::fabs(*it));
//    }
//}
}
} // EndNamespaces
