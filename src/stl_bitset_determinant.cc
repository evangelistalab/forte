#include <psi4-dec.h>
#include <libmoinfo/libmoinfo.h>
#include <libmints/matrix.h>

#include <boost/lexical_cast.hpp>

#include "stl_bitset_determinant.h"
#include "fci_vector.h"

using namespace std;
using namespace psi;

namespace psi{ namespace forte{

// Static members
std::shared_ptr<FCIIntegrals> STLBitsetDeterminant::fci_ints_;

void STLBitsetDeterminant::set_ints(std::shared_ptr<FCIIntegrals> ints)
{
    fci_ints_ = ints;

//    // Initialize the bit masks
//    int n = ints->nmo();

//    bit_mask_.clear();
//    for (int i = 0; i < n; ++i){
//        bit_t b(n);
//        for (int j = 0; j < i; ++j){
//            b[j] = 1;
//        }
//        bit_mask_.push_back(b);
//    }
}

STLBitsetDeterminant::STLBitsetDeterminant() : nmo_(0)
{
}

STLBitsetDeterminant::STLBitsetDeterminant(int nmo)
    : nmo_(nmo), bits_(2 * nmo_)
{
}

/// Construct the determinant from an occupation vector that
/// specifies the alpha and beta strings.  occupation = [Ia,Ib]
STLBitsetDeterminant::STLBitsetDeterminant(const std::vector<int>& occupation)
    : nmo_(occupation.size() / 2), bits_(2 * nmo_)
{
    for(int p = 0; p < 2 * nmo_; ++p){
        bits_[p] = occupation[p];
    }
}

/// Construct the determinant from an occupation vector that
/// specifies the alpha and beta strings.  occupation = [Ia,Ib]
STLBitsetDeterminant::STLBitsetDeterminant(const std::vector<bool>& occupation)
    : nmo_(occupation.size() / 2), bits_(2 * nmo_)
{
    for(int p = 0; p < 2 * nmo_; ++p){
        bits_[p] = occupation[p];
    }
}

STLBitsetDeterminant::STLBitsetDeterminant(const std::vector<bool>& occupation_a,const std::vector<bool>& occupation_b)
    : nmo_(occupation_a.size()), bits_(2 * nmo_)
{
    for(int p = 0; p < nmo_; ++p){
        bits_[p] = occupation_a[p];
        bits_[p + nmo_] = occupation_b[p];
    }
}

std::vector<int> STLBitsetDeterminant::get_alfa_occ()
{
    std::vector<int> occ;
    for (int p = 0; p < nmo_; ++p){
        if (bits_[p]) occ.push_back(p);
    }
    return occ;
}

std::vector<int> STLBitsetDeterminant::get_beta_occ()
{
    std::vector<int> occ;
    for (int p = 0; p < nmo_; ++p){
        if (bits_[nmo_ + p]) occ.push_back(p);
    }
    return occ;
}

std::vector<int> STLBitsetDeterminant::get_alfa_vir()
{
    std::vector<int> vir;
    for (int p = 0; p < nmo_; ++p){
        if (not bits_[p]) vir.push_back(p);
    }
    return vir;
}

std::vector<int> STLBitsetDeterminant::get_beta_vir()
{
    std::vector<int> vir;
    for (int p = 0; p < nmo_; ++p){
        if (not bits_[nmo_ + p]) vir.push_back(p);
    }
    return vir;
}

std::vector<int> STLBitsetDeterminant::get_alfa_occ() const
{
    std::vector<int> occ;
    for (int p = 0; p < nmo_; ++p){
        if (bits_[p]) occ.push_back(p);
    }
    return occ;
}

std::vector<int> STLBitsetDeterminant::get_beta_occ() const
{
    std::vector<int> occ;
    for (int p = 0; p < nmo_; ++p){
        if (bits_[nmo_ + p]) occ.push_back(p);
    }
    return occ;
}

std::vector<int> STLBitsetDeterminant::get_alfa_vir() const
{
    std::vector<int> vir;
    for (int p = 0; p < nmo_; ++p){
        if (not bits_[p]) vir.push_back(p);
    }
    return vir;
}

std::vector<int> STLBitsetDeterminant::get_beta_vir() const
{
    std::vector<int> vir;
    for (int p = 0; p < nmo_; ++p){
        if (not bits_[nmo_ + p]) vir.push_back(p);
    }
    return vir;
}

double STLBitsetDeterminant::create_alfa_bit(int n)
{
    if (bits_[n]) return 0.0;
    bits_[n] = true;
    return SlaterSign(bits_,n);
}

double STLBitsetDeterminant::create_beta_bit(int n)
{
    if (bits_[nmo_ + n]) return 0.0;
    bits_[nmo_ + n] = true;
    return SlaterSign(bits_,nmo_ + n);
}

double STLBitsetDeterminant::destroy_alfa_bit(int n)
{
    if (not bits_[n]) return 0.0;
    bits_[n] = false;
    return SlaterSign(bits_,n);
}

/// Set the value of a beta bit
double STLBitsetDeterminant::destroy_beta_bit(int n)
{
    if (not bits_[nmo_ + n]) return 0.0;
    bits_[nmo_ + n] = false;
    return SlaterSign(bits_,nmo_ + n);
}

/// Switch alfa and beta bits
void STLBitsetDeterminant::spin_flip()
{
    for(int p = 0; p < nmo_; ++p){
        bool temp = bits_[p];
        bits_[p] = bits_[nmo_ + p];
        bits_[nmo_ + p] = temp;
    }
}

/**
 * Print the determinant
 */
void STLBitsetDeterminant::print() const
{
    outfile->Printf("\n  |");
    for(int p = 0; p < 2 * nmo_; ++p){
        outfile->Printf("%d",bits_[p] ? 1 :0);
    }
    outfile->Printf(">");
    outfile->Flush();
}

/**
 * Print the determinant
 */
std::string STLBitsetDeterminant::str() const
{
    std::string s;
    s += "|";
    for(int p = 0; p < nmo_; ++p){
        s += boost::lexical_cast<std::string>(bits_[p]);
    }
    s += "|";
    for(int p = 0; p < nmo_; ++p){
        s += boost::lexical_cast<std::string>(bits_[nmo_ + p]);
    }
    s += ">";
    return s;
}

/**
 * Compute the energy of this determinant
 * @return the electronic energy (does not include the nuclear repulsion energy)
 */
double STLBitsetDeterminant::energy() const
{
    double matrix_element = fci_ints_->frozen_core_energy();
    for(int p = 0; p < nmo_; ++p){
        if(bits_[p]){
            matrix_element += fci_ints_->oei_a(p,p);
            for(int q = p + 1; q < nmo_; ++q){
                if(bits_[q]){
                    matrix_element += fci_ints_->diag_tei_aa(p,q);
                }
            }
            for(int q = 0; q < nmo_; ++q){
                if(bits_[nmo_ + q]){
                    matrix_element += fci_ints_->diag_tei_ab(p,q);
                }
            }
        }
        if(bits_[nmo_ + p]){
            matrix_element += fci_ints_->oei_b(p,p);
            for(int q = p + 1; q < nmo_; ++q){
                if(bits_[nmo_ + q]){
                    matrix_element += fci_ints_->diag_tei_bb(p,q);
                }
            }
        }
    }
    return(matrix_element);
}

/**
 * Compute the matrix element of the Hamiltonian between this determinant and a given one
 * @param rhs
 * @return
 */
double STLBitsetDeterminant::slater_rules(const STLBitsetDeterminant& rhs) const
{
    const bit_t& I = bits_;
    const bit_t& J = rhs.bits_;

    int nadiff = 0;
    int nbdiff = 0;
    // Count how many differences in mos are there
    for (int n = 0; n < nmo_; ++n) {
        if (I[n] != J[n]) nadiff++;
        if (I[nmo_ + n] != J[nmo_ + n]) nbdiff++;
        if (nadiff + nbdiff > 4) return 0.0; // Get out of this as soon as possible
    }
    nadiff /= 2;
    nbdiff /= 2;

    double matrix_element = 0.0;
    // Slater rule 1 PhiI = PhiJ
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element = fci_ints_->frozen_core_energy();
        for(int p = 0; p < nmo_; ++p){
            if(bits_[p]) matrix_element += fci_ints_->oei_a(p,p);
            if(bits_[nmo_ +p]) matrix_element += fci_ints_->oei_b(p,p);
            for(int q = 0; q < nmo_; ++q){
                if(bits_[p] and bits_[q])
                    matrix_element +=   0.5 * fci_ints_->diag_tei_aa(p,q);
                //                    matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
                if(bits_[nmo_ +p] and bits_[nmo_ +q])
                    matrix_element +=   0.5 * fci_ints_->diag_tei_bb(p,q);
                //                    matrix_element +=   0.5 * ints_->diag_ce_rtei(p,q);
                if(bits_[p] and bits_[nmo_ +q])
                    matrix_element +=   fci_ints_->diag_tei_ab(p,q);
                //                    matrix_element += fci_ints_->diag_c_rtei(p,q);
            }
        }
    }

    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for(int p = 0; p < nmo_; ++p){
            if((I[p] != J[p]) and I[p]) i = p;
            if((I[p] != J[p]) and J[p]) j = p;
        }
        double sign = SlaterSign(I,i) * SlaterSign(J,j);
        matrix_element = sign * fci_ints_->oei_a(i,j);
        for(int p = 0; p < nmo_; ++p){
            if(I[p] and J[p]){
                matrix_element += sign * fci_ints_->tei_aa(i,p,j,p);
            }
            if(I[nmo_ + p] and J[nmo_ + p]){
                matrix_element += sign * fci_ints_->tei_ab(i,p,j,p);
            }
        }
    }
    // Slater rule 2 PhiI = j_b^+ i_b PhiJ
    if ((nadiff == 0) and (nbdiff == 1)) {
        // Diagonal contribution
        int i = 0;
        int j = 0;
        for(int p = 0; p < nmo_; ++p){
            if((I[nmo_ + p] != J[nmo_ + p]) and I[nmo_ + p]) i = p;
            if((I[nmo_ + p] != J[nmo_ + p]) and J[nmo_ + p]) j = p;
        }
        double sign = SlaterSign(I,nmo_ + i) * SlaterSign(J,nmo_ + j);
        matrix_element = sign * fci_ints_->oei_b(i,j);
        for(int p = 0; p < nmo_; ++p){
            if(I[p] and J[p]){
                matrix_element += sign * fci_ints_->tei_ab(p,i,p,j);
            }
            if(I[nmo_ + p] and J[nmo_ + p]){
                matrix_element += sign * fci_ints_->tei_bb(i,p,j,p);
            }
        }
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 2) and (nbdiff == 0)) {
        // Diagonal contribution
        int i = -1;
        int j =  0;
        int k = -1;
        int l =  0;
        for(int p = 0; p < nmo_; ++p){
            if((I[p] != J[p]) and I[p]){
                if (i == -1) { i = p; } else { j = p; }
            }
            if((I[p] != J[p]) and J[p]){
                if (k == -1) { k = p; } else { l = p; }
            }
        }
        double sign = SlaterSign(I,i) * SlaterSign(I,j) * SlaterSign(J,k) * SlaterSign(J,l);
        matrix_element = sign * fci_ints_->tei_aa(i,j,k,l);
    }

    // Slater rule 3 PhiI = k_a^+ l_a^+ j_a i_a PhiJ
    if ((nadiff == 0) and (nbdiff == 2)) {
        // Diagonal contribution
        int i,j,k,l;
        i = -1;
        j = -1;
        k = -1;
        l = -1;
        for(int p = 0; p < nmo_; ++p){
            if((I[nmo_ + p] != J[nmo_ + p]) and I[nmo_ + p]){
                if (i == -1) { i = p; } else { j = p; }
            }
            if((I[nmo_ + p] != J[nmo_ + p]) and J[nmo_ + p]){
                if (k == -1) { k = p; } else { l = p; }
            }
        }
        double sign = SlaterSign(I,nmo_ + i) * SlaterSign(I,nmo_ + j) * SlaterSign(J,nmo_ + k) * SlaterSign(J,nmo_ + l);
        matrix_element = sign * fci_ints_->tei_bb(i,j,k,l);
    }

    // Slater rule 3 PhiI = j_a^+ i_a PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Diagonal contribution
        int i,j,k,l;
        i = j = k = l = -1;
        for(int p = 0; p < nmo_; ++p){
            if((I[p] != J[p]) and I[p]) i = p;
            if((I[nmo_ + p] != J[nmo_ + p]) and I[nmo_ + p]) j = p;
            if((I[p] != J[p]) and J[p]) k = p;
            if((I[nmo_ + p] != J[nmo_ + p]) and J[nmo_ + p]) l = p;
        }
        double sign = SlaterSign(I,i) * SlaterSign(I,nmo_ + j) * SlaterSign(J,k) * SlaterSign(J,nmo_ + l);
        matrix_element = sign * fci_ints_->tei_ab(i,j,k,l);
    }
    return(matrix_element);
}

double STLBitsetDeterminant::slater_rules_single_alpha(int i, int a) const
{
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = SlaterSign(bits_,i) * SlaterSign(bits_,a) * (a > i ? -1.0 : 1.0);
    double matrix_element = fci_ints_->oei_a(i,a);
    for(int p = 0; p < nmo_; ++p){
        if(bits_[p]){
            matrix_element += fci_ints_->tei_aa(i,p,a,p);
        }
        if(bits_[nmo_ +p]){
            matrix_element += fci_ints_->tei_ab(i,p,a,p);
        }
    }
    return sign * matrix_element;
}

double STLBitsetDeterminant::slater_rules_single_beta(int i, int a) const
{
    // Slater rule 2 PhiI = j_a^+ i_a PhiJ
    double sign = SlaterSign(bits_,nmo_ + i) * SlaterSign(bits_,nmo_ + a) * (a > i ? -1.0 : 1.0);
    double matrix_element = fci_ints_->oei_b(i,a);
    for(int p = 0; p < nmo_; ++p){
        if(bits_[p]){
            matrix_element += fci_ints_->tei_ab(p,i,p,a);
        }
        if(bits_[nmo_ +p]){
            matrix_element += fci_ints_->tei_bb(i,p,a,p);
        }
    }
    return sign * matrix_element;
}

double STLBitsetDeterminant::slater_sign_alpha(int n) const
{
    double sign = 1.0;
    for(int i = 0; i < n; ++i){  // This runs up to the operator before n
        if(bits_[i]) sign *= -1.0;
    }
    return(sign);
}

double STLBitsetDeterminant::slater_sign_beta(int n) const
{
    double sign = 1.0;
    for(int i = 0; i < n; ++i){  // This runs up to the operator before n
        if(bits_[nmo_ + i]) sign *= -1.0;
    }
    return(sign);
}

double STLBitsetDeterminant::double_excitation_aa(int i, int j, int a, int b)
{
    double sign = 1.0;
    sign *= slater_sign_alpha(i);
    sign *= slater_sign_alpha(j);
    bits_[i] = false;
    bits_[j] = false;
    bits_[a] = true;
    bits_[b] = true;
    sign *= slater_sign_alpha(a);
    sign *= slater_sign_alpha(b);
    return sign;
}

double STLBitsetDeterminant::double_excitation_ab(int i, int j, int a, int b)
{
    double sign = 1.0;
    sign *= slater_sign_alpha(i);
    sign *= slater_sign_beta(j);
    bits_[i] = false;
    bits_[nmo_ + j] = false;
    bits_[a] = true;
    bits_[nmo_ + b] = true;
    sign *= slater_sign_alpha(a);
    sign *= slater_sign_beta(b);
    return sign;
}

double STLBitsetDeterminant::double_excitation_bb(int i, int j, int a, int b)
{
    double sign = 1.0;
    sign *= slater_sign_beta(i);
    sign *= slater_sign_beta(j);
    bits_[nmo_ + i] = false;
    bits_[nmo_ + j] = false;
    bits_[nmo_ + a] = true;
    bits_[nmo_ + b] = true;
    sign *= slater_sign_beta(a);
    sign *= slater_sign_beta(b);
    return sign;
}

/**
 * Compute the S^2 matrix element of the Hamiltonian between two determinants specified by the strings (Ia,Ib) and (Ja,Jb)
 * @return S^2
 */
double STLBitsetDeterminant::spin2(const STLBitsetDeterminant& rhs) const
{
    const bit_t& I = bits_;
    const bit_t& J = rhs.bits_;

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    int nadiff = 0;
    int nbdiff = 0;
    int na = 0;
    int nb = 0;
    int npair = 0;
    int nmo = nmo_;
    // Count how many differences in mos are there and the number of alpha/beta electrons
    for (int n = 0; n < nmo; ++n) {
        if (I[n] != J[n]) nadiff++;
        if (I[nmo_ + n] != J[nmo_ + n]) nbdiff++;
        if (I[n]) na++;
        if (I[nmo_ + n]) nb++;
        if ((I[n] and I[nmo_ + n])) npair += 1;
    }
    nadiff /= 2;
    nbdiff /= 2;

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta + Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here is a bit complex
        for(int p = 0; p < nmo; ++p){
            if(J[p] and I[nmo_ + p] and (not J[nmo_ + p]) and  (not I[p])) i = p; //(p)
            if(J[nmo_ + p] and I[p] and (not J[p]) and  (not I[nmo_ + p])) j = p; //(q)
        }
        if (i != j and i >= 0 and j >= 0){
            double sign = SlaterSign(J,i) * SlaterSign(J,nmo_ + j) * SlaterSign(I,nmo_ + i) * SlaterSign(I,j);
            matrix_element -= sign;
        }
    }
    return(matrix_element);
}

double STLBitsetDeterminant::SlaterSign(const bit_t& I,int n)
{
    double sign = 1.0;
    for(int i = 0; i < n; ++i){  // This runs up to the operator before n
        if(I[i]) sign *= -1.0;
    }
    return(sign);
}

}} // end namespace

