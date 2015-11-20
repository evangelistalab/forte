#include "alternativescasscf.h"
#include <libpsio/psio.h>
#include <libmints/wavefunction.h>
#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <vector>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <liboptions/liboptions.h>
#include <libmoinfo/libmoinfo.h>
#include <libmints/molecule.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <libfock/jk.h>
#include "helpers.h"
#include <numeric>

namespace psi { namespace forte {

FiniteTemperatureHF::FiniteTemperatureHF(boost::shared_ptr<Wavefunction> wfn, Options& options, std::shared_ptr<MOSpaceInfo> mo_space)
    : RHF(options, _default_psio_lib_),
      wfn_(wfn),
      mo_space_info_(mo_space),
      options_(options)
{
    //copy(wfn);
    startup();
}

void FiniteTemperatureHF::startup()
{
    sMat_ = wfn_->S();
    hMat_ = wfn_->H();
    nmo_  = mo_space_info_->size("ALL");
    na_   = mo_space_info_->size("ACTIVE");
    rdocc_  = mo_space_info_->size("INACTIVE_DOCC");


    rdocc_dim_ = mo_space_info_->get_dimension("INACTIVE_DOCC");
    active_dim_ = mo_space_info_->get_dimension("ACTIVE");
    rdocc_p_active_ = rdocc_dim_ + active_dim_;

    print_method_banner({"Finite Temperature Hartree-Fock","Kevin Hannon"});
    outfile->Printf("\n\n Running a FT-HF computation");
    outfile->Printf("\n\n The variable occupation is restricted to active space");

    size_t total_electrons = wfn_->nalpha();
    outfile->Printf("\n %d", total_electrons);
    size_t naelec = wfn_->nalpha() - rdocc_;

    outfile->Printf("\n There are %d doubly occupied orbitals", rdocc_);
    outfile->Printf("\n There are %d active electrons", naelec);
    outfile->Printf("\n There are %d active orbitals", na_);
    eps_ = wfn_->epsilon_a();
    nirrep_ = wfn_->nirrep();
    SharedMatrix D = wfn_->Da();

    debug_ = options_.get_int("PRINT");
}

double FiniteTemperatureHF::compute_energy()
{
    /// Get the active_orb_energy into a vector
    std::vector<double> dirac(nmo_);
    fermidirac_ = dirac;
    ///Initialize some things
    /// Set occupation vector to 2 for RDOCC, 1 for ACTIVE, and 0 for RUOCC
    initialize_occupation_vector(fermidirac_);

    form_G();
    scf_energy_ = 0.0;
    scf_energy_ = RHF::compute_energy();
    outfile->Printf("\n FT-HF orbitals converged");
    outfile->Printf("\n SCF_ENERGY = %8.8f", scf_energy_);

    return scf_energy_;
}
void FiniteTemperatureHF::frac_occupation()
{
    double T = 0.0;
    T = options_.get_double("TEMPERATURE");
    if(debug_)
    {
        outfile->Printf("\n Running a Temperature of %8.8f", T);
    }
    T /= 3.157746E5;

    std::vector<double> ni(na_);
    ///The bisection root finding is to find the E_f value that satisfies
    /// the sum of occupations equal number of active orbitals
    if(na_ > 0)
    {
        ef_ = bisection(ni, T);
    }
    auto active_vector = mo_space_info_->get_absolute_mo("ACTIVE");
    int count = 0;
    ///Fill the occupation for active with variable occupations
    for(auto& active_array : active_vector)
    {
        fermidirac_[active_array] = ni[count];
        count++;
    }

    Dimension nmopi = mo_space_info_->get_dimension("ALL");
    Dimension doubly_occupied_sym = mo_space_info_->get_dimension("INACTIVE_DOCC");
    Dimension active_sym          = mo_space_info_->get_dimension("ACTIVE");
    /// The Dirac with symmetry
    SharedVector Dirac_sym(new Vector("Dirac_Symmetry", nirrep_, nmopi));

    int doubly_occupied = 0;
    int offset = 0;
    for(int h = 0; h < nirrep_; h++)
    {
        for(int double_sym = 0; double_sym < doubly_occupied_sym[h]; double_sym++)
        {
           Dirac_sym->set(h, double_sym, 1.0);
        }
        doubly_occupied = doubly_occupied_sym[h];
        for(int a = 0; a < active_sym[h]; a++)
        {
           Dirac_sym->set(h, a + doubly_occupied, fermidirac_[a + offset]);
        }
        offset += nmopi[h];
    }
    SharedMatrix C  = wfn_->Ca();
    SharedMatrix C_nochange = C->clone();


    SharedMatrix C_rd_a(new Matrix("C_rdocc_active", nirrep_, doubly_occupied_sym + active_sym, doubly_occupied_sym + active_sym));
    SharedMatrix C_rd_a_2(new Matrix("C_nochange", nirrep_, doubly_occupied_sym + active_sym, doubly_occupied_sym + active_sym));
    Dimension rd_a = doubly_occupied_sym + active_sym;
    ///Scale the columns with the occupation.
    /// This C matrix will be passed to JK object for CLeft
    for(int h = 0; h < nirrep_; h++)
    {
        for(int mu = 0; mu < nmopi[h]; mu++)
        {
            C->scale_column(h, mu, Dirac_sym->get(h, mu));
        }
        for(int ra = 0; ra < rd_a[h]; ra++)
        {
            for(int ra_b = 0; ra_b < rd_a[h]; ra_b++)
            {
                C_rd_a->set(h, ra, ra_b, C->get(h, ra, ra_b));
                C_rd_a_2->set(h, ra, ra_b, C_nochange->get(h, ra, ra_b));
            }

        }
    }
    C_occ_folded_ = C_rd_a;
    C_occ_a_ = C_rd_a_2;
}
void FiniteTemperatureHF::initialize_occupation_vector(std::vector<double>& dirac)
{
    auto double_occupied_vector = mo_space_info_->get_absolute_mo("INACTIVE_DOCC");
    for(auto& docc_array : double_occupied_vector)
    {
        dirac[docc_array] = 1.0;
    }
    auto unoccpied_vector = mo_space_info_->get_absolute_mo("INACTIVE_UOCC");
    nuocc_ = mo_space_info_->size("INACTIVE_UOCC");
    for(auto& uocc_array : unoccpied_vector)
    {
        dirac[uocc_array] = 0.0;
    }
    auto active_vector = mo_space_info_->get_absolute_mo("ACTIVE");
    for(auto& active_array : active_vector)
    {
        dirac[active_array] = 0.5;
    }

}
std::vector<double> FiniteTemperatureHF::get_active_orbital_energy()
{
    Dimension inactive_docc_dim = mo_space_info_->get_dimension("INACTIVE_DOCC");
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    int nirrep = wfn_->nirrep();
    std::vector<double> na_vec(na_);
    int offset = 0;
    int count =  0;
    for(int h = 0; h < nirrep; h++){
        offset = inactive_docc_dim[h];
        for(int a = 0; a < active_dim[h]; a++)
        {
            na_vec[count] = eps_->get(h, a + offset);
            count++;
        }

    }
    return na_vec;
}
double FiniteTemperatureHF::bisection(std::vector<double> & ni, double T)
{
    double ef1 = active_orb_energy_[0];
    double ef2 = active_orb_energy_[active_orb_energy_.size() - 1];
    size_t naelec = wfn_->nalpha() - rdocc_;
    if(na_ == 0)
    {
        outfile->Printf("\n This code is not designed to work with zero active space");
    //    throw PSIEXCEPTION(" The active space is zero.  Set the active space");
    }

    std::vector<double> nibisect(na_);

    nibisect = ni;
    double sum, sumef, sumef1;
    int iter = 0.0;
    double ef = 0.0;
    /// The number of iterations needed for bisection to converge
    /// (b - a) / 2^n <= tolerance
    double iterations = fabs(log(1e-10) / log(fabs(ef2 - ef1)));
    outfile->Printf("\n %8.8f", std::ceil(iterations));
    int max_iter = std::ceil(iterations);

    if(debug_)
    {
        outfile->Printf("\n In Bisection function HAMO = %6.3f  LAMO = %6.3f\n", ef1, ef2);
        outfile->Printf("\n Bisection should converged in %d iterations", max_iter);
        outfile->Printf("\n Iterations NA   ERROR   E_f");
    }
    while(iter < 100)
    {
        ef = ef1 + (ef2 - ef1)/2.0;

        sum = 0.0;
        sum = occ_vec(nibisect, ef, T);

        if(std::fabs((sum - naelec)) < 1e-6 && std::fabs(ef2 - ef1)/2.0 < 1e-6)
        {
             break;
        }
        if(debug_)
        {
            outfile->Printf("\n %d %d %8.8f  %8.8f", iter, naelec, std::fabs(sum - naelec), ef);
        }

        iter++;

        sumef = 0.0;
        sumef = occ_vec(nibisect, ef, T);

        sumef1 = 0.0;
        sumef1 = occ_vec(nibisect, ef1, T);

        auto sign = [](double a, double b) {return a * b > 0; };
        if(sign((sumef - naelec), (sumef1 - naelec)) == true)
        {
             ef1 = ef;
        }
        else{
             ef2 = ef;
        }


    }
    if(std::fabs((sum - naelec)) >  1e-6)
    {
        outfile->Printf("\n Bisection did not converge");
        outfile->Printf("\n Bisection gives %8.8f", sum);
        outfile->Printf("\n While it should be %d", naelec);

        throw PSIEXCEPTION(" Bisection root finding method failed ");

    }

    sumef = 0.0;
    ni = nibisect;
    int count = 0;
    if(debug_)
    {
        for(auto occupancy: nibisect){
            sumef+=occupancy;
            count++;
            outfile->Printf("\n occupancy[%d]=%10.10f", count, occupancy);
        }
    }

    outfile->Printf("\n ef = %6.5f sum = %6.5f \n", ef, sumef);
    return ef;
}
double FiniteTemperatureHF::occ_vec(std::vector<double>& nibisect, double ef, double T)
{
    double sum = 0.0;
    if(T > 0.0000000001)
    {
        for(size_t i = 0; i <nibisect.size(); i++){
            //Fermi Dirac distribution - 1.0 / (1.0 + exp(\beta (e_i - ef)))
            double fi = 1.0/(1.0 + exp(1.0/(0.99994*T)*(active_orb_energy_[i] - ef)));
            nibisect[i] = fi;
            //outfile->Printf("\n fi(T, e_i, ef) = fi(%8.8f, %8.8f, %8.8f) = %8.8f", T, active_orb_energy_[i], ef, fi);
            sum+=nibisect[i];
        }
    }
    //else
    //{
    //    double ef_noT = (active_orb_energy_[naelec_ - 1] + active_orb_energy_[naelec_]) / 2;
    //    for(size_t i = 0; i < na_; i++)
    //    {
    //        if(active_orb_energy_[i] < ef_noT){
    //            nibisect[i] = 1.0;
    //            sum+=nibisect[i];
    //        }
    //        else{
    //            nibisect[i] = 0.0;
    //        }
    //    }
    //}

    return sum;
}
void FiniteTemperatureHF::form_G()
{
    if(na_ > 0)
    {
        active_orb_energy_ = get_active_orbital_energy();
    }
    frac_occupation();
    form_D();
    boost::shared_ptr<JK> JK_core = JK::build_JK();

    JK_core->set_memory(Process::environment.get_memory() * 0.8);
    JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_core->C_left();
    std::vector<boost::shared_ptr<Matrix> >&Cr = JK_core->C_right();

    Cl.clear();
    if(na_ > 0)
    {
        Cl.push_back(C_occ_folded_);
    }
    else { Cl.push_back(C_occ_a_);}

    Cr.clear();
    Cr.push_back(C_occ_a_);

    JK_core->compute();

    SharedMatrix J_core = JK_core->J()[0];
    SharedMatrix K_core = JK_core->K()[0];

    J_core->scale(2.0);
    SharedMatrix F_core = J_core->clone();
    F_core->subtract(K_core);
    G_->copy(F_core);
    const std::vector<boost::shared_ptr<Matrix> >&D = JK_core->D();
    D[0]->print();

}
void FiniteTemperatureHF::form_D()
{
    Dimension nsopi  = wfn_->nsopi();
    Dimension nmopi  = wfn_->nmopi();
    Dimension doccpi = rdocc_p_active_;
    if(na_ < 1)
    {
        C_occ_a_ = wfn_->Ca();
        C_occ_folded_ = C_occ_a_;
    }

    for (int h = 0; h < nirrep_; ++h) {
        int nso = nsopi[h];
        int nmo = nmopi[h];
        int na = doccpi[h];

        if (nso == 0 || nmo == 0 || na == 0) continue;

        double** Ca_left = C_occ_folded_->pointer(h);
        double** Ca_right = C_occ_a_->pointer(h);
        double** D = D_->pointer(h);

        if (na == 0)
            memset(static_cast<void*>(D[0]), '\0', sizeof(double)*nso*nso);
        C_DGEMM('N','T',nso,nso,na,1.0,Ca_left[0],nmo,Ca_right[0],nmo,0.0,D[0],nso);

    }

}


}}
