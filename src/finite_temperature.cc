#include "finite_temperature.h"
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

    print_method_banner({"Finite Temperature Hartree-Fock","Kevin Hannon"});

    eps_ = wfn_->epsilon_a();
    nirrep_ = wfn_->nirrep();

    debug_ = options_.get_int("PRINT");
    SharedMatrix C(wfn_->Ca()->clone());
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

    ///It seems that HF class does not actually copy Ca into Process::Environment
    SharedMatrix Ca = wfn_->Ca();
    Ca->copy(Ca_);
    wfn_->Cb()->copy(Ca);

    print_h2("FT-HF Converged");
    outfile->Printf("\n  FT-SCF = %12.16f\n\n", scf_energy_);

    return scf_energy_;
}
void FiniteTemperatureHF::frac_occupation()
{
    double T = 0.0;
    T = options_.get_double("TEMPERATURE");
    if(debug_ > 1)
    {
        outfile->Printf("\n Running a Temperature of %8.8f", T);
    }
    T /= 3.157746E5;

    std::vector<double> ni(nmo_);
    if(nmo_ > 0)
    {
        ef_ = bisection(ni, T);
    }
    auto active_vector = mo_space_info_->get_absolute_mo("ALL");
    ///Fill the occupation for active with variable occupations
    for(auto& active_array : active_vector)
    {
        fermidirac_[active_array] = ni[active_array];
    }

    Dimension nmopi = mo_space_info_->get_dimension("ALL");
    SharedVector Dirac_sym(new Vector("Dirac_Symmetry", nirrep_, nmopi));

    int offset = 0;
    Dimension occupation(nirrep_);
    for(int h = 0; h < nirrep_; h++)
    {
        int nonzero_occupation = 0;
        for(int p = 0; p < nmopi[h]; p++)
        {
           Dirac_sym->set(h, p, fermidirac_[p + offset]);
           if(fermidirac_[p + offset] > 1e-6)
           {
               nonzero_occupation++;

               occupation[h] = nonzero_occupation;
           }
        }
        offset += nmopi[h];
    }

    SharedMatrix C(new Matrix("C_matrix", wfn_->nsopi(), occupation));
    SharedMatrix Call(wfn_->Ca()->clone());


    Dimension nsopi = wfn_->nsopi();
    SharedMatrix C_scaled(new Matrix("C_rdocc_active", nirrep_, nsopi, occupation));
    SharedMatrix C_no_scale(new Matrix("C_nochange", nirrep_, nsopi , occupation));
    ///Scale the columns with the occupation.
    /// This C matrix will be passed to JK object for CLeft
    for(int h = 0; h < nirrep_; h++)
    {
        for(int mu = 0; mu < occupation[h]; mu++)
        {
            C_scaled->set_column(h, mu, Call->get_column(h, mu));
            C_no_scale->set_column(h, mu, Call->get_column(h, mu));
            C_scaled->scale_column(h, mu, Dirac_sym->get(h, mu));
        }
    }
    C_occ_folded_ = C_scaled;
    C_occ_a_ = C_no_scale;
}
void FiniteTemperatureHF::initialize_occupation_vector(std::vector<double>& dirac)
{
    auto nmo_vector = mo_space_info_->get_absolute_mo("ALL");
    for(auto& active_array : nmo_vector)
    {
        dirac[active_array] = 1.0;
    }

}
std::vector<std::pair<double, int> > FiniteTemperatureHF::get_active_orbital_energy()
{
    int nirrep = wfn_->nirrep();
    Dimension nmopi = mo_space_info_->get_dimension("ALL");
    std::vector<std::pair<double, int> > nmo_vec;
    int offset = 0;
    int count =  0;
    for(int h = 0; h < nirrep; h++){
        for(int p = 0; p < nmopi[h]; p++)
        {
            nmo_vec.push_back(std::make_pair(eps_->get(h, p), p + offset));
        }
        offset += nmopi[h];

    }
    std::sort(nmo_vec.begin(), nmo_vec.end(), [](const std::pair<double, int> &left,const std::pair<double, int> &right) {
        return left.first < right.first;
    });

    return nmo_vec;
}
double FiniteTemperatureHF::bisection(std::vector<double> & ni, double T)
{
    size_t naelec = wfn_->nalpha();
    double ef1 = active_orb_energy_[naelec - 1].first;
    double ef2 = active_orb_energy_[naelec].first;

    std::vector<double> nibisect(nmo_);

    nibisect = ni;
    double sum, sumef, sumef1;
    int iter = 0.0;
    double ef = 0.0;
    /// The number of iterations needed for bisection to converge
    /// (b - a) / 2^n <= tolerance
    double iterations = fabs(log(1e-10) / log(fabs(ef2 - ef1)));
    int max_iter = std::ceil(iterations);

    if(debug_ > 1)
    {
        outfile->Printf("\n In Bisection function HAMO = %6.3f  LAMO = %6.3f\n", ef1, ef2);
        outfile->Printf("\n Bisection should converged in %d iterations", max_iter);
        outfile->Printf("\n Iterations NA   ERROR   E_f");
    }
    while(iter < 500)
    {
        ef = ef1 + (ef2 - ef1)/2.0;

        sum = 0.0;
        sum = occ_vec(nibisect, ef, T);

        if(std::fabs((sum - naelec)) < 1e-2 || std::fabs(ef2 - ef1)/2.0 < 1e-6)
        {
             break;
        }
        if(debug_ > 1)
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
    if(std::fabs((sum - naelec)) >  1e-2)
    {
        outfile->Printf("\n Bisection did not converge");
        outfile->Printf("\n Bisection gives %8.8f", sum);
        outfile->Printf("\n While it should be %d", naelec);

        throw PSIEXCEPTION(" Bisection root finding method failed ");

    }

    sumef = 0.0;
    ni = nibisect;
    int count = 0;
    if(debug_ > 2)
    {
        for(auto occupancy: nibisect){
            sumef+=occupancy;
            count++;
            outfile->Printf("\n occupancy[%d]=%10.10f", count, occupancy);
        }
    }

    return ef;
}
double FiniteTemperatureHF::occ_vec(std::vector<double>& nibisect, double ef, double T)
{
    double sum = 0.0;
    for(size_t i = 0; i <nibisect.size(); i++){
        //Fermi Dirac distribution - 1.0 / (1.0 + exp(\beta (e_i - ef)))
        double fi = 1.0/(1.0 + exp(1.0/(0.99994*T)*(active_orb_energy_[i].first - ef)));

        nibisect[active_orb_energy_[i].second] = fi;
        sum+=nibisect[i];
    }

    return sum;
}
void FiniteTemperatureHF::form_G()
{
    if(nmo_ > 0)
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
    if(nmo_ > 0)
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
}
void FiniteTemperatureHF::form_D()
{
    D_ = Matrix::doublet(C_occ_folded_, C_occ_a_, false, true);
}


}}
