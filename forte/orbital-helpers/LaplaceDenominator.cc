#include <regex>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>

#include "psi4/lib3index/dftensor.h"
#include "psi4/psifiles.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libqt/qt.h"
#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

#include "LaplaceDenominator.h"

// MKL Header
#ifdef USING_LAPACK_MKL
#include <mkl.h>
#endif

// OpenMP Header
//_OPENMP is defined by the compiler if it exists
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace psi;

namespace forte {

// the third parameter of from_string() should be
// one of std::hex, std::dec or std::oct
template <class T>
bool from_string(T &t, const std::string &s, std::ios_base &(*f)(std::ios_base &)) {
    std::istringstream iss(s);
    return !(iss >> f >> t).fail();
}


LaplaceDenominator::LaplaceDenominator(std::shared_ptr<psi::Vector> eps_occ, std::shared_ptr<psi::Vector> eps_vir, double delta)
    : eps_occ_(eps_occ), eps_vir_(eps_vir), delta_(delta) {
    decompose_ccvv();
}

LaplaceDenominator::LaplaceDenominator(std::shared_ptr<psi::Vector> eps_occ, std::shared_ptr<psi::Vector> eps_act, std::shared_ptr<psi::Vector> eps_vir, double delta, bool cavv)
    : eps_occ_(eps_occ), eps_vir_(eps_vir), eps_act_(eps_act), delta_(delta), cavv_(cavv) {
    if (cavv_) {
        decompose_cavv();
    } else {
        decompose_ccav();
    }
}

LaplaceDenominator::~LaplaceDenominator() {}

void LaplaceDenominator::decompose_ccvv() {
    outfile->Printf("\n\n  ==> FORTE Laplace Denominator <==\n\n");
    int nocc = eps_occ_->dimpi()[0];
    int nvir = eps_vir_->dimpi()[0];

    double E_LOMO = eps_occ_->get(0, 0);
    double E_HOMO = eps_occ_->get(0, nocc - 1);
    double E_LUMO = eps_vir_->get(0, 0);
    double E_HUMO = eps_vir_->get(0, nvir - 1);

    double A = 2.0 * (E_LUMO - E_HOMO);
    double B = 2.0 * (E_HUMO - E_LOMO);
    double R = B / A;

    // Pick appropriate quadrature file and read contents
    std::string PSIDATADIR = Process::environment.get_datadir();
    std::string err_table_filename = PSIDATADIR + "/quadratures/1_x/error.bin";
    std::string R_filename = PSIDATADIR + "/quadratures/1_x/R_avail.bin";

    std::ifstream err_table_file(err_table_filename.c_str(), std::ios::in | std::ios::binary);
    std::ifstream R_avail_file(R_filename.c_str(), std::ios::in | std::ios::binary);

    if (!err_table_file)
        throw PSIEXCEPTION(
            "LaplaceQuadrature: Cannot locate error property file for quadrature rules (should be "
            "PSIDATADIR/quadratures/1_x/error.bin)");
    if (!R_avail_file)
        throw PSIEXCEPTION(
            "LaplaceQuadrature: Cannot locate R property file for quadrature rules (should be "
            "PSIDATADIR/quadratures/1_x/R_avail.bin)");

    int nk = 53;
    int nR = 99;

    // Read in the R available
    auto *R_availp = new double[nR];
    R_avail_file.read((char *)R_availp, nR * sizeof(double));

    auto err_table = std::make_shared<psi::Matrix>("Error Table (nR x nk)", nR, nk);
    double **err_tablep = err_table->pointer();
    err_table_file.read((char *)err_tablep[0], static_cast<unsigned long> (nR) * nk * sizeof(double));

    R_avail_file.close();
    err_table_file.close();

    // for (int r2 = 0; r2 < nR; r2++)
    //    outfile->Printf( "  R[%4d] = %20.14E\n", r2+1, R_availp[r2]);
    // err_table->print();

    int indR;
    for (indR = 0; indR < nR; indR++) {
        if (R < R_availp[indR]) break;
    }
    if (indR == nR) {
        // TODO: Relax this
        throw PSIEXCEPTION(
            "Laplace Quadrature requested for (E_HUMO - E_LOMO)/(E_LUMO-E_HOMO) > 7.0 * 10^12, quadratures are not "
            "designed for this range.");
    }

    double accuracy;
    int k, r;
    bool found = false;
    for (k = 0; k < nk; k++) {
        for (r = indR; r < nR; r++) {
            double err = err_tablep[r][k];
            if (err != 0.0 && err < delta_) {
                accuracy = err;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    if (!found) {
        throw PSIEXCEPTION("Laplace Quadrature rule could not be found with specified accuracy for this system");
    }

    nvector_ = k + 1;

    // A bit hacky, but OK
    int exponent = (int)floor(log(R_availp[r]) / log(10.0));
    int mantissa = (int)round(R_availp[r] / pow(10.0, exponent));
    if (mantissa == 10) {
        exponent++;
        mantissa = 1;
    }

    std::stringstream st;
    st << std::setfill('0');
    st << "1_xk" << std::setw(2) << nvector_;
    st << "_" << mantissa;
    st << "E" << exponent;

    std::string quadfile = PSIDATADIR + "/quadratures/1_x/" + st.str().c_str();

    outfile->Printf("  This system has an intrinsic R = (E_HUMO - E_LOMO)/(E_LUMO - E_HOMO) of %7.4E.\n", R);
    outfile->Printf("  A %d point minimax quadrature with R of %1.0E will be used for the denominator.\n", nvector_,
                    R_availp[r]);
    outfile->Printf("  The worst-case Chebyshev norm for this quadrature rule is %7.4E.\n", accuracy);
    outfile->Printf("  Quadrature rule read from file %s.\n\n", quadfile.c_str());

    // The quadrature is defined as \omega_v exp(-\alpha_v x) = 1/x
    auto *alpha = new double[nvector_];
    auto *omega = new double[nvector_];

    std::vector<std::string> lines;
    std::string text;
    std::ifstream infile(quadfile.c_str());
    if (!infile) throw PSIEXCEPTION("LaplaceDenominator: Unable to open quadrature rule file: " + quadfile);
    while (infile.good()) {
        std::getline(infile, text);
        lines.push_back(text);
    }

#define NUMBER "((?:[-+]?\\d*\\.\\d+(?:[DdEe][-+]?\\d+)?)|(?:[-+]?\\d+\\.\\d*(?:[DdEe][-+]?\\d+)?))"
    std::regex numberline("^\\s*(" NUMBER ").*");
    std::smatch what;

    // We'll be rigorous, the files are extremely well defined
    int lineno = 0;
    for (int index = 0; index < nvector_; index++) {
        std::string line = lines[lineno++];
        if (!std::regex_match(line, what, numberline))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to read grid file line: \n" + line);
        if (!from_string<double>(omega[index], what[1], std::dec))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to convert grid file line: \n" + line);
    }
    for (int index = 0; index < nvector_; index++) {
        std::string line = lines[lineno++];
        if (!std::regex_match(line, what, numberline))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to read grid file line: \n" + line);
        if (!from_string<double>(alpha[index], what[1], std::dec))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to convert grid file line: \n" + line);
    }

    // for (int k = 0; k < nvector_; k++)
    //    printf("  %24.16E, %24.16E\n", omega[k], alpha[k]);

    // Cast weights back to problem size
    for (int k = 0; k < nvector_; k++) {
        alpha[k] /= A;
        omega[k] /= A;
    }

    denominator_occ_ = std::make_shared<psi::Matrix>("Occupied Laplace Delta Tensor", nvector_, nocc);
    denominator_vir_ = std::make_shared<psi::Matrix>("Virtual Laplace Delta Tensor", nvector_, nvir);
    // denominator_ = std::make_shared<Matrix>("OV Laplace Delta Tensor", nvector_, nocc * nvir);

    double **dop = denominator_occ_->pointer();
    double **dvp = denominator_vir_->pointer();
    // double **dovp = denominator_->pointer();

    double *e_o = eps_occ_->pointer();
    double *e_v = eps_vir_->pointer();

    for (int k = 0; k < nvector_; k++) {
        for (int i = 0; i < nocc; i++) {
            dop[k][i] = pow(omega[k], 0.25) * exp(alpha[k] * e_o[i]);
        }
        for (int a = 0; a < nvir; a++) {
            dvp[k][a] = pow(omega[k], 0.25) * exp(-alpha[k] * e_v[a]);
        }
        // for (int i = 0; i < nocc; i++) {
        //     for (int a = 0; a < nvir; a++) {
        //         dovp[k][i * nvir + a] = pow(omega[k], 0.5) * exp(-alpha[k] * (e_v[a] - e_o[i]));
        //     }
        // }
    }

    delete[] alpha;
    delete[] omega;
    delete[] R_availp;
    outfile->Printf("\n  ==>Finish: FORTE Laplace Denominator <==\n\n");
}

void LaplaceDenominator::decompose_cavv() {
    outfile->Printf("\n\n  ==> FORTE Laplace Denominator (CAVV) <==\n\n");
    int nocc = eps_occ_->dimpi()[0];
    int nvir = eps_vir_->dimpi()[0];
    int nact = eps_act_->dimpi()[0];

    // double E_LOMO = eps_occ_->get(0, 0);
    // double E_HOMO = eps_occ_->get(0, nocc - 1);
    // double E_LUMO = eps_vir_->get(0, 0);
    // double E_HUMO = eps_vir_->get(0, nvir - 1);

    // double A = 2.0 * (E_LUMO - E_HOMO); // min
    // double B = 2.0 * (E_HUMO - E_LOMO); // max

    double E_vir_max = eps_vir_->get(0, nvir - 1);
    double E_vir_min = eps_vir_->get(0, 0);
    double E_occ_max = eps_occ_->get(0, nocc - 1);
    double E_occ_min = eps_occ_->get(0, 0);
    double E_act_max = eps_act_->get(0, nact - 1);
    double E_act_min = eps_act_->get(0, 0);

    double A = 2 * E_vir_min - E_occ_max - E_act_max;
    double B = 2 * E_vir_max - E_occ_min - E_act_min;

    double R = B / A;

    // Pick appropriate quadrature file and read contents
    std::string PSIDATADIR = Process::environment.get_datadir();
    std::string err_table_filename = PSIDATADIR + "/quadratures/1_x/error.bin";
    std::string R_filename = PSIDATADIR + "/quadratures/1_x/R_avail.bin";

    std::ifstream err_table_file(err_table_filename.c_str(), std::ios::in | std::ios::binary);
    std::ifstream R_avail_file(R_filename.c_str(), std::ios::in | std::ios::binary);

    if (!err_table_file)
        throw PSIEXCEPTION(
            "LaplaceQuadrature: Cannot locate error property file for quadrature rules (should be "
            "PSIDATADIR/quadratures/1_x/error.bin)");
    if (!R_avail_file)
        throw PSIEXCEPTION(
            "LaplaceQuadrature: Cannot locate R property file for quadrature rules (should be "
            "PSIDATADIR/quadratures/1_x/R_avail.bin)");

    int nk = 53;
    int nR = 99;

    // Read in the R available
    auto *R_availp = new double[nR];
    R_avail_file.read((char *)R_availp, nR * sizeof(double));

    auto err_table = std::make_shared<psi::Matrix>("Error Table (nR x nk)", nR, nk);
    double **err_tablep = err_table->pointer();
    err_table_file.read((char *)err_tablep[0], static_cast<unsigned long> (nR) * nk * sizeof(double));

    R_avail_file.close();
    err_table_file.close();

    // for (int r2 = 0; r2 < nR; r2++)
    //    outfile->Printf( "  R[%4d] = %20.14E\n", r2+1, R_availp[r2]);
    // err_table->print();

    int indR;
    for (indR = 0; indR < nR; indR++) {
        if (R < R_availp[indR]) break;
    }
    if (indR == nR) {
        // TODO: Relax this
        throw PSIEXCEPTION(
            "Laplace Quadrature requested for (E_HUMO - E_LOMO)/(E_LUMO-E_HOMO) > 7.0 * 10^12, quadratures are not "
            "designed for this range.");
    }

    double accuracy;
    int k, r;
    bool found = false;
    for (k = 0; k < nk; k++) {
        for (r = indR; r < nR; r++) {
            double err = err_tablep[r][k];
            if (err != 0.0 && err < delta_) {
                accuracy = err;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    if (!found) {
        throw PSIEXCEPTION("Laplace Quadrature rule could not be found with specified accuracy for this system");
    }

    nvector_ = k + 1;

    // A bit hacky, but OK
    int exponent = (int)floor(log(R_availp[r]) / log(10.0));
    int mantissa = (int)round(R_availp[r] / pow(10.0, exponent));
    if (mantissa == 10) {
        exponent++;
        mantissa = 1;
    }

    std::stringstream st;
    st << std::setfill('0');
    st << "1_xk" << std::setw(2) << nvector_;
    st << "_" << mantissa;
    st << "E" << exponent;

    std::string quadfile = PSIDATADIR + "/quadratures/1_x/" + st.str().c_str();

    outfile->Printf("  This system has an intrinsic R = (E_HUMO - E_LOMO)/(E_LUMO - E_HOMO) of %7.4E.\n", R);
    outfile->Printf("  A %d point minimax quadrature with R of %1.0E will be used for the denominator.\n", nvector_,
                    R_availp[r]);
    outfile->Printf("  The worst-case Chebyshev norm for this quadrature rule is %7.4E.\n", accuracy);
    outfile->Printf("  Quadrature rule read from file %s.\n\n", quadfile.c_str());

    // The quadrature is defined as \omega_v exp(-\alpha_v x) = 1/x
    auto *alpha = new double[nvector_];
    auto *omega = new double[nvector_];

    std::vector<std::string> lines;
    std::string text;
    std::ifstream infile(quadfile.c_str());
    if (!infile) throw PSIEXCEPTION("LaplaceDenominator: Unable to open quadrature rule file: " + quadfile);
    while (infile.good()) {
        std::getline(infile, text);
        lines.push_back(text);
    }

#define NUMBER "((?:[-+]?\\d*\\.\\d+(?:[DdEe][-+]?\\d+)?)|(?:[-+]?\\d+\\.\\d*(?:[DdEe][-+]?\\d+)?))"
    std::regex numberline("^\\s*(" NUMBER ").*");
    std::smatch what;

    // We'll be rigorous, the files are extremely well defined
    int lineno = 0;
    for (int index = 0; index < nvector_; index++) {
        std::string line = lines[lineno++];
        if (!std::regex_match(line, what, numberline))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to read grid file line: \n" + line);
        if (!from_string<double>(omega[index], what[1], std::dec))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to convert grid file line: \n" + line);
    }
    for (int index = 0; index < nvector_; index++) {
        std::string line = lines[lineno++];
        if (!std::regex_match(line, what, numberline))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to read grid file line: \n" + line);
        if (!from_string<double>(alpha[index], what[1], std::dec))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to convert grid file line: \n" + line);
    }

    // for (int k = 0; k < nvector_; k++)
    //    printf("  %24.16E, %24.16E\n", omega[k], alpha[k]);

    // Cast weights back to problem size
    for (int k = 0; k < nvector_; k++) {
        alpha[k] /= A;
        omega[k] /= A;
    }

    denominator_occ_ = std::make_shared<psi::Matrix>("Occupied Laplace Delta Tensor", nvector_, nocc);
    denominator_vir_ = std::make_shared<psi::Matrix>("Virtual Laplace Delta Tensor", nvector_, nvir);
    denominator_act_ = std::make_shared<psi::Matrix>("Active Laplace Delta Tensor", nvector_, nact);

    double **dop = denominator_occ_->pointer();
    double **dvp = denominator_vir_->pointer();
    double **dap = denominator_act_->pointer();

    double *e_o = eps_occ_->pointer();
    double *e_v = eps_vir_->pointer();
    double *e_a = eps_act_->pointer();

    for (int k = 0; k < nvector_; k++) {
        for (int i = 0; i < nocc; i++) {
            dop[k][i] = pow(omega[k], 0.25) * exp(alpha[k] * e_o[i]);
        }
        for (int a = 0; a < nvir; a++) {
            dvp[k][a] = pow(omega[k], 0.25) * exp(-alpha[k] * e_v[a]);
        }
        for (int u = 0; u < nact; u++) {
            dap[k][u] = pow(omega[k], 0.25) * exp(alpha[k] * e_a[u]);
        }
    }

    delete[] alpha;
    delete[] omega;
    delete[] R_availp;
    outfile->Printf("\n  ==>Finish: FORTE Laplace Denominator (CAVV) <==\n\n");
}

void LaplaceDenominator::decompose_ccav() {
    outfile->Printf("\n\n  ==> FORTE Laplace Denominator (CCAV) <==\n\n");
    int nocc = eps_occ_->dimpi()[0];
    int nvir = eps_vir_->dimpi()[0];
    int nact = eps_act_->dimpi()[0];

    // double E_LOMO = eps_occ_->get(0, 0);
    // double E_HOMO = eps_occ_->get(0, nocc - 1);
    // double E_LUMO = eps_vir_->get(0, 0);
    // double E_HUMO = eps_vir_->get(0, nvir - 1);

    // double A = 2.0 * (E_LUMO - E_HOMO); // min
    // double B = 2.0 * (E_HUMO - E_LOMO); // max

    double E_vir_max = eps_vir_->get(0, nvir - 1);
    double E_vir_min = eps_vir_->get(0, 0);
    double E_occ_max = eps_occ_->get(0, nocc - 1);
    double E_occ_min = eps_occ_->get(0, 0);
    double E_act_max = eps_act_->get(0, nact - 1);
    double E_act_min = eps_act_->get(0, 0);

    double A = E_vir_min + E_act_min - 2 * E_occ_max;
    double B = E_vir_max + E_act_max - 2 * E_occ_min;

    double R = B / A;

    // Pick appropriate quadrature file and read contents
    std::string PSIDATADIR = Process::environment.get_datadir();
    std::string err_table_filename = PSIDATADIR + "/quadratures/1_x/error.bin";
    std::string R_filename = PSIDATADIR + "/quadratures/1_x/R_avail.bin";

    std::ifstream err_table_file(err_table_filename.c_str(), std::ios::in | std::ios::binary);
    std::ifstream R_avail_file(R_filename.c_str(), std::ios::in | std::ios::binary);

    if (!err_table_file)
        throw PSIEXCEPTION(
            "LaplaceQuadrature: Cannot locate error property file for quadrature rules (should be "
            "PSIDATADIR/quadratures/1_x/error.bin)");
    if (!R_avail_file)
        throw PSIEXCEPTION(
            "LaplaceQuadrature: Cannot locate R property file for quadrature rules (should be "
            "PSIDATADIR/quadratures/1_x/R_avail.bin)");

    int nk = 53;
    int nR = 99;

    // Read in the R available
    auto *R_availp = new double[nR];
    R_avail_file.read((char *)R_availp, nR * sizeof(double));

    auto err_table = std::make_shared<psi::Matrix>("Error Table (nR x nk)", nR, nk);
    double **err_tablep = err_table->pointer();
    err_table_file.read((char *)err_tablep[0], static_cast<unsigned long> (nR) * nk * sizeof(double));

    R_avail_file.close();
    err_table_file.close();

    // for (int r2 = 0; r2 < nR; r2++)
    //    outfile->Printf( "  R[%4d] = %20.14E\n", r2+1, R_availp[r2]);
    // err_table->print();

    int indR;
    for (indR = 0; indR < nR; indR++) {
        if (R < R_availp[indR]) break;
    }
    if (indR == nR) {
        // TODO: Relax this
        throw PSIEXCEPTION(
            "Laplace Quadrature requested for (E_HUMO - E_LOMO)/(E_LUMO-E_HOMO) > 7.0 * 10^12, quadratures are not "
            "designed for this range.");
    }

    double accuracy;
    int k, r;
    bool found = false;
    for (k = 0; k < nk; k++) {
        for (r = indR; r < nR; r++) {
            double err = err_tablep[r][k];
            if (err != 0.0 && err < delta_) {
                accuracy = err;
                found = true;
                break;
            }
        }
        if (found) break;
    }

    if (!found) {
        throw PSIEXCEPTION("Laplace Quadrature rule could not be found with specified accuracy for this system");
    }

    nvector_ = k + 1;

    // A bit hacky, but OK
    int exponent = (int)floor(log(R_availp[r]) / log(10.0));
    int mantissa = (int)round(R_availp[r] / pow(10.0, exponent));
    if (mantissa == 10) {
        exponent++;
        mantissa = 1;
    }

    std::stringstream st;
    st << std::setfill('0');
    st << "1_xk" << std::setw(2) << nvector_;
    st << "_" << mantissa;
    st << "E" << exponent;

    std::string quadfile = PSIDATADIR + "/quadratures/1_x/" + st.str().c_str();

    outfile->Printf("  This system has an intrinsic R = (E_HUMO - E_LOMO)/(E_LUMO - E_HOMO) of %7.4E.\n", R);
    outfile->Printf("  A %d point minimax quadrature with R of %1.0E will be used for the denominator.\n", nvector_,
                    R_availp[r]);
    outfile->Printf("  The worst-case Chebyshev norm for this quadrature rule is %7.4E.\n", accuracy);
    outfile->Printf("  Quadrature rule read from file %s.\n\n", quadfile.c_str());

    // The quadrature is defined as \omega_v exp(-\alpha_v x) = 1/x
    auto *alpha = new double[nvector_];
    auto *omega = new double[nvector_];

    std::vector<std::string> lines;
    std::string text;
    std::ifstream infile(quadfile.c_str());
    if (!infile) throw PSIEXCEPTION("LaplaceDenominator: Unable to open quadrature rule file: " + quadfile);
    while (infile.good()) {
        std::getline(infile, text);
        lines.push_back(text);
    }

#define NUMBER "((?:[-+]?\\d*\\.\\d+(?:[DdEe][-+]?\\d+)?)|(?:[-+]?\\d+\\.\\d*(?:[DdEe][-+]?\\d+)?))"
    std::regex numberline("^\\s*(" NUMBER ").*");
    std::smatch what;

    // We'll be rigorous, the files are extremely well defined
    int lineno = 0;
    for (int index = 0; index < nvector_; index++) {
        std::string line = lines[lineno++];
        if (!std::regex_match(line, what, numberline))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to read grid file line: \n" + line);
        if (!from_string<double>(omega[index], what[1], std::dec))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to convert grid file line: \n" + line);
    }
    for (int index = 0; index < nvector_; index++) {
        std::string line = lines[lineno++];
        if (!std::regex_match(line, what, numberline))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to read grid file line: \n" + line);
        if (!from_string<double>(alpha[index], what[1], std::dec))
            throw PSIEXCEPTION("LaplaceDenominator: Unable to convert grid file line: \n" + line);
    }

    // for (int k = 0; k < nvector_; k++)
    //    printf("  %24.16E, %24.16E\n", omega[k], alpha[k]);

    // Cast weights back to problem size
    for (int k = 0; k < nvector_; k++) {
        alpha[k] /= A;
        omega[k] /= A;
    }

    denominator_occ_ = std::make_shared<psi::Matrix>("Occupied Laplace Delta Tensor", nvector_, nocc);
    denominator_vir_ = std::make_shared<psi::Matrix>("Virtual Laplace Delta Tensor", nvector_, nvir);
    denominator_act_ = std::make_shared<psi::Matrix>("Active Laplace Delta Tensor", nvector_, nact);

    double **dop = denominator_occ_->pointer();
    double **dvp = denominator_vir_->pointer();
    double **dap = denominator_act_->pointer();

    double *e_o = eps_occ_->pointer();
    double *e_v = eps_vir_->pointer();
    double *e_a = eps_act_->pointer();

    for (int k = 0; k < nvector_; k++) {
        for (int i = 0; i < nocc; i++) {
            dop[k][i] = pow(omega[k], 0.25) * exp(alpha[k] * e_o[i]);
        }
        for (int a = 0; a < nvir; a++) {
            dvp[k][a] = pow(omega[k], 0.25) * exp(-alpha[k] * e_v[a]);
        }
        for (int u = 0; u < nact; u++) {
            dap[k][u] = pow(omega[k], 0.25) * exp(-alpha[k] * e_a[u]);
        }
    }

    delete[] alpha;
    delete[] omega;
    delete[] R_availp;
    outfile->Printf("\n  ==>Finish: FORTE Laplace Denominator (CCAV) <==\n\n");
}
    
} // namespace forte