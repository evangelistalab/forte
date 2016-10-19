#ifndef _dsrg_source_h_
#define _dsrg_source_h_

#include <cmath>

namespace psi{ namespace forte{

class DSRG_SOURCE {
public:
    /**
     * DSRG_SOURCE Constructor
     * @param s The flow parameter
     * @param taylor_threshold The threshold for Taylor expansion
     */
    DSRG_SOURCE(double s, double taylor_threshold);

    /// Bare effect of source operator
    virtual double compute_renormalized(const double& D) = 0;
    /// Renormalize denominator
    virtual double compute_renormalized_denominator(const double& D) = 0;

protected:
    /// Flow parameter
    double s_;
    /// Smaller than which we will do Taylor expansion
    double taylor_threshold_;
};

/// Standard source
class STD_SOURCE : public DSRG_SOURCE {
public:
    /// Constructor
    STD_SOURCE(double s, double taylor_threshold);

    /// Return exp(-s * D^2)
    virtual double compute_renormalized(const double& D) {
        return std::exp(-s_ * std::pow(D, 2.0));
    }

    /// Return [1 - exp(-s * D^2)] / D
    virtual double compute_renormalized_denominator(const double& D) {
        double Z = std::sqrt(s_) * D;
        if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
            return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
        }else{
            return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
        }
    }

private:
    /// Order of the Taylor expansion
    int taylor_order_ = static_cast<int>(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    /// Taylor Expansion of [1 - exp(- Z^2)] / Z
    double Taylor_Exp(const double& Z, const int& n){
        if(n > 0){
            double value = Z, tmp = Z;
            for(int x = 0; x < n - 1; ++x){
                tmp *= -1.0 * std::pow(Z, 2.0) / (x + 2);
                value += tmp;
            }
            return value;
        }else{return 0.0;}
    }
};

/// Linear absolute exponential source
class LABS_SOURCE : public DSRG_SOURCE {
public:
    /// Constructor
    LABS_SOURCE(double s, double taylor_threshold);

    /// Return exp(-s * |D|)
    virtual double compute_renormalized(const double& D) {
        return std::exp(-s_ * std::fabs(D));
    }

    /// Return [1 - exp(-s * |D|)] / D
    virtual double compute_renormalized_denominator(const double& D) {
        double Z = s_ * D;
        if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
            return Taylor_Exp_Linear(Z, taylor_order_ * 2) * s_;
        }else{
            return (1.0 - std::exp(-s_ * std::fabs(D))) / D;
        }
    }

private:
    /// Order of the Taylor expansion
    int taylor_order_ = static_cast<int>(15.0 / taylor_threshold_ + 1) + 1;

    /// Taylor Expansion of [1 - exp(-|Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n){
        double Zabs = std::fabs(Z);
        if(n > 0){
            double value = 1.0, tmp = 1.0;
            for(int x = 0; x < n - 1; ++x){
                tmp *= -1.0 * Zabs / (x + 2);
                value += tmp;
            }
            if(Z >= 0.0){
                return value;
            }else{
                return -value;
            }
        }else{return 0.0;}
    }
};

/// Dyson source
class DYSON_SOURCE : public DSRG_SOURCE {
public:
    /// Constructor
    DYSON_SOURCE(double s, double taylor_threshold);

    /// Return 1.0 / (1.0 + s * D^2)
    virtual double compute_renormalized(const double& D) {
        return 1.0 / (1.0 + s_ * D * D);
    }

    /// Return s * D / (1.0 + s * D^2)
    virtual double compute_renormalized_denominator(const double& D) {
        return s_ * D / (1.0 + s_ * D * D);
    }
};
class MP2_SOURCE : public DSRG_SOURCE {
public:
    MP2_SOURCE(double s, double taylor_threshold);
    

    virtual double compute_renormalized(const double& D) {
        return 1.0;
    }
    
    virtual double compute_renormalized_denominator(const double& D) {
        return 1.0 / D ;
    }
};

}}

#endif // DSRG_SOURCE_H
