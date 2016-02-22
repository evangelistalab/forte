#ifndef _dsrg_time_h_
#define _dsrg_time_h_

#include <vector>
#include <string>
#include <map>

#include <libtrans/integraltransform.h>

namespace psi{ namespace forte{

class DSRG_TIME
{
public:
    /// Constructor
    DSRG_TIME();

    /// Accumulate timings
    void add(const std::string& code, const double& t);

    /// Subtract timings
    void subtract(const std::string& code, const double& t);

    /// Reset timings
    void reset();                           // reset all timings to zero
    void reset(const std::string& code);    // reset timing of the code

    /// Create info of a code
    void create_code(const std::string& code);

    /// Delete info of a code
    void delete_code(const std::string& code);

    /// Print summary for with default code
    void print_comm_time();

    /// Print the timing in a generic way
    void print();
    void print(const std::string& code);

    /// Clear all the private variables
    void clear(){
        code_.clear();
        code_to_tidx_.clear();
        timing_.clear();
    }

private:
    /**
     * commutator: [ H, T ] = C
     * time code: 1st digit: rank of H
     *            2nd digit: rank of T
     *            3rd digit: rank of C
     */
    std::vector<std::string> code_;

    /// Map from code to values
    std::map<std::string, int> code_to_tidx_;

    /// Timings for commutators
    std::vector<double> timing_;

    /// Test code
    bool test_code(const std::string& code);
};

}}

#endif // DSRG_TIME_H
