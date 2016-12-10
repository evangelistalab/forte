#include "psi4/libmints/basisset.h"
#include "psi4/psi4-dec.h"
namespace psi { namespace forte {

class ParallelDFMO {
    public:
        ParallelDFMO(std::shared_ptr<BasisSet> primary, std::shared_ptr<BasisSet> auxiliary);
        void set_C(std::shared_ptr<Matrix> C)
        {
            Ca_ = C;
        }
        void compute_integrals();
        int Q_PQ(){return GA_Q_PQ_;}
    protected:
        SharedMatrix Ca_;
        /// (A | Q)^{-1/2}
        void J_one_half();
        ///Compute (A|mn) integrals (distribute via mn indices)
        void transform_integrals();
        /// (A | pq) (A | Q)^{-1/2}

        std::shared_ptr<BasisSet> primary_;
        std::shared_ptr<BasisSet> auxiliary_;

        /// Distributed DF (Q | pq) integrals
        int GA_Q_PQ_;
        /// GA for J^{-1/2}
        int GA_J_onehalf_;
        
        size_t memory_;
        size_t nmo_;
};
}}
