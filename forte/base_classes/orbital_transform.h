#ifndef _orbital_transform_h_
#define _orbital_transform_h_

namespace psi {
class Matrix;
} // namespace psi

namespace forte {

class ForteIntegrals;
class MOSpaceInfo;
class SCFInfo;
class ForteOptions;

class OrbitalTransform {

  public:
    /// Constructor
    OrbitalTransform(std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Default constructor
    OrbitalTransform() = default;

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~OrbitalTransform() = default;

    virtual void compute_transformation() = 0;

    std::shared_ptr<psi::Matrix> get_Ua() { return Ua_; };

    std::shared_ptr<psi::Matrix> get_Ub() { return Ub_; };

  protected:
    // The integrals
    std::shared_ptr<ForteIntegrals> ints_;
    /// The MOSpace info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// @brief Unitary matrix for alpha orbital rotations
    std::shared_ptr<psi::Matrix> Ua_;
    /// @brief Unitary matrix for beta orbital rotations
    std::shared_ptr<psi::Matrix> Ub_;
};

std::unique_ptr<OrbitalTransform>
make_orbital_transformation(const std::string& type, std::shared_ptr<SCFInfo> scf_info,
                            std::shared_ptr<ForteOptions> options,
                            std::shared_ptr<ForteIntegrals> ints,
                            std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif
