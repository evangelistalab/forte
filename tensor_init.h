#ifndef TENSOR_INIT_H
#define TENSOR_INIT_H

namespace psi{ namespace main{

class TENSOR_INIT
{
public:
    // Class constructor
    TENSOR_INIT();

    // Class destructor
    ~TENSOR_INIT();

    // Initialize a 4d tensor
    void init_matrix4d(double****& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4);

    // Free a 4d tensor
    void free_matrix4d(double****& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4);

    // Initialize a 6d tensor
    void init_matrix6d(double******& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4, const size_t &size5, const size_t &size6);

    // Free a 6d tensor
    void free_matrix6d(double******& matrix, const size_t &size1, const size_t &size2, const size_t &size3, const size_t &size4, const size_t &size5, const size_t &size6);

};

}}

#endif // TENSOR_INIT_H
