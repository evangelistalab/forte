import re
import pytest
import numpy as np

from forte import ndarray, ndarray_from_numpy, ndarray_copy_from_numpy, DataType


def test_ndarray_creation():
    """Test creation of ndarray with a given shape."""
    tensor = ndarray((2, 3))
    assert tensor.shape == [2, 3]
    assert tensor.size == 6
    assert tensor.dtype.name == "float64"  # Assuming T is double
    assert tensor.dtype.kind == "f"
    assert tensor.dtype.itemsize == 8  # Size of double in bytes


def test_ndarray_fill_and_access():
    """Test filling the tensor and accessing elements."""
    tensor = ndarray((2, 2))
    tensor.fill(5.0)
    for i in range(2):
        for j in range(2):
            assert tensor.at([i, j]) == 5.0
            assert tensor[i, j] == 5.0  # Using __getitem__

    # Test set_at and __setitem__
    tensor.set_at([0, 1], 3.0)
    assert tensor.at(0, 1) == 3.0
    tensor[1, 0] = 2.0  # Using __setitem__
    assert tensor.at(1, 0) == 2.0


def test_ndarray_numpy_array():
    """Test the numpy_array method and data synchronization."""
    tensor = ndarray((2, 2))
    tensor.fill(1.0)
    nparray = tensor.numpy_array()
    assert isinstance(nparray, np.ndarray)
    assert nparray.shape == (2, 2)
    assert np.all(nparray == 1.0)

    # Modify NumPy array and check if changes reflect in tensor
    nparray[0, 0] = 7.0
    assert tensor.at(0, 0) == 7.0


def test_ndarray_from_numpy():
    """Test creating an ndarray from a NumPy array."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    tensor = ndarray_from_numpy(arr)
    assert tensor.shape == [2, 2]
    assert tensor.at(0, 0) == 1.0

    # Check if the tensor is iterable and verify the values using isclose
    assert all(np.isclose(i, j) for i, j in zip(tensor, [1.0, 2.0, 3.0, 4.0]))

    # Modify the original NumPy array and check if changes reflect in tensor
    arr[0, 1] = 5.0
    assert tensor.at(0, 1) == 5.0

    arr = np.array([[1, 2j], [3j, 4]], dtype=np.complex128)
    tensor = ndarray_from_numpy(arr)
    assert tensor.shape == [2, 2]
    assert tensor.at(0, 1) == 2.0j

    # Modify the original NumPy array and check if changes reflect in tensor
    arr[0, 1] = 5.0j
    assert tensor.at(0, 1) == 5.0j


def test_ndarray_indexing_errors():
    """Test indexing errors."""
    tensor = ndarray((2, 2))

    with pytest.raises(IndexError):
        tensor.at(2, 0)  # Index out of bounds

    with pytest.raises(IndexError):
        tensor.at([0])  # Incorrect number of indices

    with pytest.raises(IndexError):
        tensor.at([0, 0, 0])  # Incorrect number of indices


def test_ndarray_dtype():
    """Test the dtype property."""
    tensor = ndarray((2, 2))
    dtype = tensor.dtype
    assert isinstance(dtype, DataType)
    assert dtype.name == "float64"
    assert dtype.kind == "f"
    assert dtype.itemsize == 8


def test_ndarray_strides():
    """Test the strides of the ndarray."""
    tensor = ndarray((2, 3))
    assert tensor.strides == [3, 1]  # Assuming row-major order


def test_ndarray_to_string():
    """Test the string representation of the ndarray."""
    tensor = ndarray((2, 2))
    tensor.fill(1.0)
    s = str(tensor)
    assert "ndarray(shape=[2, 2], data=[1, 1, 1, 1]" in s


def test_ndarray_set_to():
    """Test setting the data of the ndarray."""
    tensor = ndarray((2, 3))
    values = [i for i in range(6)]
    tensor.set_to(values)
    for i in range(2):
        for j in range(3):
            assert tensor.at([i, j]) == values[i * 3 + j]


def test_ndarray_zero_shape():
    """Test an ndarray with zero dimensions (scalar)."""
    tensor = ndarray(())
    assert tensor.shape == []
    assert tensor.size == 1
    tensor.set_at([], 42.0)
    assert tensor.at([]) == 42.0


def test_ndarray_data_ownership():
    """Test that the ndarray properly manages data ownership with NumPy arrays."""
    tensor = ndarray((2, 2))
    nparray = tensor.numpy_array()
    del tensor  # The data should still be valid through nparray
    assert np.all(nparray == 0.0)
    nparray[0, 0] = 10.0
    assert nparray[0, 0] == 10.0


def test_ndarray_from_numpy_readonly():
    """Test creating an ndarray from a read-only NumPy array."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
    arr.setflags(write=False)
    with pytest.raises(RuntimeError):
        ndarray_from_numpy(arr)


def test_ndarray_type_mismatch():
    """Test creating an ndarray from a NumPy array with a mismatched data type."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    with pytest.raises(TypeError):
        ndarray_from_numpy(arr)


def test_ndarray_large_dimensions():
    """Test creating and working with a large ndarray."""
    tensor = ndarray((100, 100))
    tensor.fill(1.0)
    assert tensor.size == 10000
    nparray = tensor.numpy_array()
    assert nparray.shape == (100, 100)
    assert np.all(nparray == 1.0)


def test_ndarray_slicing():
    """Test slicing of the ndarray (if supported)."""
    tensor = ndarray((4, 4))
    tensor.set_to([i for i in range(16)])
    nparray = tensor.numpy_array()
    # Assuming that slicing is supported via numpy_array
    slice_tensor = nparray[1:3, 1:3]
    expected = np.array([[5, 6], [9, 10]], dtype=np.float64)
    assert np.array_equal(slice_tensor, expected)


def test_ndarray_math_operations():
    """Test basic mathematical operations."""
    tensor = ndarray((2, 2))
    tensor.set_to([1, 2, 3, 4])
    nparray = tensor.numpy_array()
    result = nparray * 2
    expected = np.array([[2, 4], [6, 8]], dtype=np.float64)
    assert np.array_equal(result, expected)


def test_ndarray_transpose():
    """Test transposing the ndarray."""
    tensor = ndarray((2, 3))
    tensor.set_to([1, 2, 3, 4, 5, 6])
    nparray = tensor.numpy_array()
    transposed = nparray.T
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64)
    assert np.array_equal(transposed, expected)


def test_ndarray_reshape():
    """Test reshaping the ndarray."""
    tensor = ndarray((2, 3))
    tensor.set_to([1, 2, 3, 4, 5, 6])
    nparray = tensor.numpy_array()
    reshaped = nparray.reshape((3, 2))
    expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    assert np.array_equal(reshaped, expected)


def test_ndarray_exception_messages():
    """Test that appropriate exceptions are raised with clear messages."""
    tensor = ndarray((2, 4))

    with pytest.raises(IndexError, match="Index 2 out of bounds for dimension 0 with size 2"):
        tensor.at(2, 0)

    with pytest.raises(IndexError, match="Index 5 out of bounds for dimension 1 with size 4"):
        tensor.at(0, 5)

    with pytest.raises(IndexError, match=re.escape("Rank 2 array addressed with incorrect number of indices (3)")):
        tensor.at(0, 5, 7)

    with pytest.raises(IndexError, match="Index 2 out of bounds for dimension 0 with size 2"):
        tensor[2, 0]

    with pytest.raises(IndexError, match="Incorrect number of indices. Expected 2 but got 1"):
        tensor[2]

    with pytest.raises(IndexError, match="Incorrect number of indices. Expected 2 but got 3"):
        tensor[2, 3, 6]


def test_ndarray_dtype_consistency():
    """Test that the dtype remains consistent across operations."""
    tensor = ndarray((2, 2))
    tensor.fill(1.0)
    nparray = tensor.numpy_array()
    assert nparray.dtype == np.float64


def test_ndarray_invalid_initialization():
    """Test initializing an ndarray with invalid parameters."""
    with pytest.raises(TypeError):
        ndarray("invalid_shape")

    with pytest.raises(TypeError):
        tensor = ndarray([-1, 2])  # Negative dimension size


def test_ndarray_copy():
    """Test copying the ndarray."""
    tensor = ndarray((2, 2))
    tensor.fill(1.0)
    nparray = tensor.numpy_array()
    nparray_copy = nparray.copy()
    nparray_copy[0, 0] = 5.0
    # Ensure original tensor is not affected
    assert tensor.at(0, 0) == 1.0


def test_ndarray_non_contiguous_memory():
    """Test behavior with non-contiguous memory (if applicable)."""
    tensor = ndarray((4, 4))
    tensor.set_to([i for i in range(16)])
    nparray = tensor.numpy_array()
    sliced = nparray[::2, ::2]  # Non-contiguous slice
    expected = np.array([[0, 2], [8, 10]], dtype=np.float64)
    assert np.array_equal(sliced, expected)


def test_ndarray_data_persistence():
    """Test that data persists as expected when objects are deleted."""
    tensor = ndarray((2, 2))
    tensor.fill(5.0)
    nparray = tensor.numpy_array()
    tensor_ref = tensor  # Keep a reference
    del tensor
    assert np.all(nparray == 5.0)
    del tensor_ref
    # nparray should still be valid
    assert np.all(nparray == 5.0)


def test_ndarray_complex_numbers():
    """Test ndarray with complex numbers (if supported)."""
    # Assuming there is a complex version of ndarray
    tensor = ndarray((2, 2), dtype="complex128")
    tensor.fill(complex(1, 1))
    nparray = tensor.numpy_array()
    expected = np.array([[1 + 1j, 1 + 1j], [1 + 1j, 1 + 1j]], dtype=np.complex128)
    assert np.array_equal(nparray, expected)


def test_ndarray_data_type_traits():
    """Test that DataType traits are correctly set for different types."""
    tensor = ndarray((2, 2))
    dtype = tensor.dtype
    assert dtype.name == "float64"
    assert dtype.kind == "f"
    assert dtype.itemsize == 8

    tensor = ndarray((2, 2), dtype="complex64")
    dtype = tensor.dtype
    assert dtype.name == "complex64"
    assert dtype.kind == "c"
    assert dtype.itemsize == 8


def test_ndarray_property_access():
    """Test access to properties like shape, strides, and dtype."""
    tensor = ndarray((3, 4, 5))
    assert tensor.shape == [3, 4, 5]
    assert tensor.dtype.name == "float64"
    assert isinstance(tensor.strides, list)
    assert len(tensor.strides) == 3


def test_ndarray_memory_leaks():
    """Test for memory leaks by monitoring object counts."""
    import gc

    tensor = ndarray((1000, 1000))
    tensor.fill(1.0)
    nparray = tensor.numpy_array()
    tensor_ref = tensor  # Keep a reference
    del tensor
    del nparray
    gc.collect()
    # Assuming we have a way to check for memory leaks, e.g., using tracemalloc
    # For this test, we just ensure no exceptions occur


def test_ndarray_copy_from_numpy():
    """Test copying the ndarray from numpy array."""
    nparray = np.array([[1, 2], [3, 4]], dtype=np.complex128)
    tensor = ndarray_copy_from_numpy(nparray)
    assert tensor.shape == [2, 2]
    assert tensor.dtype.name == "complex128"
    assert tensor.at(0, 0) == 1.0
    nparray[0, 1] = 5.0
    assert tensor.at(0, 1) == 2.0
    assert nparray[0, 1] == 5.0


if __name__ == "__main__":
    # call all the tests defined above
    test_ndarray_creation()  # 1
    test_ndarray_fill_and_access()  # 2
    test_ndarray_numpy_array()  # 3
    test_ndarray_from_numpy()  # 4
    test_ndarray_indexing_errors()  # 5
    test_ndarray_dtype()  # 6
    test_ndarray_strides()  # 7
    test_ndarray_to_string()  # 8
    test_ndarray_set_to()  # 9
    test_ndarray_zero_shape()  # 10
    test_ndarray_data_ownership()  # 11
    test_ndarray_from_numpy_readonly()  # 12
    test_ndarray_type_mismatch()  # 13
    test_ndarray_large_dimensions()  # 14
    test_ndarray_slicing()  # 15
    test_ndarray_math_operations()  # 16
    test_ndarray_transpose()  # 17
    test_ndarray_reshape()  # 18
    test_ndarray_exception_messages()  # 19
    test_ndarray_dtype_consistency()  # 20
    test_ndarray_invalid_initialization()  # 21
    test_ndarray_copy()  # 22
    test_ndarray_non_contiguous_memory()  # 23
    test_ndarray_data_persistence()  # 24
    test_ndarray_complex_numbers()  # 25
    test_ndarray_data_type_traits()  # 26
    test_ndarray_property_access()  # 27
    test_ndarray_memory_leaks()  # 28
    test_ndarray_copy_from_numpy()  # 29

    print("All tests passed!")
