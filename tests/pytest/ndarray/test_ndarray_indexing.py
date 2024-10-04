import pytest

from forte import ndarray


def test_ndarray_fill_and_access_1d():
    """Test filling the tensor and accessing elements of a 1D tensor."""
    tensor = ndarray((5,))
    tensor.fill(3.0)
    for i in range(5):
        print(i)
        tensor[i] = i
        assert tensor.at([i]) == i
        assert tensor.at(i) == i
        assert tensor[i] == i


def test_ndarray_fill_and_access_2d():
    """Test filling the tensor and accessing elements of a 2D tensor."""
    tensor = ndarray((5, 7))
    tensor.fill(3.0)
    for i in range(5):
        for j in range(7):
            print(i, j)
            tensor[i, j] = i + j * 5
            assert tensor.at([i, j]) == i + j * 5
            assert tensor.at(i, j) == i + j * 5
            assert tensor[i, j] == i + j * 5


def test_ndarray_fill_and_access_3d():
    """Test filling the tensor and accessing elements of a 3D tensor."""
    tensor = ndarray((5, 7, 3))
    tensor.fill(3.0)
    for i in range(5):
        for j in range(7):
            for k in range(3):
                print(i, j, k)
                tensor[i, j, k] = i + j * 5 + k * 7
                assert tensor.at([i, j, k]) == i + j * 5 + k * 7
                assert tensor.at(i, j, k) == i + j * 5 + k * 7
                assert tensor[i, j, k] == i + j * 5 + k * 7


if __name__ == "__main__":
    test_ndarray_fill_and_access_1d()
    test_ndarray_fill_and_access_2d()
