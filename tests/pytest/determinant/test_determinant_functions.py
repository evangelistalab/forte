import forte
import pytest


def test_determinant_hilber_space():
    dets = forte.hilbert_space(4, 2, 2, 1, [0, 0, 0, 0], 0)
    print(len(dets))


if __name__ == "__main__":
    test_determinant_hilber_space()
