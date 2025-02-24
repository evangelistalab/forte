#!/usr/bin/env python
# -*- coding: utf-8 -*-

import forte


def test_sparse_vector():
    import pytest
    import forte
    from forte import det
    import math

    ### Overlap tests ###
    ref = forte.SparseState({det(""): 1.0, det("+"): 1.0, det("-"): 1.0, det("2"): 1.0, det("02"): 1.0})
    ref2 = forte.SparseState({det("02"): 0.3})
    ref3 = forte.SparseState({det("002"): 0.5})
    assert forte.overlap(ref, ref) == pytest.approx(5.0, abs=1e-9)
    assert forte.overlap(ref, ref2) == pytest.approx(0.3, abs=1e-9)
    assert forte.overlap(ref2, ref) == pytest.approx(0.3, abs=1e-9)
    assert forte.overlap(ref, ref3) == pytest.approx(0.0, abs=1e-9)

    ref_str = ref.str(2)

    ### Number projection tests ###
    proj1 = forte.SparseState({det("2"): 1.0, det("02"): 1.0})
    test_proj1 = forte.apply_number_projector(1, 1, ref)
    assert proj1 == test_proj1

    proj2 = forte.SparseState({det(""): 1.0})
    test_proj2 = forte.apply_number_projector(0, 0, ref)
    assert proj2 == test_proj2

    proj3 = forte.SparseState({det("+"): 1.0})
    test_proj3 = forte.apply_number_projector(1, 0, ref)
    assert proj3 == test_proj3

    proj4 = forte.SparseState({det("-"): 1.0})
    test_proj4 = forte.apply_number_projector(0, 1, ref)
    assert proj4 == test_proj4

    ref4 = forte.SparseState({det("+"): 1, det("-"): 1})
    ref4 = forte.normalize(ref4)
    assert ref4.norm() == pytest.approx(1.0, abs=1e-9)
    assert forte.spin2(ref4, ref4) == pytest.approx(0.75, abs=1e-9)

    ref5 = forte.SparseState({det("2"): 1})
    assert forte.spin2(ref5, ref5) == pytest.approx(0, abs=1e-9)


if __name__ == "__main__":
    test_sparse_vector()
