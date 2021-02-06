#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_mospaceinfo2():
    """Test the MOSpaceInfo python API"""
    import psi4
    import forte
    from forte import forte_options

    print("Testing the MOSpaceInfo python API")

# *
# * Irrep                 A1(0)       A2(1)    B1(2)   B2(3)
# *
# * Indexing:
# * ALL              | 0 1 2 3 4 | 5 6 7 8 9 | 10 11 | 12 13 | <- absolute index in the full orbital space
# * CORRELATED       | - 0 1 2 3 | - 4 5 6 7 |  8  - |  9 10 | <- absolute index in the space of non-frozen orbitals
# * RELATIVE         | 0 1 2 3 4 | 0 1 2 3 4 |  0  1 |  0  1 | <- index relative to the irrep in the full orbital space
# *
# * FROZEN_DOCC        *           *
# * RESTRICTED_DOCC      *           * *        *       *
# * GAS1                   *             *
# * GAS2                     *
# * RESTRICED_UOCC             *           *               *
# * FROZEN_UOCC                                    *

    # Setup forte and prepare the active space integral class
    mos_spaces = {'FROZEN_DOCC' :     [1,1,0,0],
                  'RESTRICTED_DOCC' : [1,2,1,1],
                  'GAS1' :            [1,1,0,0],
                  'GAS2' :            [1,0,0,0],
                  'RESTRICTED_UOCC' : [1,1,0,1],
                  }

    nmopi = psi4.core.Dimension([5,5,2,2])
    point_group = 'C2V'

    mo_space_info = forte.make_mo_space_info_from_map(nmopi,point_group,mos_spaces,[])

    space_names = ['FROZEN_DOCC','RESTRICTED_DOCC','GAS1','GAS2','GAS3','GAS4','GAS5','GAS6','RESTRICTED_UOCC','FROZEN_UOCC']

    assert mo_space_info.nirrep() == 4

    assert mo_space_info.space_names() == space_names

    ref_spaces = ['FROZEN_DOCC','RESTRICTED_DOCC','GAS1','GAS2','GAS3','GAS4','GAS5','GAS6','RESTRICTED_UOCC','FROZEN_UOCC','ACTIVE']

    ref_size = [2,5,2,1,0,0,0,0,3,1,3]
    for name, val in zip(ref_spaces,ref_size):
        assert mo_space_info.size(name) == val

    ref_dimension = [(1,1,0,0),(1,2,1,1),(1,1,0,0),(1,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(1,1,0,1),(0,0,1,0),(2,1,0,0)]
    for name, val in zip(ref_spaces,ref_dimension):
        assert mo_space_info.dimension(name).to_tuple() == val

    ref_absolute_mo = [[0, 5],[1, 6, 7, 10, 12],[2, 8],[3],[],[],[],[],[4, 9, 13],[11],[2, 3, 8]]
    for name, val in zip(ref_spaces,ref_absolute_mo):
        assert mo_space_info.absolute_mo(name) == val

    ref_corr_absolute_mo = [[1000000000, 1000000000],[0, 4, 5, 8, 9],[1, 6],[2],[],[],[],[],[3, 7, 10],[1000000000],[1, 2, 6]]
    for name, val in zip(ref_spaces,ref_corr_absolute_mo):
        assert mo_space_info.corr_absolute_mo(name) == val

    ref_relative_mo = [[(0, 0), (1, 0)],[(0, 1), (1, 1), (1, 2), (2, 0), (3, 0)],[(0, 2), (1, 3)],[(0, 3)],[],[],[],[],[(0, 4), (1, 4), (3, 1)],[(2, 1)],[(0, 2), (0, 3), (1, 3)]]
    for name, val in zip(ref_spaces,ref_relative_mo):
        assert mo_space_info.relative_mo(name) == val

    assert mo_space_info.pos_in_space('GAS1','ACTIVE') == [0, 2]
    assert mo_space_info.pos_in_space('GAS2','ACTIVE') == [1]
    assert mo_space_info.pos_in_space('ACTIVE','ACTIVE') == [0,1,2]
    assert mo_space_info.pos_in_space('ACTIVE','ALL') == [2, 3, 8]

    for name, val in zip(ref_spaces,ref_absolute_mo):
        assert mo_space_info.pos_in_space(name,'ALL') == val

    ref_spaces_non_frozen = ['RESTRICTED_DOCC','GAS1','GAS2','GAS3','GAS4','GAS5','GAS6','RESTRICTED_UOCC','ACTIVE']
    ref_corr_absolute_mo_non_frozen = [[0, 4, 5, 8, 9],[1, 6],[2],[],[],[],[],[3, 7, 10],[1, 2, 6]]

    for name, val in zip(ref_spaces_non_frozen,ref_corr_absolute_mo_non_frozen):
        assert mo_space_info.pos_in_space(name,'CORRELATED') == val

if __name__ == '__main__':
    test_mospaceinfo2()
