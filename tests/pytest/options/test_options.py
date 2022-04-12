import pytest
import forte
from forte import forte_options, ForteOptions


def test_options():

    forte.clean_options()

    # read an option via the ForteOption class interface
    test1 = forte_options.get_double('E_CONVERGENCE')

    # get the py::dict object in the ForteOption object
    d = forte_options.dict()

    # grab one variable
    e_conv = d['E_CONVERGENCE']

    # read option type and value
    assert test1 == 1e-09
    assert e_conv['type'] == 'float'
    assert e_conv['value'] == 1e-09

    # compare the value to the one obtained via the class interface
    assert e_conv['value'] == test1

    # write options via the python dictionary and test propagation to the ForteOption object
    e_conv['value'] = 1e-05
    test2 = forte_options.get_double('E_CONVERGENCE')
    assert e_conv['value'] == 1e-05
    assert test2 == 1e-05

    forte_options.add_bool('MY_BOOL', False, 'A boolean')
    forte_options.add_int('MY_INT', 0, 'An integer')
    forte_options.add_double('MY_FLOAT', 0, 'A float')
    forte_options.add_str('MY_STR', 0, 'A string')
    forte_options.add_int_list('MY_INT_LIST', 'A list of integers')
    forte_options.add_double_list('MY_FLOAT_LIST', 'A list of floating point numbers')
    forte_options.add_list('MY_GEN_LIST', 'A general list')

    # test setting an option from a dictionary
    forte_options.set_from_dict(
        {
            'MY_BOOL': True,
            'MY_INT': 2,
            'MY_FLOAT': 1.0e-12,
            'MY_STR': 'NEW STRING',  # this also tests conversion of string options to upper case
            'MY_INT_LIST': [1, 1, 2, 3, 5, 8],
            'MY_FLOAT_LIST': [1.0, 2.0, 3.0],
            'MY_GEN_LIST': ['singlet', 'triplet', 42]
        }
    )
    assert forte_options.get_bool('MY_BOOL')
    assert forte_options.get_int('MY_INT') == 2
    assert forte_options.get_double('MY_FLOAT') == 1.0e-12
    assert forte_options.get_str('MY_STR') == 'NEW STRING'
    assert forte_options.get_int_list('MY_INT_LIST') == [1, 1, 2, 3, 5, 8]
    assert forte_options.get_double_list('MY_FLOAT_LIST') == [1.0, 2.0, 3.0]
    assert forte_options.get_list('MY_GEN_LIST') == ['singlet', 'triplet', 42]

    forte_options.set_from_dict({'E_CONVERGENCE': 1.0e-12})
    assert forte_options.get_double('E_CONVERGENCE') == 1.0e-12

    # test setting an option with the wrong label
    with pytest.raises(RuntimeError):
        forte_options.set_from_dict({'E_CONVERGENCEE': 1.0e-12})

    # test setting an option with the wrong type (here list instead of float)
    with pytest.raises(RuntimeError):
        forte_options.set_from_dict({'E_CONVERGENCE': [1.0]})

    # create a new ForteOptions object
    new_options = forte.ForteOptions()
    # call set_dict to copy the dictionary from another ForteOptions
    # this does a deepcopy
    new_options.set_dict(forte_options.dict())

    # verify that the option value was copied
    assert new_options.get_double('E_CONVERGENCE') == 1.0e-12
    # verify that changing an option in one object does not affect the other
    new_options.set_double('E_CONVERGENCE', 1.0e-2)
    assert forte_options.get_double('E_CONVERGENCE') == 1.0e-12
    assert new_options.get_double('E_CONVERGENCE') == 1.0e-2

    # now reset new_options, define new options, and print them to test str()
    new_options = ForteOptions()
    new_options.add_bool('MY_BOOL', False, 'A boolean')
    new_options.add_int('MY_INT', 0, 'An integer')
    new_options.add_double('MY_FLOAT', 0, 'A float')
    new_options.add_str('MY_STR', 0, 'A string')
    new_options.add_int_list('MY_INT_LIST', 'A list of integers')
    new_options.add_double_list('MY_FLOAT_LIST', 'A list of floating point numbers')
    new_options.add_list('MY_GEN_LIST', 'A general list')
    new_options.add_int('MY_NONE', None, 'An integer')

    new_options.set_str('MY_STR', 'NEW STRING')

    new_options.set_from_dict(
        {
            'MY_BOOL': True,
            'MY_INT': 2,
            'MY_FLOAT': 1.0e-12,
            'MY_STR': 'NEW STRING',
            'MY_INT_LIST': [1, 1, 2, 3, 5, 8],
            'MY_FLOAT_LIST': [1.0, 2.0, 3.0],
            'MY_GEN_LIST': ['singlet', 'triplet', 42]
        }
    )

    test_str = """MY_BOOL: 1
MY_INT: 2
MY_FLOAT: 0.000000
MY_STR: NEW STRING
MY_INT_LIST: [1,1,2,3,5,8,]
MY_FLOAT_LIST: [1.000000,2.000000,3.000000,]
MY_GEN_LIST: gen_list()
MY_NONE: None
"""
    assert str(new_options) == test_str

    # test catching impossible type conversions
    with pytest.raises(RuntimeError):
        new_options.set_int_list('MY_FLOAT', [1, 2, 3])
    with pytest.raises(RuntimeError):
        new_options.set_double_list('MY_FLOAT', [1, 2, 3])
    with pytest.raises(RuntimeError):
        new_options.set_list('MY_FLOAT', [1, 2, 3])


if __name__ == "__main__":
    test_options()
