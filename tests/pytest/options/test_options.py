import pytest
import forte
from forte import forte_options


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
    forte_options.add_double_array('MY_FLOAT_LIST', 'A float list')

    # test setting an option from a dictionary
    forte_options.set_from_dict(
        {
            'MY_BOOL': True,
            'MY_INT': 2,
            'MY_FLOAT': 1.0e-12,
            'MY_STR': 'New string',
            'MY_FLOAT_LIST': [1.0, 2.0, 3.0]
        }
    )
    assert forte_options.get_bool('MY_BOOL')
    assert forte_options.get_int('MY_INT') == 2
    assert forte_options.get_double('MY_FLOAT') == 1.0e-12
    assert forte_options.get_str('MY_STR') == 'New string'
    assert forte_options.get_double_vec('MY_FLOAT_LIST') == [1.0, 2.0, 3.0]

    forte_options.set_from_dict({'E_CONVERGENCE': 1.0e-12})
    assert forte_options.get_double('E_CONVERGENCE') == 1.0e-12

    # test setting an option with the wrong label
    with pytest.raises(RuntimeError):
        forte_options.set_from_dict({'E_CONVERGENCEE': 1.0e-12})

    # test setting an option with the wrong type (here list instead of float)
    with pytest.raises(RuntimeError):
        forte_options.set_from_dict({'E_CONVERGENCE': [1.0]})


if __name__ == "__main__":
    test_options()
