#!/usr/bin/env python
# -*- coding: utf-8 -*-

def test_options():
    import forte
    from forte import forte_options

    # read an option via the ForteOption class interface
    test1 = forte_options.get_double('E_CONVERGENCE')

    # get the py::dict object in the ForteOption object
    d = forte_options.dict()

    # grab one variable
    e_conv = d['E_CONVERGENCE']

    # read option type and value
    assert e_conv['type'] == 'float'
    assert e_conv['value'] == 1e-09

    # compare the value to the one obtained via the class interface
    assert e_conv['value'] == test1

    # write options via the python dictionary and test propagation to the ForteOption object
    e_conv['value'] = 1e-08
    test2 = forte_options.get_double('E_CONVERGENCE')
    assert e_conv['value'] == test2

test_options()
