.. _`sec:running_forte`:

Running Forte
=============

.. sectionauthor:: Francesco A. Evangelista


Obtaining Forte
---------------

You can download the source code of Forte from
`GitHub <https://github.com/evangelistalab/forte>`_.

To clone the latest version of the repository run::

    git clone https://github.com/evangelistalab/forte.git forte


Compiling Forte
---------------

Once you have the current versions of Psi4, CMake, and Ambit, follow these
instructions to install Forte:

1. Run psi4 in the Forte folder::

    psi4 --plugin-compile

Psi4 will generate a CMake command for building Forte that looks like::

    cmake -C /usr/local/psi4/stage/usr/local/psi4/share/cmake/psi4/psi4PluginCache.cmake
        -DCMAKE_PREFIX_PATH=/usr/local/psi4/stage/usr/local/psi4

2. Run the cmake command generated in 1. appending the location of Ambit's cmake files (via the ``-Dambit_DIR option``)::

    cmake -C /usr/local/psi4/stage/usr/local/psi4/share/cmake/psi4/psi4PluginCache.cmake
        -DCMAKE_PREFIX_PATH=/usr/local/psi4/stage/usr/local/psi4 .
        -Dambit_DIR=<ambit-bin-dir>/share/cmake/ambit


3. Run make::

    make  


Setting up the ``PYTHONPATH``
-----------------------------

If Forte in installed in the folder ``/<path>/forte``, then ``PYTHONPATH`` should
contain ``/path/``. Note that if you include ``/<path>/forte`` in ``PYTHONPATH``
you will get an error (see Frequently asked questions).

Running the test cases
----------------------

After compiling and setting up ``PYTHONPATH``, you can run the test cases::

    cd tests/methods
    python run_forte_tests.py

Frequently asked questions
--------------------------

"ImportError: dynamic module does not define init function (initforte)"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure that your ``PYTHONPATH`` does not include the Forte directory.
That is, if Forte is in ``/<path>/forte`` then ``PYTHONPATH`` should
contain ``/path/`` and not ``/<path>/forte``.

