Writing Forte’s documentation
=============================

Location and structure of Forte’s documentation
-----------------------------------------------

Forte uses sphinx to generate its documentation. The documentation is
written in part in sphinx, with some of the content generated from
Jupyter notebooks. The documentation is contained in the directory
``docs``, which has the following structure:

::

   docs
   ├── notebooks
   └── source

``source`` contains the restructured text files (rst) that are compiled
by sphinx. The directory ``notebooks`` contains Jupyter notebooks that
are used to generate some of the rst files. Restructured text file
prefixed with ``nb_`` that live in ``source`` are generated from jupyter
notebooks contained in the ``notebooks`` directory.

Note that the location of these converted jupyter notebooks reflects the
relative location in the ``notebooks`` directory. For example, the file
``docs/source/nb_00_overview.rst`` is generated from the file
``docs/notebooks/nb_00_overview.ipynb``.

Compiling the documentation
---------------------------

To compile the documentation on your local machine, from a terminal
change to the ``docs`` folder and type

.. code:: bash

   docs> make html

This command will run sphinx and generate the documentation in the
folder ``docs/build/html``. The documentation main page can be accessed
via web browser using the url ``docs/build/html/index.html``

Contributing to the documentation
---------------------------------

To modify a section of Forte’s documentation you should first identify
which file to modify. If a ``rst`` file begins with ``nb_``, then you
should edit the corresponding jupyter notebook located in
``docs/notebooks`` or one of its subdirectories.

If you modified notebook files, you can update the corresponding rst
files using the ``update_rst.py`` script in the ``docs`` directory:

.. code:: bash

   docs> python update_rst.py 

Since Jupyter facilitates the editing and rendering of the
documentation, it is recommended to do all edits of Jupyter documents in
Jupyter, and only at the end (for example, before a commit) to convert
the content to rst files.
