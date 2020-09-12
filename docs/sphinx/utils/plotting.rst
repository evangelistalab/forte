.. _`sec:methods:fci`:

Plotting MOs
============

.. sectionauthor:: Francesco A. Evangelista

Forte includes a set of utilities for plotting molecular orbitals saved in the cube file format directly from a Jupyter notebook.
A good place to start is this: 
`Tutorial_02.00_plotting_mos.ipynb <https://github.com/evangelistalab/forte/tree/master/tutorials/Tutorial_02.00_plotting_mos.ipynb>`_.

Generating cube files 
---------------------

The ``forte.utils.psi4_cubeprop`` function offers a convenient way to generate cube files
from the information stored in a psi4 ``Wavefunction`` object::

    import forte.utils
    forte.utils.psi4_cubeprop(wfn,path='cubes',nocc=4,nvir=4)

By default this function plots the HOMO-2 to the LUMO+2 orbitals, but in this example we specifically indicate that we want
4 occupied and 4 virtual orbitals.
This function can also take a list of the orbitals to plot via the ``orbs`` option and it can return a set of CubeFile objects::

    cubes = forte.utils.psi4_cubeprop(wfn, path = '.', orbs = [3,4,5,6], load = True)
    
Reading, manipulating, and saving cube files
--------------------------------------------

Cube files can be read from disk via the ``CubeFile`` class. To read a cube file instantiate a ``CubeFile`` object
by passing the file name::

    cube = forte.CubeFile('cubes/Psi_a_15_15-A.cube')
    
From this object, we can plot the cube file or extract useful information::

    # number of atoms
    print(f'cube.natoms() -> {cube.natoms()}')

    # number of grid points along each direction
    print(f'cube.num() -> {cube.num()}')
    
The ``CubeFile`` class supports three type of operations:

``scale(double factor)``: scale all the values on the grid by ``factor``
    :math:`\phi(\mathbf{r}_i) \leftarrow \mathtt{factor} * \phi(\mathbf{r}_i)`  
``add(CubeFile cube)``: add to this cube file the grid values stored in cube
    :math:`\phi(\mathbf{r}_i) \leftarrow \phi(\mathbf{r}_i) + \psi(\mathbf{r}_i)`  
``pointwise_product(CubeFile cube)``: multiply each point of this cube file with the values stored in cube
    :math:`\phi(\mathbf{r}_i) \leftarrow \phi(\mathbf{r}_i) * \psi(\mathbf{r}_i)`

For example, we can compute the density of an orbital by taking the pointwise product with itself::

    cube = forte.CubeFile('cubes/Psi_a_15_15-A.cube')
    dens = forte.CubeFile(cube)
    dens.pointwise_product(dens)
    
To write a ``CubeFile`` object to disk for later use just call the ``save`` function::

    dens.save('cubes/dens.cube')

Plotting cube files
-------------------

Forte includes a low-level 3D renderer based on ``pythreejs`` and a simple interface to this renderer, the ``CubeViewer`` class.
We can tell the CubeViewer class to look for cube files in a specific path (via the path option)::
    
    cv = forte.utils.CubeViewer(path='cubes')

Alternatively, we can pass a list of cube files to load (via the cubes options). Here we specify two files and we also change the color scheme::
    
    cv2 = forte.utils.CubeViewer(cubes=['cubes/Psi_a_13_13-A.cube','cubes/Psi_a_16_16-A.cube'],colorscheme='electron')

Generating cube files 
---------------------

The ``forte.utils.psi4_cubeprop`` function offers a convenient way to generate cube files
from the information stored in a psi4 ``Wavefunction`` object::