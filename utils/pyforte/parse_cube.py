import numpy as np
import re

class CubeFile():
    """
    A class to read, write, and manipulate cube files

    This class assumes that all coordinates (atoms, grid points)
    are stored in atomic units

    Attributes
    ----------
    data : type
        description

    Methods
    -------
    load(filename)
        Load a cube file (standard format)
    save(filename)
        Save a cube file (standard format)
    save_npz(filename)
        Save a cube file using numpy's .npz format
    load_npz(filename)
        Load a cube file using numpy's .npz format
    scale(factor)
        Multiply the data by a given factor
        Performs: self.data *= factor
    add(other)
        To each grid point add the value of grid poins from another CubeFile
        Performs: self.data += other.data
    pointwise_product(other):
        Multiply every grid point with the value of grid points from another CubeFile.
        Performs: self.data *= other.data
    """
    def __init__(self, filename = None):
        self.filename = filename
        self.comment1 = None
        self.comment2 = None
        self.levels = []
        self.num = [None, None, None]
        self.min = [None, None, None]
        self.max = [None, None, None]
        self.inc = [None, None, None]
        self.natoms = None
        self.atom_numbers = None
        self.atom_coords = None
        self.data = None
        if self.filename:
            self.load(self.filename)

    def load(self, filename):
        with open(filename) as fp:
            self.comment1 = fp.readline().rstrip()
            self.comment2 = fp.readline().rstrip()
            m = re.search(r"\(([-+]?[0-9]*\.?[0-9]+)\,([-+]?[0-9]*\.?[0-9]+)\)",
            self.comment2)
            if (m):
                self.levels = [float(s) for s in m.groups()]

            origin = fp.readline().split()
            self.natoms = int(origin[0])
            self.min = tuple(float(entry) for entry in origin[1:])

            infox = fp.readline().split()
            numx = int(infox[0])
            incx = float(infox[1])

            infoy = fp.readline().split()
            numy = int(infoy[0])
            incy = float(infoy[2])

            infoz = fp.readline().split()
            numz = int(infoz[0])
            incz = float(infoz[3])

            self.num = (numx, numy, numz)
            self.inc = (incx, incy, incz)
            self.max = tuple(self.min[i] + self.inc[i] * self.num[i] for i in range(3))

            atnums = []
            coords = []
            for atom in range(self.natoms):
                coordinfo = fp.readline().split()
                atnums.append(int(coordinfo[0]))
                coords.append(list(map(float, coordinfo[2:])))
                self.atom_numbers = np.array(atnums)
                self.atom_coords = np.array(coords)

            data = np.array(
            [float(entry) for line in fp for entry in line.split()])

            if len(data) != numx * numy * numz:
                raise Exception(
            "Number of data points is inconsistent with header in Cube file!"
            )
            self.data = data.reshape((numx, numy, numz))

    def save(self, filename):
        with open(filename,'w+') as fp:
            fp.write('{}\n{}\n'.format(self.comment1,self.comment2))
            fp.write('{0:6d} {1[0]:10.6f} {1[1]:10.6f} {1[2]:10.6f}\n'.format(self.natoms,
            self.min))
            fp.write('{:6d} {:10.6f} {:10.6f} {:10.6f}\n'.format(self.num[0], self.inc[0], 0.0, 0.0))
            fp.write('{:6d} {:10.6f} {:10.6f} {:10.6f}\n'.format(self.num[1], 0.0, self.inc[1], 0.0))
            fp.write('{:6d} {:10.6f} {:10.6f} {:10.6f}\n'.format(self.num[2], 0.0, 0.0, self.inc[2]))
            for atom in range(self.natoms):
                Z = self.atom_numbers[atom]
                xyz = self.atom_coords[atom]
                fp.write('{0:3d} {1[0]:10.6f} {1[1]:10.6f} {1[2]:10.6f}\n'.format(Z,xyz))

            # flatten the data and write to disk
            flatdata = np.ndarray.flatten(self.data)
            nfullrows = len(flatdata) // 6
            fstr = '{0[0]:12.5E} {0[1]:12.5E} {0[2]:12.5E} {0[3]:12.5E} {0[4]:12.5E} {0[5]:12.5E}\n'
            for k in range(nfullrows):
                fp.write(fstr.format(flatdata[6 * k: 6 * k + 6]))
            nleftover = len(flatdata) - 6 * nfullrows
            for n in range(nleftover):
                fp.write('{:12.5E} '.format(flatdata[6 * nfullrows + n]))

    def save_npz(self, filename):
        np.savez_compressed(file=filename,
            comment1=self.comment1,
            comment2=self.comment2,
            num=self.num,
            min=self.min,
            max=self.max,
            inc=self.inc,
            natoms=self.natoms,
            atom_numbers=self.atom_numbers,
            atom_coords=self.atom_coords,
            levels=self.levels,
            data=self.data
            )

    def load_npz(self, filename):
        file = np.load(filename)
        self.comment1=file['comment1']
        self.comment2=file['comment2']
        self.num=file['num']
        self.min=file['min']
        self.max=file['max']
        self.inc=file['inc']
        self.natoms=file['natoms']
        self.atom_numbers=file['atom_numbers']
        self.atom_coords=file['atom_coords']
        self.levels=file['levels']
        self.data=file['data']

    def scale(self, factor):
        # multiplication by a scalar
        if isinstance(factor, float):
            self.data *= factor

    def add(self, other):
        # add another cube file
        self.data += other.data

    def pointwise_product(self, other):
        # multiplication by a scalar
        self.data *= other.data
        self.levels = []

    def __str__(self):
        s = 'title: {}\ncomment: {}'.format(self.comment1, self.comment2)
        s += '\ntotal grid points = {}'.format(self.num[0] * self.num[1] * self.num[2])
        s += '\ngrid points = [{0[0]},{0[1]},{0[2]}]'.format(self.num)
        s += '\nmin = [{0[0]:9.3f},{0[1]:9.3f},{0[2]:9.3f}]'.format(self.min)
        s += '\nmax = [{0[0]:9.3f},{0[1]:9.3f},{0[2]:9.3f}]'.format(self.max)
        s += '\ninc = [{0[0]:9.3f},{0[1]:9.3f},{0[2]:9.3f}]'.format(self.inc)
        s += '\ndata = {}'.format(self.data)
        return s

def parse_cube(filename: str) -> dict:
    """
    Parse a cube file and return a dictionary with the information contained in it.

    The cubefile data is stored in a numpy array.

    Written by Andy Simmonett.

    Parameters
    ----------
    filename : str
        The name of the cube file
    """

    with open(filename) as fp:
        results = {}

        # skip over the title
        title = fp.readline()
        comment = fp.readline()
        m = re.search(r"\(([-+]?[0-9]*\.?[0-9]+)\,([-+]?[0-9]*\.?[0-9]+)\)",
                      comment)
        if (m):
            results['levels'] = [float(s) for s in m.groups()]

        origin = fp.readline().split()
        natoms = int(origin[0])
        results['minx'] = minx = float(origin[1])
        results['miny'] = miny = float(origin[2])
        results['minz'] = minz = float(origin[3])

        infox = fp.readline().split()
        numx = int(infox[0])
        incx = float(infox[1])
        results['incx'] = incx
        results['numx'] = numx
        results['maxx'] = minx + incx * numx

        infoy = fp.readline().split()
        numy = int(infoy[0])
        incy = float(infoy[2])
        results['incy'] = incy
        results['numy'] = numy
        results['maxy'] = miny + incy * numy

        infoz = fp.readline().split()
        numz = int(infoz[0])
        incz = float(infoz[3])
        results['incz'] = incz
        results['numz'] = numz
        results['maxz'] = minz + incz * numz

        atnums = []
        coords = []
        for atom in range(natoms):
            coordinfo = fp.readline().split()
            atnums.append(int(coordinfo[0]))
            coords.append(list(map(float, coordinfo[2:])))
        results['atom_numbers'] = np.array(atnums)
        results['atom_coords'] = np.array(coords)

        data = np.array(
            [float(entry) for line in fp for entry in line.split()])
        if len(data) != numx * numy * numz:
            raise Exception(
                "Amount of parsed data is inconsistent with header in Cube file!"
            )
        results['data'] = data.reshape((numx, numy, numz))

        return results
