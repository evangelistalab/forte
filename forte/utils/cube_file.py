import numpy as np
import re
import functools


class CubeFile():
    """
    A class to read, write, and manipulate cube files

    This class assumes that all coordinates (atoms, grid points)
    are stored in atomic units

    Uses code from the parse_cube function written by Andy Simmonett (psi4 project)

    Attributes
    ----------
    data : numpy array
        The values on a grid stored as a 3d array

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
        Performs self.data *= factor
    add(other)
        To each grid point add the value of grid poins from another CubeFile
        Performs: self.data += other.data
    pointwise_product(other):
        Multiply every grid point with the value of grid points from another CubeFile
        Performs: self.data *= other.data
    """
    def __init__(self, filename=None):
        self.filename = filename
        self.title = None
        self.comment = None
        self.levels = []
        self.__num = [None, None, None]
        self.__min = [None, None, None]
        self.__max = [None, None, None]
        self.__inc = [None, None, None]
        self.__natoms = None
        self.__atom_numbers = None
        self.__atom_coords = None
        self.__data = None
        if self.filename:
            self.load(self.filename)

    def natoms(self):
        return self.__natoms

    def atom_numbers(self):
        return self.__atom_numbers

    def atom_coords(self):
        return self.__atom_coords

    def num(self):
        return self.__num

    def min(self):
        return self.__min

    def max(self):
        return self.__max

    def inc(self):
        return self.__inc

    def data(self):
        return self.__data

    def load(self, filename):
        with open(filename) as fp:
            self.title = fp.readline().rstrip()
            self.comment = fp.readline().rstrip()
            m = re.search(
                r"\(([-+]?[0-9]*\.?[0-9]+)\,([-+]?[0-9]*\.?[0-9]+)\)",
                self.comment)
            if (m):
                self.levels = [float(s) for s in m.groups()]

            origin = fp.readline().split()
            self.__natoms = int(origin[0])
            self.__min = tuple(float(entry) for entry in origin[1:])

            infox = fp.readline().split()
            numx = int(infox[0])
            incx = float(infox[1])

            infoy = fp.readline().split()
            numy = int(infoy[0])
            incy = float(infoy[2])

            infoz = fp.readline().split()
            numz = int(infoz[0])
            incz = float(infoz[3])

            self.__num = (numx, numy, numz)
            self.__inc = (incx, incy, incz)
            self.__max = tuple(self.__min[i] + self.__inc[i] * self.__num[i]
                               for i in range(3))

            atnums = []
            coords = []
            for atom in range(self.__natoms):
                coordinfo = fp.readline().split()
                atnums.append(int(coordinfo[0]))
                coords.append(list(map(float, coordinfo[2:])))

            self.__atom_numbers = np.array(atnums)
            self.__atom_coords = np.array(coords)

            data = np.array(
                [float(entry) for line in fp for entry in line.split()])

            if len(data) != numx * numy * numz:
                raise Exception(
                    "Number of data points is inconsistent with header in Cube file!"
                )
            self.__data = data.reshape((numx, numy, numz))

    def save(self, filename):
        with open(filename, 'w+') as fp:
            fp.write('{}\n{}\n'.format(self.title, self.comment))
            fp.write('{0:6d} {1[0]:10.6f} {1[1]:10.6f} {1[2]:10.6f}\n'.format(
                self.__natoms, self.__min))
            fp.write('{:6d} {:10.6f} {:10.6f} {:10.6f}\n'.format(
                self.__num[0], self.__inc[0], 0.0, 0.0))
            fp.write('{:6d} {:10.6f} {:10.6f} {:10.6f}\n'.format(
                self.__num[1], 0.0, self.__inc[1], 0.0))
            fp.write('{:6d} {:10.6f} {:10.6f} {:10.6f}\n'.format(
                self.__num[2], 0.0, 0.0, self.__inc[2]))
            for atom in range(self.__natoms):
                Z = self.__atom_numbers[atom]
                xyz = self.__atom_coords[atom]
                fp.write(
                    '{0:3d} {1[0]:10.6f} {1[1]:10.6f} {1[2]:10.6f}\n'.format(
                        Z, xyz))

            # flatten the data and write to disk
            flatdata = np.ndarray.flatten(self.data)
            nfullrows = len(flatdata) // 6
            fstr = '{0[0]:12.5E} {0[1]:12.5E} {0[2]:12.5E} {0[3]:12.5E} {0[4]:12.5E} {0[5]:12.5E}\n'
            for k in range(nfullrows):
                fp.write(fstr.format(flatdata[6 * k:6 * k + 6]))
            nleftover = len(flatdata) - 6 * nfullrows
            for n in range(nleftover):
                fp.write('{:12.5E} '.format(flatdata[6 * nfullrows + n]))

    def save_npz(self, filename):
        np.savez_compressed(file=filename,
                            title=self.title,
                            comment=self.comment,
                            num=self.__num,
                            __min=self.__min,
                            max=self.__max,
                            inc=self.__inc,
                            natoms=self.__natoms,
                            __atom_numbers=self.__atom_numbers,
                            __atom_coords=self.__atom_coords,
                            levels=self.levels,
                            data=self.data)

    def load_npz(self, filename):
        file = np.load(filename)
        self.title = file['title']
        self.comment = file['comment']
        self.__num = file['num']
        self.__min = file['min']
        self.__max = file['max']
        self.__inc = file['inc']
        self.__natoms = file['natoms']
        self.__atom_numbers = file['__atom_numbers']
        self.__atom_coords = file['__atom_coords']
        self.levels = file['levels']
        self.data = file['data']

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

    def compute_levels(self, mo_type, fraction):
        sorted_data = sorted(self.__data.flatten(), key=abs, reverse=True)
        power = 2
        if mo_type == "density":
            power = 1

        neg_level = 0.0
        pos_level = 0.0
        sum = functools.reduce(lambda i, j: i + j**power,
                               [sorted_data[0]**power] + sorted_data[1:])
        partial_sum = 0.0
        for n in range(len(sorted_data)):
            partial_sum += sorted_data[n]**power
            if partial_sum / sum < fraction:
                if sorted_data[n] < 0.0:
                    neg_level = sorted_data[n]
                else:
                    pos_level = sorted_data[n]
            else:
                break
        return (pos_level, neg_level)

    def __str__(self):
        s = 'title: {}\ncomment: {}'.format(self.title, self.comment)
        s += '\ntotal grid points = {}'.format(self.__num[0] * self.__num[1] *
                                               self.__num[2])
        s += '\ngrid points = [{0[0]},{0[1]},{0[2]}]'.format(self.__num)
        s += '\nmin = [{0[0]:9.3f},{0[1]:9.3f},{0[2]:9.3f}]'.format(self.__min)
        s += '\nmax = [{0[0]:9.3f},{0[1]:9.3f},{0[2]:9.3f}]'.format(self.__max)
        s += '\ninc = [{0[0]:9.3f},{0[1]:9.3f},{0[2]:9.3f}]'.format(self.__inc)
        s += '\ndata = {}'.format(self.__data)
        return s
