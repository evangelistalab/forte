#include "cube_file.h"

namespace forte {

CubeFile::CubeFile(const std::string& filename) : filename_(filename) { load(filename_); }

const CubeFile::std::vector<double>& data() const { return data_; }

void CubeFile::load(std::string filename) {


//        with open(filename) as fp:
//            self.title = fp.readline().rstrip()
//            self.comment = fp.readline().rstrip()
//            m = re.search(r"\(([-+]?[0-9]*\.?[0-9]+)\,([-+]?[0-9]*\.?[0-9]+)\)",
//            self.comment)
//            if (m):
//                self.levels = [float(s) for s in m.groups()]

//            origin = fp.readline().split()
//            self.natoms = int(origin[0])
//            self.min = tuple(float(entry) for entry in origin[1:])

//            infox = fp.readline().split()
//            numx = int(infox[0])
//            incx = float(infox[1])

//            infoy = fp.readline().split()
//            numy = int(infoy[0])
//            incy = float(infoy[2])

//            infoz = fp.readline().split()
//            numz = int(infoz[0])
//            incz = float(infoz[3])

//            self.num = (numx, numy, numz)
//            self.inc = (incx, incy, incz)
//            self.max = tuple(self.min[i] + self.inc[i] * self.num[i] for i in range(3))

//            atnums = []
//            coords = []
//            for atom in range(self.natoms):
//                coordinfo = fp.readline().split()
//                atnums.append(int(coordinfo[0]))
//                coords.append(list(map(float, coordinfo[2:])))
//                self.atom_numbers = np.array(atnums)
//                self.atom_coords = np.array(coords)

//            data = np.array(
//            [float(entry) for line in fp for entry in line.split()])

//            if len(data) != numx * numy * numz:
//                raise Exception(
//            "Number of data points is inconsistent with header in Cube file!"
//            )
//            self.data = data.reshape((numx, numy, numz))

}

} // namespace forte
