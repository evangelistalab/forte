import math
import re
import collections
import numpy as np
import skimage.measure

from IPython.display import display
from pythreejs import *
from .atom_data import *


def rgb2hex(r, g, b):
    r = max(0, min(r, 255))
    g = max(0, min(g, 255))
    b = max(0, min(b, 255))
    return '#%02x%02x%02x' % (r, g, b)


def xyz_to_atoms_list(xyz):
    """
    Converts an xyz geometry to a list of the form

    Parameters
    ----------
    xyz : str
        An xyz geometry where each entry has the format "<atom symbol> <x> <y> <z>".
        Any comment will be ignored

    Returns
    -------
    A list(tuple(str, float, float, float)) containing the atomic symbol and coordinates of the atom.
    """
    atoms_list = []
    re_xyz = re.compile(
        r"(\w+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)"
    )
    for line in xyz.split('\n'):
        m = re.search(re_xyz, line)
        if (m):
            symbol, x, y, z = m.groups()
            atoms_list.append((symbol, float(x), float(y), float(z)))
    return atoms_list


class Py3JSRenderer():
    """
    A lightweight molecule and orbital renderer

    Attributes
    ----------
    bond_color : color
        color of the bonds
    bond_radius : float
        the radius of the bonds
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    display()
        Display the pythreejs renderer
    renderer()
        Return the pythreejs renderer
    add_molecule(atoms_list, bohr=False, shift_to_com=True)
        Add a molecule specified by a list of (symbol,x,y,z) tuples
    add_molecule_xyz(xyz, bohr=False, shift_to_com=True)
        Add a molecule specified in xyz format
    add_cubefile(cube,type='mo',levels=None,colors=None,colorscheme=None,opacity=1.0,scale=1.0,sumlevel=0.85,add_geom=True,shift_to_com=True)
        Add a cube file
    add_sphere(self, position, radius, color, opacity=1.0)
        Add a sphere (should not be used to draw molecules)
    add_cylinder(self, xyz1, xyz2, color, radius)
        Add a cylinder (should not be used to draw molecules)
    add_arrow(xyz1,xyz2,color,radius_small=0.1,radius_large=0.3,arrow_height=0.6)
        Add an arrow between two points
    add_plane(position,color,plane=None,normal=(0.0, 0.0, 1.0),type='circle',width=4,height=4,opacity=1.0)
        Add a plane
    def add_box(position,width,height,depth,color,opacity=1.0,normal=(0, 0, 1))
        Add a box
    """
    def __init__(self, width=400, height=400):
        """
        Class initialization function

        Parameters
        ----------
        width : int
            The width of the scene in pixels (default = 400)
        height : int
            The height of the scene in pixels (default = 400)
        """
        self.width = width
        self.height = height
        # aspect ratio
        self.aspect = float(self.width) / float(self.height)
        self.bond_radius = 0.175  # a.u.
        self.bond_color = '#555555'
        self.angtobohr = 1.88973  # TODO: use Psi4's value
        self.atom_size = 0.6  # scaling factor for atom geometry
        self.atom_geometries = {}
        self.atom_materials = {}
        self.bond_materials = {}
        self.bond_geometry = None

        # set an initial scene size
        self.camera_width = 10.0
        self.camera_height = self.camera_width / self.aspect
        self.__initialize_pythreejs_renderer()

    def display(self):
        """
        Display this renderer
        """
        display(self.renderer)

    def renderer(self):
        """
        Return the Renderer object
        """
        return self.renderer

    def add_molecule(self, atoms_list, bohr=False, shift_to_com=True):
        """
        Add a molecular geometry to the scene. The geometry is given as a list of atoms
        symbols and xyz coordinates

        Parameters
        ----------
        atoms_list : list(tuple(str, float, float, float))
            A list of tuples containing the atomic symbol and coordinates of the atom using the format
            (atomic symbol,x,y,z)
        bohr : bool
            Are the coordinate in units of bohr? (default = False)
        scale : float
            Scale factor to change the size of the scene (default = 1.0)
        shift_to_com : bool
            Shift the molecule so that the center of mass is at the origin (default = True)
        """
        if bohr == False:
            atoms_list2 = []
            for atom in atoms_list:
                symbol, x, y, z = atom
                new_atom = (symbol, self.angtobohr * x, self.angtobohr * y,
                            self.angtobohr * z)
                atoms_list2.append(new_atom)
            atoms_list = atoms_list2

        self.molecule = Group()

        # Add the atoms
        # Performance optimization using CloneArray
        # First find all the atoms of the same type
        if shift_to_com:
            Xcm, Ycm, Zcm = self.__center_of_mass(atoms_list)
        else:
            Xcm, Ycm, Zcm = (0.0, 0.0, 0.0)

        atom_positions = collections.defaultdict(list)
        for symbol, x, y, z in atoms_list:
            atom_positions[symbol].append([x - Xcm, y - Ycm, z - Zcm])
        # Then add the unique atoms at all the positions
        for atom_type in atom_positions:
            atom_mesh = self.__get_atom_mesh((atom_type, 0.0, 0.0, 0.0))
            clone_geom = CloneArray(original=atom_mesh,
                                    positions=atom_positions[atom_type])
            self.scene.add(clone_geom)

        # Add the bonds
        for i in range(len(atoms_list)):
            atom1 = atoms_list[i]
            for j in range(i + 1, len(atoms_list)):
                atom2 = atoms_list[j]
                bond = self.__get_bond_mesh(atom1, atom2)
                if bond:
                    self.scene.add(bond)
        return self.renderer

    def add_molecule_xyz(self, xyz, bohr=False, shift_to_com=True):
        """
        Add a molecular geometry in xyz format to the scene

        Parameters
        ----------
        xyz : str
            An xyz geometry where each entry has the format "<atom symbol> <x> <y> <z>".
            Any comment will be ignored
        bohr : bool
            Are the coordinate in units of bohr? (default = False)
        scale : float
            Scale factor to change the size of the scene (default = 1.0)
        shift_to_com : bool
            Shift the molecule so that the center of mass is at the origin (default = True)
        """
        atoms_list = xyz_to_atoms_list(xyz)
        self.add_molecule(atoms_list, bohr, scale)

    def add_cubefile(self,
                     cube,
                     type='mo',
                     levels=None,
                     colors=None,
                     colorscheme=None,
                     opacity=1.0,
                     scale=1.0,
                     sumlevel=0.85,
                     add_geom=True,
                     shift_to_com=True):
        """
        Add a cube file (and optionally the molecular geometry) to the scene. This function will automatically select the levels and colors
        with which to plot the surfaces

        Parameters
        ----------
        cube : CubeFile
            A CubeFile object
        type : str
            The type of cube file ('mo' or 'density')
        levels : list(float)
            The levels to plot (default = None). If not provided, levels will be automatically selected
            using the compute_levels() function of the CubeFile class. The variable sumlevel is used to
            select the levels
        color : list(str)
            The color of each surface passed as a list of hexadecimal color codes (default = None)
        colorscheme : str
            A predefined color scheme (default = 'emory'). Possible options are ['emory', 'national', 'bright', 'electron', 'wow']
        opacity : float
            Opacity of the surfaces (default = 1.0)
        scale : float
            Scale factor to change the size of the scene (default = 1.0)
        sumlevel : float
            Cumulative electron density threshold used to find the isosurface levels
        add_geom : bool
            Show the molecular geometry (default = True)
        shift_to_com : bool
            Shift the molecule so that the center of mass is at the origin (default = True)
        """
        Xcm, Ycm, Zcm = (0.0, 0.0, 0.0)
        if shift_to_com or add_geom:
            atoms_list = []
            for Z, xyz in zip(cube.atom_numbers(), cube.atom_coords()):
                symbol = ATOM_DATA[Z]['symbol']
                atoms_list.append((symbol, xyz[0], xyz[1], xyz[2]))
            # compute the center of mass
            Xcm, Ycm, Zcm = self.__center_of_mass(atoms_list)
            if add_geom:
                self.add_molecule(atoms_list,
                                  bohr=True,
                                  shift_to_com=shift_to_com)

        # compute the isosurface levels
        if not levels:
            levels = cube.compute_levels(type, sumlevel)

        # select the color scheme
        if colorscheme == 'national':
            colors = ['#e60000', '#0033a0']
        elif colorscheme == 'bright':
            colors = ['#ffcc00', '#00bfff']
        elif colorscheme == 'electron':
            colors = ['#ff00bf', '#2eb82e']
        elif colorscheme == 'wow':
            colors = ['#AC07F2', '#D7F205']
        elif colors == None or colorscheme == 'emory':
            colors = ['#f2a900', '#0033a0']

        # grab the data and extents, shift to the center of mass automatically
        data = cube.data()
        extent = [[cube.min()[0] - Xcm,
                   cube.max()[0] - Xcm],
                  [cube.min()[1] - Ycm,
                   cube.max()[1] - Ycm],
                  [cube.min()[2] - Zcm,
                   cube.max()[2] - Zcm]]
        for level, color in zip(levels, colors):
            if abs(level) > 1.0e-4:
                mesh = self.__isosurface_mesh(data,
                                              level=level,
                                              color=color,
                                              extent=extent,
                                              opacity=opacity)
                self.scene.add(mesh)

    def add_sphere(self, position, radius, color, opacity=1.0):
        """
        This function adds a sphere

        This should not be used to draw molecules because it cannot efficiently
        handle mutiple copied of the same object

        Parameters
        ----------
        position : tuple(float, float, float)
            The (x, y, z) coordinates of the center of the sphere
        radius : float
            The sphere radius
        color : str
            Hexadecimal color code
        opacity : float
            The opacity of the sphere (default = 1.0)
        """
        geometry = SphereGeometry(radius=radius,
                                  widthSegments=24,
                                  heightSegments=24)
        material = MeshStandardMaterial(color=color,
                                        roughness=0.0,
                                        metalness=0.0,
                                        side='DoubleSide',
                                        transparent=True,
                                        opacity=opacity)

        mesh = Mesh(geometry=geometry, material=material, position=position)
        self.scene.add(mesh)

    def add_cylinder(self, xyz1, xyz2, color, radius):
        """
        This function adds a cylinder/cone between two points

        This should not be used to draw molecules.

        Parameters
        ----------
        xyz1 : tuple(float, float, float)
            The (x1, y1, z1) coordinates of the beginning of the cylinder
        xyz2 : tuple(float, float, float)
            The (x2, y2, z2) coordinates of the end of the cylinder
        color : str
            Hexadecimal color code
        radius : float, (float,float), or [float,float]
            The radius of the cylinder. If a float is passed then the cylinder is
            assumed to have constant radius. If a list/tuple is passed the two numbers
            correspond to the radius at points 1 and 2. This is useful to draw cones
        """
        if isinstance(radius, float):
            mesh = self.__get_cylinder_mesh(xyz1, xyz2, radius, radius, color)
            self.scene.add(mesh)
        elif isinstance(radius, (list, tuple)):
            if len(radius) == 2:
                mesh = self.__get_cylinder_mesh(xyz1, xyz2, radius[0], radius[1], color)
                self.scene.add(mesh)
            else:
                print(
                    f'add_cylinder(): radius (= {radius}) must be either a float or a list/tuple with two elements'
                )

    def add_arrow(self,
                  xyz1,
                  xyz2,
                  color,
                  radius_small=0.1,
                  radius_large=0.3,
                  arrow_height=0.6):
        """
        This function adds an arrow  between two points

        Parameters
        ----------
        xyz1 : tuple(float, float, float)
            The (x1, y1, z1) coordinates of the beginning of the arrow
        xyz2 : tuple(float, float, float)
            The (x2, y2, z2) coordinates of the end of the arrow
        color : str
            Hexadecimal color code
        radius_small : float
            The radius of the arrow tail
        radius_large : float
            The radius of the base of the arrow cone
        arrow_height : float
            The height of the arrow cone
        """
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        d = sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        fraction = (d - arrow_height) / d
        xyz_base = [
            x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction,
            z1 + (z2 - z1) * fraction
        ]
        mesh = self.__get_cylinder_mesh(xyz1, xyz_base, radius_small, radius_small, color)
        self.scene.add([mesh])
        mesh = self.__get_cylinder_mesh(xyz_base, xyz2, radius_large, 0.0, color)
        self.scene.add([mesh])

    def add_plane(self,
                  position,
                  color,
                  plane=None,
                  normal=(0.0, 0.0, 1.0),
                  type='circle',
                  width=4,
                  height=4,
                  opacity=1.0):
        """
        This function adds a plane centered at a given position. The type of plane
        can be specified either via a vector perpendicular to the plane or by selecting
        one of the planes that lies on two Cartesian axes

        Parameters
        ----------
        position : tuple(float, float, float)
            The (x, y, z) coordinates of the center of the plane
        plane : str
            The type of plane ('xy', 'xz', 'yz') (default = None). This overrides the `normal` argument
        normal : tuple(float, float, float)
            A vector (x, y, z) to which the plane is orthogonal (default = (0,0,1))
        color : str
            Hexadecimal color code
        type : str
            The type of plane ('circle', 'square') (default = 'circle')
        width : float
            The width (radius) of the plane (default = 4.0)
        height : float
            The height of the plane (default = 4.0)
        opacity : float
            The opacity of the plane (default = 1.0)
        """
        if type == 'square':
            geometry = PlaneGeometry(width=width,
                                     height=height,
                                     widthSegments=10,
                                     heightSegments=10)
        else:
            geometry = CircleGeometry(radius=width / 2, segments=48)

        material = MeshStandardMaterial(color=color,
                                        roughness=0.3,
                                        metalness=0.0,
                                        side='DoubleSide',
                                        transparent=True,
                                        opacity=opacity)

        mesh = Mesh(geometry=geometry, material=material, position=position)

        if plane == 'xy' or plane == 'yx':
            normal = (0.0, 0.0, 1.0)
        elif plane == 'xz' or plane == 'zx':
            normal = (0.0, 1.0, 0.0)
        elif plane == 'yz' or plane == 'zy':
            normal = (1.0, 0.0, 0.0)

        # If the plane is not rotated skip the rotation step
        if normal[2] != 1.0 or normal[2] != -1.0:
            R = self.__plane_rotation_matrix(normal)
            mesh.setRotationFromMatrix(R)
        self.scene.add(mesh)

    def add_box(self,
                position,
                width,
                height,
                depth,
                color,
                opacity=1.0,
                normal=(0, 0, 1)):
        """
        This function adds a box centered at a given position. The orientation of the
        box is specified via a vector perpendicular to the plane spanned by the width and height.

        Parameters
        ----------
        position : tuple(float, float, float)
            The (x, y, z) coordinates of the center of the plane
        width : float
            The width (x dimension) of the box
        height : float
            The height (y dimension) of the box
        depth : float
            The depth (z dimension) of the box
        color : str
            Hexadecimal color code
        opacity : float
            The opacity of the box (default = 1.0)
        normal : tuple(float, float, float)
            A vector (x, y, z) to which the plane is orthogonal (default = (0,0,1))
        """
        geometry = BoxGeometry(width=width,
                               height=height,
                               depth=depth,
                               widthSegments=10,
                               heightSegments=10,
                               depthSegments=10)
        material = MeshStandardMaterial(color=color,
                                        roughness=0.3,
                                        metalness=0.0,
                                        side='DoubleSide',
                                        transparent=True,
                                        opacity=opacity)

        mesh = Mesh(geometry=geometry, material=material, position=position)

        # If the bond rotation is 180 deg then return
        R = self.__plane_rotation_matrix(normal)
        mesh.setRotationFromMatrix(R)
        self.scene.add(mesh)

    def __initialize_pythreejs_renderer(self):
        """
        Create a pythreejs Scene and a Camera and add them to a Renderer
        """
        # create a Scene
        self.scene = Scene()
        # create a camera
        self.camera = OrthographicCamera(
            left=-self.camera_width / 2,
            right=self.camera_width / 2,
            top=self.camera_height / 2,
            bottom=-self.camera_height / 2,
            position=[0, 0, self.camera_height * 2.0],
            up=[0, 1, 0],
            children=[
                DirectionalLight(color='white',
                                 position=[5, 5, 1],
                                 intensity=0.5)
            ],
            near=.1,
            far=1000)

        # add the camera and some ambiend light to the scene
        self.scene.add([self.camera, AmbientLight(color='#999999')])

        self.renderer = Renderer(
            camera=self.camera,
            scene=self.scene,
            controls=[OrbitControls(controlling=self.camera)],
            width=self.width,
            height=self.height)

    def __get_atom_mesh(self, atom_info):
        """
        This function returns a Mesh object (Geometry + Material) that represents an atom

        Parameters
        ----------
        atom_info : tuple(str, float, float, float)
            A tuple containing the atomic symbol and coordinates of the atom using the format
            (atomic symbol , x, y, z)
        """
        symbol, x, y, z = atom_info
        geometry = self.__get_atom_geometry(symbol)
        material = self.__get_atom_material(symbol)
        mesh = Mesh(geometry=geometry, material=material, position=[x, y, z])
        return mesh

    def __get_atom_geometry(self, symbol, shininess=75):
        """
        This function returns a sphere geometry object with radius proportional to the covalent atomic radius

        Parameters
        ----------
        symbol : str
            The symbol of the atom (e.g. 'Li')
        shininess : int
            The shininess of the sphere (default = 75)
        """
        if symbol in self.atom_geometries:
            return self.atom_geometries[symbol]
        atom_data = ATOM_DATA[ATOM_SYMBOL_TO_Z[symbol]]
        radius_covalent = atom_data['radius_covalent'] * self.angtobohr
        geometry = SphereGeometry(radius=self.atom_size * radius_covalent,
                                  widthSegments=24,
                                  heightSegments=24)
        self.atom_geometries[symbol] = geometry
        return geometry

    def __get_atom_material(self, symbol, shininess=75):
        """
        This function returns a Material object used to draw atoms

        Parameters
        ----------
        symbol : str
            The symbol of the atom (e.g. 'Li')
        shininess : int
            The shininess of the material (default = 75)
        """
        if symbol in self.atom_materials:
            return self.atom_materials[symbol]
        atom_data = ATOM_DATA[ATOM_SYMBOL_TO_Z[symbol]]
        color = 'rgb({0[0]},{0[1]},{0[2]})'.format(atom_data['color'])
        #        material = MeshPhongMaterial(color=color, shininess=shininess)
        material = MeshStandardMaterial(color=color,
                                        roughness=0.25,
                                        metalness=0.1)
        self.atom_materials[symbol] = material
        return material

    def __get_bond_mesh(self, atom1_info, atom2_info, radius=None):
        """
        This function adds a bond between two atoms
        atoms 1 and 2

        Parameters
        ----------
        xyz1 : tuple(float, float, float)
            The (x1, y1, z1) coordinates of the beginning of the arrow
        xyz2 : tuple(float, float, float)
            The (x2, y2, z2) coordinates of the end of the arrow
        color : str
            Hexadecimal color code
        radius_small : float
            The radius of the arrow
        radius_large : float
            The radius of the arrow
        """
        symbol1, x1, y1, z1 = atom1_info
        symbol2, x2, y2, z2 = atom2_info
        d = sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        radius_covalent1 = ATOM_DATA[
            ATOM_SYMBOL_TO_Z[symbol1]]['radius_covalent']
        radius_covalent2 = ATOM_DATA[
            ATOM_SYMBOL_TO_Z[symbol2]]['radius_covalent']

        bond_cutoff = self.bond_cutoff(radius_covalent1, radius_covalent2)
        if d > bond_cutoff:
            return None
        if radius == None:
            radius = self.bond_radius

        d = sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        position = [(x2 + x1) / 2, (y2 + y1) / 2, (z2 + z1) / 2]
        geometry = self.__get_bond_geometry()
        material = self.__get_bond_material(color=self.bond_color)
        mesh = Mesh(geometry=geometry, material=material, position=position)
        mesh.scale = (1, d, 1)
        # If the bond rotation is 180 deg then return
        if y1 - y2 == d:
            mesh.rotateX(3.14159265359)  #math.pi)
            return mesh

        R = self.__cylinder_rotation_matrix([x1, y1, z1], [x2, y2, z2])
        mesh.setRotationFromMatrix(R)
        return mesh

    def __get_bond_geometry(self):
        """
        This function returns a cylinder geometry object of unit height used to draw bonds

        """
        if self.bond_geometry:
            return self.bond_geometry
        self.bond_geometry = CylinderGeometry(radiusTop=self.bond_radius,
                                              radiusBottom=self.bond_radius,
                                              height=1,
                                              radialSegments=12,
                                              heightSegments=6,
                                              openEnded=False)
        return self.bond_geometry

    def __get_bond_material(self, color, shininess=75):
        """
        This function returns a Material object used to draw bonds

        Parameters
        ----------
        color : str
            Hexadecimal color code
        shininess : int
            The shininess of the material (default = 75)
        """
        if color in self.bond_materials:
            return self.bond_materials[color]
        material = MeshStandardMaterial(color=color,
                                        roughness=0.25,
                                        metalness=0.1)
        self.bond_materials[color] = material
        return material

    def __get_cylinder_mesh(self, xyz1, xyz2, radius1, radius2, color):
        """
        This function returns a Mesh object (Geometry + Material) that represents a bond between
        atoms 1 and 2

        Parameters
        ----------
        xyz1 : tuple(float, float, float)
            The (x1, y1, z1) coordinates of atom 1
        xyz2 : tuple(float, float, float)
            The (x2, y2, z2) coordinates of atom 2
        radius1 : float
            The radius of the bond at atom 1
        radius2 : float
            The radius of the bond at atom 2
        color : str
            Hexadecimal color code
        """

        radius1 = max(0.01, radius1)
        radius2 = max(0.01, radius2)
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        d = sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        position = [(x2 + x1) / 2, (y2 + y1) / 2, (z2 + z1) / 2]
        geometry = CylinderGeometry(radiusTop=radius2,
                                    radiusBottom=radius1,
                                    height=d,
                                    radialSegments=12,
                                    heightSegments=6,
                                    openEnded=False)
        material = MeshPhongMaterial(color=color, shininess=100)
        mesh = Mesh(geometry=geometry, material=material, position=position)

        # If the bond rotation is 180 deg then return
        if y1 - y2 == d:
            mesh.rotateX(3.14159265359)
            return mesh

        R = self.__cylinder_rotation_matrix(xyz1, xyz2)
        mesh.setRotationFromMatrix(R)
        return mesh

    def __plane_rotation_matrix(self, normal):
        """
        Computes the rotation matrix that converts a plane (circle/square) geometry in
        its standard orientation to one in which the plane is orthogonal to a given
        vector (normal). By default, planes in pythreejs are orthogonal to the vector (0,0,1),
        that is, they lay on the xy plane

        Parameters
        ----------
        normal : tuple(float, float, float)
            The vector to which we want to make a plane orthogonal
        """
        # normalize the vector
        x, y, z = normal
        d = sqrt(x**2 + y**2 + z**2)
        x /= d
        y /= d
        z /= d

        # compute the cross product: normal x (0,0,1)
        c0 = y
        c1 = -x
        c2 = 0.0

        # compute the dot product: normal . (0,0,1)
        dot = z
        c = dot
        s = sqrt(1 - c**2)

        # rotation matrix, see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        R = [
            c + (1 - c) * c0**2, c0 * c1 * (1 - c), c1 * s, c0 * c1 * (1 - c),
            c + (1 - c) * c1**2, -c0 * s, -c1 * s, c0 * s, c
        ]
        return R

    def bond_cutoff(self, r1, r2):
        """
        Compute the cutoff value for displaying a bond between two atoms

        Parameters
        ----------
        r1 : float
            The radius of atom 1
        r2 : float
            The radius of atom 2
        """
        return 1.5 * self.angtobohr * (r1 + r2)

    def __cylinder_rotation_matrix(self, xyz1, xyz2):
        """
        Computes the rotation matrix that converts a cylinder geometry in its standard
        orientation to a cylinder that starts at point xyz1 and ends at xyz2

        Parameters
        ----------
        xyz1 : tuple(float, float, float)
            The (x1, y1, z1) coordinates of the beginning of the cylinder
        xyz2 : tuple(float, float, float)
            The (x2, y2, z2) coordinates of the end of the cylinder
        """
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        d = sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        b1 = (x2 - x1) / d
        b2 = (y2 - y1) / d
        b3 = (z2 - z1) / d
        gamma = 1 / (1 + b2)
        R = [
            1 - b1 * b1 * gamma, -b1, -b1 * b3 * gamma, b1,
            1 - (b1 * b1 + b3 * b3) * gamma, b3, -b1 * b3 * gamma, -b3,
            1 - b3 * b3 * gamma
        ]
        return R

    def __isosurface_mesh(self, data, level, color, extent=None, opacity=1.0):
        """
        This function returns a Mesh object (Geometry + Material) for an isosurface

        Parameters
        ----------
        data : numpy.ndarray
            A 3D array containing the values on a grid
        level : float
            The isosurface level. This must be included in the range of values on the grid
        color : str
            Hexadecimal code for the color used to display the surface
        extent : list
            list of [[xmin, xmax], [ymin, ymax], [zmin, zmax]] values that define the bounding box of the mesh,
            otherwise the viewport is used
        opacity : float
            The opacity of the surface (default = 1.0)
        """
        vertices, faces = self.__compute_isosurface(data,
                                                    level=level,
                                                    color=color,
                                                    extent=extent)

        # Create the geometry
        isoSurfaceGeometry = Geometry(vertices=vertices, faces=faces)

        # Calculate normals per vertex for round edges
        isoSurfaceGeometry.exec_three_obj_method('computeVertexNormals')

        if opacity == 1.0:
            material = MeshStandardMaterial(vertexColors='VertexColors',
                                            roughness=0.3,
                                            metalness=0.0,
                                            side='DoubleSide',
                                            transparent=False)
        else:
            material = MeshStandardMaterial(vertexColors='VertexColors',
                                            roughness=0.3,
                                            metalness=0.0,
                                            side='DoubleSide',
                                            transparent=True,
                                            opacity=opacity)

        # Create a mesh
        isoSurfaceMesh = Mesh(geometry=isoSurfaceGeometry, material=material)

        return isoSurfaceMesh

    def __compute_isosurface(self, data, level, color, extent=None):
        """
        Compute the vertices and faces of an isosurface from grid data

        Parameters
        ----------
        data : numpy.ndarray
            Grid data stored as a numpy 3D tensor
        level : float
            The isocontour value that defines the surface
        color :
            color of a face
        extent : list
            list of [[xmin, xmax], [ymin, ymax], [zmin, zmax]] values that define the bounding box of the mesh,
            otherwise the viewport is used

        Returns
        -------
        a tuple of vertices and faces
        """
        values = skimage.measure.marching_cubes_lewiner(data, level)
        sk_verts, sk_faces, normals, values = values
        x, y, z = sk_verts.T

        # Rescale coordinates to given limits
        if extent:
            xlim, ylim, zlim = extent
            x = x * np.diff(xlim) / (data.shape[0] - 1) + xlim[0]
            y = y * np.diff(ylim) / (data.shape[1] - 1) + ylim[0]
            z = z * np.diff(zlim) / (data.shape[2] - 1) + zlim[0]

        # Assemble the list of vertices
        vertices = []
        for n in range(len(x)):
            vertices.append([x[n], y[n], z[n]])

        # Assemble the list of faces
        faces = []
        for face in sk_faces:
            i, j, k = face
            faces.append((i, j, k, None, (color, color, color), None))
        return (vertices, faces)

    def __molecule_extents(self, atoms_list):
        """
        Compute the extent of a molecule

        Parameters
        ----------
        atoms_list : list(tuple(str, float, float, float))
            A list of tuples containing the atomic symbol and coordinates of the atom using the format
            (atomic symbol,x,y,z)

        Returns
        -------
        A tuple(float, float, float, float, float, float)  containing the minimum and maximum
        coordinates of this molecule in the format (minx, maxx, miny, maxy, minz, maxz)
        """
        minx = min(map(lambda x: x[1], atoms_list))
        maxx = min(map(lambda x: x[1], atoms_list))
        miny = min(map(lambda x: x[2], atoms_list))
        maxy = min(map(lambda x: x[2], atoms_list))
        minz = min(map(lambda x: x[3], atoms_list))
        maxz = min(map(lambda x: x[3], atoms_list))
        return (minx, maxx, miny, maxy, minz, maxz)

    def __center_of_mass(self, atoms_list):
        """
        This function returns the center of mass of a molecule

        Parameters
        ----------
        atoms_list : list(tuple(str, float, float, float))
            A list of tuples containing the atomic symbol and coordinates of the atom using the format
            (atomic symbol,x,y,z)
        """
        X = 0.0
        Y = 0.0
        Z = 0.0
        M = 0.0
        for (symbol, x, y, z) in atoms_list:
            mass = ATOM_DATA[ATOM_SYMBOL_TO_Z[symbol]]['mass']
            X += mass * x
            Y += mass * y
            Z += mass * z
            M += mass
        return (X / M, Y / M, Z / M)
