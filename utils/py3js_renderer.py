import math
import re
import collections
import numpy as np
import skimage.measure

from IPython.display import display
from pythreejs import *
from .atom_data import *

def xyz_to_atoms_list(xyz):
    atoms_list = []
    re_xyz = re.compile(r"(\w+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)")
    for line in xyz.split('\n'):
        m = re.search(re_xyz,line)
        if (m):
            label,x,y,z = m.groups()
            atoms_list.append((label,float(x),float(y),float(z)))
    return atoms_list

def compute_isosurface(data, level=None, color=None, extent=None):
    """Plot a surface at constant value (like a 2d contour).
    :param data: 3d numpy array
    :param float level: value where the surface should lie
    :param color: color of the surface, although it can be an array, the length is difficult to predict beforehand,
                  if per vertex color are needed, it is better to set them on the returned mesh afterwards.
    :param bool wireframe: draw lines between the vertices
    :param bool surface: draw faces/triangles between the vertices
    :param bool controls: add controls to change the isosurface
    :param extent: list of [[xmin, xmax], [ymin, ymax], [zmin, zmax]] values that define the bounding box of the mesh,
                   otherwise the viewport is used
    :return: :any:`Mesh`
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
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, width=400, height=400):
        self.atoms = []
        self.bonds = []
        self.iso = []
        self.width = width
        self.height = height
        self.bond_radius = 0.2  # a.u.
        self.bond_color = '#555555'
        self.angtobohr = 1.88973  # TODO: use Psi4's value
        self.atom_geometries = {}
        self.atom_materials = {}
        self.bond_materials = {}
        self.bond_geometry = None
        self.scene = Scene()

    def get_atom_geometry(self, symbol, shininess=75):
        if symbol in self.atom_geometries:
            return self.atom_geometries[symbol]
        atom_data = ATOM_DATA[ATOM_SYMBOL_TO_Z[symbol]]
        radius_covalent = atom_data['radius_covalent'] * self.angtobohr
        geometry = SphereGeometry(
            radius=0.6 * radius_covalent, widthSegments=24, heightSegments=24)
        self.atom_geometries[symbol] = geometry
        return geometry

    def get_bond_geometry(self):
        if self.bond_geometry:
            return self.bond_geometry
        self.bond_geometry = CylinderGeometry(
            radiusTop=self.bond_radius,
            radiusBottom=self.bond_radius,
            height=1,
            radialSegments=12,
            heightSegments=6,
            openEnded=False)
        return self.bond_geometry

    def get_atom_material(self, symbol, shininess=75):
        if symbol in self.atom_materials:
            return self.atom_materials[symbol]
        atom_data = ATOM_DATA[ATOM_SYMBOL_TO_Z[symbol]]
        color = 'rgb({0[0]},{0[1]},{0[2]})'.format(atom_data['color'])
#        material = MeshPhongMaterial(color=color, shininess=shininess)
        material = MeshStandardMaterial(
            color=color,
            roughness=0.25,
            metalness=0.1)
        self.atom_materials[symbol] = material
        return material

    def get_bond_material(self, color, shininess=75):
        if color in self.bond_materials:
            return self.bond_materials[color]
#        material = MeshPhongMaterial(color=color, shininess=shininess)
        material = MeshStandardMaterial(
            color=color,
            roughness=0.25,
            metalness=0.1)
        self.bond_materials[color] = material
        return material

    def atom(self, atom_info):
        symbol, x, y, z = atom_info
        geometry = self.get_atom_geometry(symbol)
        material = self.get_atom_material(symbol)
        mesh = Mesh(geometry=geometry, material=material, position=[x, y, z])
        return mesh

    def cylinder(self, xyz1, xyz2, radius1, radius2, color):
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        d = sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        position = [(x2 + x1) / 2, (y2 + y1) / 2, (z2 + z1) / 2]
        geometry = CylinderGeometry(
            radiusTop=radius2,
            radiusBottom=radius1,
            height=d,
            radialSegments=12,
            heightSegments=6,
            openEnded=False)
        material = MeshPhongMaterial(color=color, shininess=100)
        mesh = Mesh(geometry=geometry, material=material, position=position)

        # If the bond rotation is 180 deg then return
        if y1 - y2 == d:
            mesh.rotateX(3.14159265359) #math.pi)
            return mesh

        R = self.cylinder_rotation_matrix(xyz1, xyz2, d)
        mesh.setRotationFromMatrix(R)
        return mesh

    def add_cylinder(self, xyz1, xyz2, radius1, radius2, color):
        mesh = self.cylinder(xyz1, xyz2, radius1, radius2, color)
        self.scene.add(mesh)

    def add_arrow(self,
                  xyz1,
                  xyz2,
                  color,
                  radius_small=0.075,
                  radius_large=0.2):
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        fraction = 0.7
        xyz_base = [
            x1 + (x2 - x1) * fraction, y1 + (y2 - y1) * fraction,
            z1 + (z2 - z1) * fraction
        ]
        mesh = self.cylinder(xyz1, xyz_base, radius_small, radius_small, color)
        self.scene.add(mesh)
        mesh = self.cylinder(xyz_base, xyz2, radius_large, 0.0, color)
        self.scene.add(mesh)

    def bond(self, atom1_info, atom2_info, radius=None):
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
        geometry = self.get_bond_geometry()
        material = self.get_bond_material(color=self.bond_color)
        mesh = Mesh(geometry=geometry, material=material, position=position)
        mesh.scale = (1,d,1)
        # If the bond rotation is 180 deg then return
        if y1 - y2 == d:
            mesh.rotateX(3.14159265359) #math.pi)
            return mesh

        R = self.cylinder_rotation_matrix([x1,y1,z1], [x2,y2,z2], d)
        mesh.setRotationFromMatrix(R)
        return mesh

    def bond_cutoff(self, r1, r2):
        return 1.5 * self.angtobohr * (r1 + r2)

    def cylinder_rotation_matrix(self, xyz1, xyz2, d):
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
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

    def isosurface(self, data, level=None, color=None, extent=None):
        vertices, faces = compute_isosurface(
            data, level=level, color=color, extent=extent)

        # Create the geometry
        isoSurfaceGeometry = Geometry(vertices=vertices, faces=faces)

        # Calculate normals per vertex for round edges
        isoSurfaceGeometry.exec_three_obj_method('computeVertexNormals')

        # Note that the material need to be told to use the vertex colors
#        material = MeshPhongMaterial(
#            vertexColors='VertexColors',
#            shinyness=5,
#            side='DoubleSide',
#            transparent=True,
#            opacity=1.0)

        material = MeshStandardMaterial(
            vertexColors='VertexColors',
            roughness=0.3,
            metalness=0.0,
            side='DoubleSide',
            transparent=True,
            opacity=1.0)



        # Create a mesh
        isoSurfaceMesh = Mesh(geometry=isoSurfaceGeometry, material=material)

        return isoSurfaceMesh


    def add_molecule_xyz(self, xyz, bohr=False):
        atoms_list = xyz_to_atoms_list(xyz)
        self.add_molecule_dict(atoms_list, bohr)


    def add_molecule_dict(self, atoms_list, bohr=False):
        """Add a molecule"""
        if bohr == False:
            atoms_list2 = []
            for atom in atoms_list:
                symbol, x, y, z = atom
                new_atom = (symbol, self.angtobohr * x, self.angtobohr * y,
                            self.angtobohr * z)
                atoms_list2.append(new_atom)
            atoms_list = atoms_list2

        self.com = self.center_of_mass(atoms_list) # TODO: fix this if multiple molecules are added

        self.molecule = Group()

#        self.atoms = []
#        for atom in atoms_list:
#            self.atoms.append(self.atom(atom))

        # Performance optimization using CloneArray
        atom_positions = collections.defaultdict(list)
        for atom in atoms_list:
            symbol, x, y, z = atom
            atom_positions[symbol].append([x,y,z])

        for atom_type in atom_positions:
            atom_mesh = self.atom((atom_type,0.0,0.0,0.0))
            clone_geom = CloneArray(original=atom_mesh,positions=atom_positions[atom_type])
            self.atoms.append(clone_geom)

        self.bonds = []
        for i in range(len(atoms_list)):
            atom1 = atoms_list[i]
            for j in range(i + 1, len(atoms_list)):
                atom2 = atoms_list[j]
                bond = self.bond(atom1, atom2)
                if bond:
                    self.bonds.append(bond)

        aspect = float(self.width) / float(self.height)

        # determine the molecule dimension (l)
        extents = self.molecule_extents(atoms_list)
        l = 1.75 * max(max(extents), abs(min(extents)))

        camera = OrthographicCamera(
            left=-l * aspect,
            right=l * aspect,
            top=l,
            bottom=-l,
            position=[0, l * 1.5, l * 1.5],
            up=[0, 1, 0],
            children=[
                DirectionalLight(
                    color='white', position=[5, 5, 1], intensity=0.5)
            ],
            near=.1,
            far=1000)

        scene_children = [camera, AmbientLight(color='#999999')]
        scene_children = scene_children + self.atoms + self.bonds  #+ x_box,y_box,z_box

        self.scene.add(scene_children)

        self.renderer = Renderer(
            camera=camera,
            scene=self.scene,
            controls=[OrbitControls(controlling=camera)],
            width=self.width,
            height=self.height)

        return self.renderer

    def add_cubefile(self, cube, add_geom=True, levels=None, colors=None):
        if add_geom:
            atoms_list = []
            for Z, xyz in zip(cube.atom_numbers, cube.atom_coords):
                symbol = ATOM_DATA[Z]['symbol']
                atoms_list.append((symbol, xyz[0], xyz[1], xyz[2]))
            self.add_molecule_dict(atoms_list, bohr=True)

        if not levels:
            if cube.levels:
                levels = cube.levels
        if colors == None:
            colors = ['#f2a900', '#0033a0']
        data = cube.data
        extent = [[cube.min[0], cube.max[0]],
                  [cube.min[1], cube.max[1]],
                  [cube.min[2], cube.max[2]]]
        for level, color in zip(levels, colors):
            if abs(level) > 1.0e-4:
                mesh = self.isosurface(
                    data, level=level, color=color, extent=extent)
                self.iso.append(mesh)
                self.scene.add(mesh)

    def molecule_extents(self, atoms_list):
        minx = min(map(lambda x: x[1], atoms_list))
        maxx = min(map(lambda x: x[1], atoms_list))
        miny = min(map(lambda x: x[2], atoms_list))
        maxy = min(map(lambda x: x[2], atoms_list))
        minz = min(map(lambda x: x[3], atoms_list))
        maxz = min(map(lambda x: x[3], atoms_list))
        return (minx, maxx, miny, maxy, minz, maxz)

    def shift_to_com(self):
        X, Y ,Z = self.com
        for atom in self.atoms:
            pos = atom.position
            atom.position = (pos[0] - X, pos[1] - Y,pos[2] - Z)
        for bond in self.bonds:
            pos = bond.position
            bond.position = (pos[0] - X, pos[1] - Y,pos[2] - Z)

    def center_of_mass(self, atoms_list):
        X = 0.0
        Y = 0.0
        Z = 0.0
        M = 0.0
        for atom_info in atoms_list:
            symbol, x, y, z = atom_info
            mass = ATOM_DATA[
            ATOM_SYMBOL_TO_Z[symbol]]['mass']
            X += mass * x
            Y += mass * y
            Z += mass * z
            M += mass
        return (X / M, Y / M, Z / M)

    def display(self):
        display(self.renderer)

    def renderer(self):
        return self.renderer


#    def load_cube_geometry(self, filename, do_display=True):
#        cube = parse_cube(filename)
#        atoms_list = []
#        for Z, xyz in zip(cube['atom_numbers'], cube['atom_coords']):
#            symbol = ATOM_DATA[Z]['symbol']
#            atoms_list.append((symbol, xyz[0], xyz[1], xyz[2]))
#        self.add_molecule_dict(atoms_list, bohr=True)

#    def add_isosurface(self, filename, levels=None, colors=None):
#        cube = parse_cube(filename)
#        if 'levels' in cube:
#            levels = cube['levels']
#        if colors == None:
#            colors = ['#f2a900', '#0033a0']
#        data = cube['data']
#        extent = [[cube['minx'], cube['maxx']], [cube['miny'], cube['maxy']],
#                  [cube['minz'], cube['maxz']]]
#        for level, color in zip(levels, colors):
#            if abs(level) > 1.0e-4:
#                mesh = self.isosurface(
#                    data, level=level, color=color, extent=extent)
#                self.iso.append(mesh)
#                self.scene.add(mesh)
