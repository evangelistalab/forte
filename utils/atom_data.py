"""
    A map from atom symbols to atomic number
"""
ATOM_SYMBOL_TO_Z = {
    'Xx': 0,
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V': 23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y': 39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I': 53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W': 74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    'Fr': 87,
    'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U': 92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100
}
"""
    A map from atom symbols to atomic data
"""
ATOM_DATA = [{
    'symbol': 'Xx',
    'name': 'Dummy',
    'mass': 0,
    'radius_covalent': 0.18,
    'radius_VDW': 0.69,
    'color': [17, 127, 178]
}, {
    'symbol': 'H',
    'name': 'Hydrogen',
    'mass': 1.00784,
    'radius_covalent': 0.32,
    'radius_VDW': 1.2,
    'color': [240, 240, 240]
}, {
    'symbol': 'He',
    'name': 'Helium',
    'mass': 4.0026,
    'radius_covalent': 0.46,
    'radius_VDW': 1.43,
    'color': [217, 255, 255]
}, {
    'symbol': 'Li',
    'name': 'Lithium',
    'mass': 6.938,
    'radius_covalent': 1.33,
    'radius_VDW': 2.12,
    'color': [204, 128, 255]
}, {
    'symbol': 'Be',
    'name': 'Beryllium',
    'mass': 9.01218,
    'radius_covalent': 1.02,
    'radius_VDW': 1.98,
    'color': [194, 255, 0]
}, {
    'symbol': 'B',
    'name': 'Boron',
    'mass': 10.806,
    'radius_covalent': 0.85,
    'radius_VDW': 1.91,
    'color': [255, 181, 181]
}, {
    'symbol': 'C',
    'name': 'Carbon',
    'mass': 12.011,
    'radius_covalent': 0.75,
    'radius_VDW': 1.77,
    'color': [127, 127, 127]
}, {
    'symbol': 'N',
    'name': 'Nitrogen',
    'mass': 14.006,
    'radius_covalent': 0.71,
    'radius_VDW': 1.66,
    'color': [48, 80, 255]
}, {
    'symbol': 'O',
    'name': 'Oxygen',
    'mass': 15.9994,
    'radius_covalent': 0.63,
    'radius_VDW': 1.5,
    'color': [200, 13, 13]
}, {
    'symbol': 'F',
    'name': 'Fluorine',
    'mass': 18.9984,
    'radius_covalent': 0.64,
    'radius_VDW': 1.46,
    'color': [178, 255, 255]
}, {
    'symbol': 'Ne',
    'name': 'Neon',
    'mass': 20.1797,
    'radius_covalent': 0.67,
    'radius_VDW': 1.58,
    'color': [178, 227, 245]
}, {
    'symbol': 'Na',
    'name': 'Sodium',
    'mass': 22.9898,
    'radius_covalent': 1.55,
    'radius_VDW': 2.5,
    'color': [171, 91, 242]
}, {
    'symbol': 'Mg',
    'name': 'Magnesium',
    'mass': 24.305,
    'radius_covalent': 1.39,
    'radius_VDW': 2.51,
    'color': [138, 255, 0]
}, {
    'symbol': 'Al',
    'name': 'Aluminium',
    'mass': 26.9815,
    'radius_covalent': 1.26,
    'radius_VDW': 2.25,
    'color': [191, 166, 166]
}, {
    'symbol': 'Si',
    'name': 'Silicon',
    'mass': 28.0855,
    'radius_covalent': 1.16,
    'radius_VDW': 2.19,
    'color': [240, 200, 160]
}, {
    'symbol': 'P',
    'name': 'Phosphorus',
    'mass': 30.9738,
    'radius_covalent': 1.11,
    'radius_VDW': 1.9,
    'color': [255, 128, 0]
}, {
    'symbol': 'S',
    'name': 'Sulfur',
    'mass': 32.065,
    'radius_covalent': 1.03,
    'radius_VDW': 1.89,
    'color': [255, 255, 48]
}, {
    'symbol': 'Cl',
    'name': 'Chlorine',
    'mass': 35.453,
    'radius_covalent': 0.99,
    'radius_VDW': 1.82,
    'color': [31, 240, 31]
}, {
    'symbol': 'Ar',
    'name': 'Argon',
    'mass': 39.948,
    'radius_covalent': 0.96,
    'radius_VDW': 1.83,
    'color': [128, 209, 227]
}, {
    'symbol': 'K',
    'name': 'Potassium',
    'mass': 39.0983,
    'radius_covalent': 1.96,
    'radius_VDW': 2.73,
    'color': [143, 64, 212]
}, {
    'symbol': 'Ca',
    'name': 'Calcium',
    'mass': 40.078,
    'radius_covalent': 1.71,
    'radius_VDW': 2.62,
    'color': [61, 255, 0]
}, {
    'symbol': 'Sc',
    'name': 'Scandium',
    'mass': 44.9559,
    'radius_covalent': 1.48,
    'radius_VDW': 2.58,
    'color': [230, 230, 230]
}, {
    'symbol': 'Ti',
    'name': 'Titanium',
    'mass': 47.867,
    'radius_covalent': 1.36,
    'radius_VDW': 2.46,
    'color': [191, 194, 199]
}, {
    'symbol': 'V',
    'name': 'Vanadium',
    'mass': 50.9415,
    'radius_covalent': 1.34,
    'radius_VDW': 2.42,
    'color': [166, 166, 171]
}, {
    'symbol': 'Cr',
    'name': 'Chromium',
    'mass': 51.9961,
    'radius_covalent': 1.22,
    'radius_VDW': 2.45,
    'color': [138, 153, 199]
}, {
    'symbol': 'Mn',
    'name': 'Manganese',
    'mass': 54.938,
    'radius_covalent': 1.19,
    'radius_VDW': 2.45,
    'color': [156, 122, 199]
}, {
    'symbol': 'Fe',
    'name': 'Iron',
    'mass': 55.845,
    'radius_covalent': 1.16,
    'radius_VDW': 2.44,
    'color': [224, 102, 51]
}, {
    'symbol': 'Co',
    'name': 'Cobalt',
    'mass': 58.9332,
    'radius_covalent': 1.11,
    'radius_VDW': 2.4,
    'color': [240, 144, 160]
}, {
    'symbol': 'Ni',
    'name': 'Nickel',
    'mass': 58.6934,
    'radius_covalent': 1.1,
    'radius_VDW': 2.4,
    'color': [80, 208, 80]
}, {
    'symbol': 'Cu',
    'name': 'Copper',
    'mass': 63.546,
    'radius_covalent': 1.12,
    'radius_VDW': 2.38,
    'color': [200, 128, 51]
}, {
    'symbol': 'Zn',
    'name': 'Zinc',
    'mass': 65.38,
    'radius_covalent': 1.18,
    'radius_VDW': 2.39,
    'color': [125, 128, 176]
}, {
    'symbol': 'Ga',
    'name': 'Gallium',
    'mass': 69.723,
    'radius_covalent': 1.24,
    'radius_VDW': 2.32,
    'color': [194, 143, 143]
}, {
    'symbol': 'Ge',
    'name': 'Germanium',
    'mass': 72.64,
    'radius_covalent': 1.21,
    'radius_VDW': 2.29,
    'color': [102, 143, 143]
}, {
    'symbol': 'As',
    'name': 'Arsenic',
    'mass': 74.9216,
    'radius_covalent': 1.21,
    'radius_VDW': 1.88,
    'color': [189, 128, 227]
}, {
    'symbol': 'Se',
    'name': 'Selenium',
    'mass': 78.971,
    'radius_covalent': 1.16,
    'radius_VDW': 1.82,
    'color': [255, 161, 0]
}, {
    'symbol': 'Br',
    'name': 'Bromine',
    'mass': 79.904,
    'radius_covalent': 1.14,
    'radius_VDW': 1.86,
    'color': [166, 41, 41]
}, {
    'symbol': 'Kr',
    'name': 'Krypton',
    'mass': 83.798,
    'radius_covalent': 1.17,
    'radius_VDW': 2.25,
    'color': [92, 184, 209]
}, {
    'symbol': 'Rb',
    'name': 'Rubidium',
    'mass': 85.4678,
    'radius_covalent': 2.1,
    'radius_VDW': 3.21,
    'color': [112, 46, 176]
}, {
    'symbol': 'Sr',
    'name': 'Strontium',
    'mass': 87.62,
    'radius_covalent': 1.85,
    'radius_VDW': 2.84,
    'color': [0, 255, 0]
}, {
    'symbol': 'Y',
    'name': 'Yttrium',
    'mass': 88.9058,
    'radius_covalent': 1.63,
    'radius_VDW': 2.75,
    'color': [148, 255, 255]
}, {
    'symbol': 'Zr',
    'name': 'Zirconium',
    'mass': 91.224,
    'radius_covalent': 1.54,
    'radius_VDW': 2.52,
    'color': [148, 224, 224]
}, {
    'symbol': 'Nb',
    'name': 'Niobium',
    'mass': 92.9064,
    'radius_covalent': 1.47,
    'radius_VDW': 2.56,
    'color': [115, 194, 201]
}, {
    'symbol': 'Mo',
    'name': 'Molybdenum',
    'mass': 95.95,
    'radius_covalent': 1.38,
    'radius_VDW': 2.45,
    'color': [84, 181, 181]
}, {
    'symbol': 'Tc',
    'name': 'Technetium',
    'mass': 97,
    'radius_covalent': 1.28,
    'radius_VDW': 2.44,
    'color': [59, 158, 158]
}, {
    'symbol': 'Ru',
    'name': 'Ruthenium',
    'mass': 101.07,
    'radius_covalent': 1.25,
    'radius_VDW': 2.46,
    'color': [36, 143, 143]
}, {
    'symbol': 'Rh',
    'name': 'Rhodium',
    'mass': 102.9055,
    'radius_covalent': 1.25,
    'radius_VDW': 2.44,
    'color': [10, 125, 140]
}, {
    'symbol': 'Pd',
    'name': 'Palladium',
    'mass': 106.42,
    'radius_covalent': 1.2,
    'radius_VDW': 2.15,
    'color': [0, 105, 133]
}, {
    'symbol': 'Ag',
    'name': 'Silver',
    'mass': 107.8682,
    'radius_covalent': 1.28,
    'radius_VDW': 2.53,
    'color': [192, 192, 192]
}, {
    'symbol': 'Cd',
    'name': 'Cadmium',
    'mass': 112.414,
    'radius_covalent': 1.36,
    'radius_VDW': 2.49,
    'color': [255, 217, 143]
}, {
    'symbol': 'In',
    'name': 'Indium',
    'mass': 114.818,
    'radius_covalent': 1.42,
    'radius_VDW': 2.43,
    'color': [166, 117, 115]
}, {
    'symbol': 'Sn',
    'name': 'Tin',
    'mass': 118.71,
    'radius_covalent': 1.4,
    'radius_VDW': 2.42,
    'color': [102, 128, 128]
}, {
    'symbol': 'Sb',
    'name': 'Antimony',
    'mass': 121.76,
    'radius_covalent': 1.4,
    'radius_VDW': 2.47,
    'color': [158, 99, 181]
}, {
    'symbol': 'Te',
    'name': 'Tellurium',
    'mass': 127.6,
    'radius_covalent': 1.36,
    'radius_VDW': 1.99,
    'color': [211, 122, 0]
}, {
    'symbol': 'I',
    'name': 'Iodine',
    'mass': 126.9045,
    'radius_covalent': 1.33,
    'radius_VDW': 2.04,
    'color': [148, 0, 148]
}, {
    'symbol': 'Xe',
    'name': 'Xenon',
    'mass': 131.293,
    'radius_covalent': 1.31,
    'radius_VDW': 2.06,
    'color': [66, 158, 176]
}, {
    'symbol': 'Cs',
    'name': 'Caesium',
    'mass': 132.9055,
    'radius_covalent': 2.32,
    'radius_VDW': 3.48,
    'color': [87, 23, 143]
}, {
    'symbol': 'Ba',
    'name': 'Barium',
    'mass': 137.327,
    'radius_covalent': 1.96,
    'radius_VDW': 3.03,
    'color': [0, 201, 0]
}, {
    'symbol': 'La',
    'name': 'Lanthanum',
    'mass': 138.9055,
    'radius_covalent': 1.8,
    'radius_VDW': 2.98,
    'color': [112, 212, 255]
}, {
    'symbol': 'Ce',
    'name': 'Cerium',
    'mass': 140.116,
    'radius_covalent': 1.63,
    'radius_VDW': 2.88,
    'color': [255, 255, 199]
}, {
    'symbol': 'Pr',
    'name': 'Praseodymium',
    'mass': 140.9077,
    'radius_covalent': 1.76,
    'radius_VDW': 2.92,
    'color': [217, 255, 199]
}, {
    'symbol': 'Nd',
    'name': 'Neodymium',
    'mass': 144.242,
    'radius_covalent': 1.74,
    'radius_VDW': 2.95,
    'color': [199, 255, 199]
}, {
    'symbol': 'Pm',
    'name': 'Promethium',
    'mass': 145,
    'radius_covalent': 1.73,
    'radius_VDW': 2.9,
    'color': [163, 255, 199]
}, {
    'symbol': 'Sm',
    'name': 'Samarium',
    'mass': 150.36,
    'radius_covalent': 1.72,
    'radius_VDW': 2.87,
    'color': [143, 255, 199]
}, {
    'symbol': 'Eu',
    'name': 'Europium',
    'mass': 151.964,
    'radius_covalent': 1.68,
    'radius_VDW': 2.83,
    'color': [97, 255, 199]
}, {
    'symbol': 'Gd',
    'name': 'Gadolinium',
    'mass': 157.25,
    'radius_covalent': 1.69,
    'radius_VDW': 2.79,
    'color': [69, 255, 199]
}, {
    'symbol': 'Tb',
    'name': 'Terbium',
    'mass': 158.9253,
    'radius_covalent': 1.68,
    'radius_VDW': 2.87,
    'color': [48, 255, 199]
}, {
    'symbol': 'Dy',
    'name': 'Dysprosium',
    'mass': 162.5,
    'radius_covalent': 1.67,
    'radius_VDW': 2.81,
    'color': [31, 255, 199]
}, {
    'symbol': 'Ho',
    'name': 'Holmium',
    'mass': 164.9303,
    'radius_covalent': 1.66,
    'radius_VDW': 2.83,
    'color': [0, 255, 156]
}, {
    'symbol': 'Er',
    'name': 'Erbium',
    'mass': 167.259,
    'radius_covalent': 1.65,
    'radius_VDW': 2.79,
    'color': [0, 230, 117]
}, {
    'symbol': 'Tm',
    'name': 'Thulium',
    'mass': 168.9342,
    'radius_covalent': 1.64,
    'radius_VDW': 2.8,
    'color': [0, 212, 82]
}, {
    'symbol': 'Yb',
    'name': 'Ytterbium',
    'mass': 173.045,
    'radius_covalent': 1.7,
    'radius_VDW': 2.74,
    'color': [0, 191, 56]
}, {
    'symbol': 'Lu',
    'name': 'Lutetium',
    'mass': 174.9668,
    'radius_covalent': 1.62,
    'radius_VDW': 2.63,
    'color': [0, 171, 36]
}, {
    'symbol': 'Hf',
    'name': 'Hafnium',
    'mass': 178.49,
    'radius_covalent': 1.52,
    'radius_VDW': 2.53,
    'color': [77, 194, 255]
}, {
    'symbol': 'Ta',
    'name': 'Tantalum',
    'mass': 180.9479,
    'radius_covalent': 1.46,
    'radius_VDW': 2.57,
    'color': [77, 166, 255]
}, {
    'symbol': 'W',
    'name': 'Tungsten',
    'mass': 183.84,
    'radius_covalent': 1.37,
    'radius_VDW': 2.49,
    'color': [33, 148, 214]
}, {
    'symbol': 'Re',
    'name': 'Rhenium',
    'mass': 186.207,
    'radius_covalent': 1.31,
    'radius_VDW': 2.48,
    'color': [38, 102, 150]
}, {
    'symbol': 'Os',
    'name': 'Osmium',
    'mass': 190.23,
    'radius_covalent': 1.29,
    'radius_VDW': 2.41,
    'color': [38, 102, 150]
}, {
    'symbol': 'Ir',
    'name': 'Iridium',
    'mass': 192.217,
    'radius_covalent': 1.22,
    'radius_VDW': 2.29,
    'color': [23, 84, 135]
}, {
    'symbol': 'Pt',
    'name': 'Platinum',
    'mass': 195.084,
    'radius_covalent': 1.23,
    'radius_VDW': 2.32,
    'color': [208, 208, 224]
}, {
    'symbol': 'Au',
    'name': 'Gold',
    'mass': 196.9666,
    'radius_covalent': 1.24,
    'radius_VDW': 2.45,
    'color': [255, 209, 35]
}, {
    'symbol': 'Hg',
    'name': 'Mercury',
    'mass': 200.592,
    'radius_covalent': 1.33,
    'radius_VDW': 2.47,
    'color': [184, 194, 208]
}, {
    'symbol': 'Tl',
    'name': 'Thallium',
    'mass': 204.38,
    'radius_covalent': 1.44,
    'radius_VDW': 2.6,
    'color': [166, 84, 77]
}, {
    'symbol': 'Pb',
    'name': 'Lead',
    'mass': 207.2,
    'radius_covalent': 1.44,
    'radius_VDW': 2.54,
    'color': [87, 89, 97]
}, {
    'symbol': 'Bi',
    'name': 'Bismuth',
    'mass': 208.9804,
    'radius_covalent': 1.51,
    'radius_VDW': 2.5,
    'color': [158, 79, 181]
}, {
    'symbol': 'Po',
    'name': 'Polonium',
    'mass': 209,
    'radius_covalent': 1.45,
    'radius_VDW': 2.5,
    'color': [171, 92, 0]
}, {
    'symbol': 'At',
    'name': 'Astatine',
    'mass': 210,
    'radius_covalent': 1.47,
    'radius_VDW': 2.5,
    'color': [117, 79, 69]
}, {
    'symbol': 'Rn',
    'name': 'Radon',
    'mass': 222,
    'radius_covalent': 1.42,
    'radius_VDW': 2.5,
    'color': [66, 130, 150]
}, {
    'symbol': 'Fr',
    'name': 'Francium',
    'mass': 223,
    'radius_covalent': 2.23,
    'radius_VDW': 2.5,
    'color': [66, 0, 102]
}, {
    'symbol': 'Ra',
    'name': 'Radium',
    'mass': 226,
    'radius_covalent': 2.01,
    'radius_VDW': 2.8,
    'color': [0, 124, 0]
}, {
    'symbol': 'Ac',
    'name': 'Actinium',
    'mass': 227,
    'radius_covalent': 1.86,
    'radius_VDW': 2.93,
    'color': [112, 170, 249]
}, {
    'symbol': 'Th',
    'name': 'Thorium',
    'mass': 232.0377,
    'radius_covalent': 1.75,
    'radius_VDW': 2.88,
    'color': [0, 186, 255]
}, {
    'symbol': 'Pa',
    'name': 'Protactinium',
    'mass': 231.0358,
    'radius_covalent': 1.69,
    'radius_VDW': 2.71,
    'color': [0, 160, 255]
}, {
    'symbol': 'U',
    'name': 'Uranium',
    'mass': 238.0289,
    'radius_covalent': 1.7,
    'radius_VDW': 2.82,
    'color': [0, 142, 255]
}, {
    'symbol': 'Np',
    'name': 'Neptunium',
    'mass': 237,
    'radius_covalent': 1.71,
    'radius_VDW': 2.81,
    'color': [0, 127, 255]
}, {
    'symbol': 'Pu',
    'name': 'Plutonium',
    'mass': 244,
    'radius_covalent': 1.72,
    'radius_VDW': 2.83,
    'color': [0, 107, 255]
}, {
    'symbol': 'Am',
    'name': 'Americium',
    'mass': 243,
    'radius_covalent': 1.66,
    'radius_VDW': 3.05,
    'color': [84, 91, 242]
}, {
    'symbol': 'Cm',
    'name': 'Curium',
    'mass': 247,
    'radius_covalent': 1.66,
    'radius_VDW': 3.38,
    'color': [119, 91, 226]
}, {
    'symbol': 'Bk',
    'name': 'Berkelium',
    'mass': 247,
    'radius_covalent': 1.68,
    'radius_VDW': 3.05,
    'color': [137, 79, 226]
}, {
    'symbol': 'Cf',
    'name': 'Californium',
    'mass': 251,
    'radius_covalent': 1.68,
    'radius_VDW': 3.0,
    'color': [160, 53, 211]
}, {
    'symbol': 'Es',
    'name': 'Einsteinium',
    'mass': 252,
    'radius_covalent': 1.65,
    'radius_VDW': 3.0,
    'color': [178, 30, 211]
}, {
    'symbol': 'Fm',
    'name': 'Fermium',
    'mass': 257,
    'radius_covalent': 1.67,
    'radius_VDW': 3.0,
    'color': [178, 30, 186]
}]
