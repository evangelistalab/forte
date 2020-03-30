import numpy as np
import scipy
import forte
import psi4
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
import glob
import os
import re
from PIL import Image

from .renderer import FortePy3JSRenderer

# %matplotlib notebook
#from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

# run aci
import time

import psi4
import forte


def get_occ(ordm, nact):
    alfa = np.asarray(ordm[0])
    beta = np.asarray(ordm[1])
    sf = np.add(alfa, beta)
    occs = []
    for n in range(nact):
        occs.append(sf[n * nact + n])
    return occs


def run_sci(ints, scf_info, wfn, selector):

    mo_space_map = {
        'RESTRICTED_DOCC': selector.restricted_docc,
        'ACTIVE': selector.active,
        'RESTRICTED_UOCC': selector.restricted_uocc
    }
    mo_reordering = selector.mo_reorder

    #     print(mo_space_map)
    #     print(mo_reordering)

    mo_space_info = forte.make_mo_space_info_from_map(wfn,
                                                      mo_space_map,
                                                      reorder=mo_reordering)

    #     print(mo_space_info.get_corr_abs_mo('ACTIVE'))
    #     print(mo_space_info.get_corr_abs_mo('RESTRICTED_DOCC'))
    #     print(mo_space_info.get_corr_abs_mo('RESTRICTED_UOCC'))

    nact = 0
    for n in mo_space_map['ACTIVE']:
        nact += n

    nactv = sum(mo_space_map['ACTIVE'])
    nrdocc = sum(mo_space_map['RESTRICTED_DOCC'])

    #    options = psi4.core.get_options()
    #    options.set_current_module('FORTE')
    #    forte.forte_options.update_psi_options(options)

    options = forte.forte_options
    options.set_str('ACTIVE_REF_TYPE', 'CISD')
    options.set_int('SCI_MAX_CYCLE', 3)
    options.set_str('SCI_EXCITED_ALGORITHM', 'AVERAGE')
    options.set_str('INT_TYPE', 'CHOLESKY')
    options.set_int('ASCI_CDET', 50)
    options.set_int('ASCI_TDET', 200)

    as_ints = forte.make_active_space_ints(mo_space_info, ints, "ACTIVE",
                                           ["RESTRICTED_DOCC"])

    na = wfn.nalpha()
    nb = wfn.nbeta()
    npair = (na + nb) // 2
    nunpair = na + nb - 2 * npair
    na = npair + nunpair
    na = npair

    state_vec = []
    state_map = {}
    lowest_twice_ms = abs(na - nb)
    for k in range(2):
        na_act = na - nrdocc + k
        nb_act = nb - nrdocc - k
        #         print("k = ", k)
        #         print("na_act = ", na_act)
        #         print("nb_act = ", nb_act)
        #         print("binom({},{}) = {}".format(na_act,nactv,scipy.special.binom(nactv,na_act)))
        #         print("binom({},{}) = {}".format(nb_act,nactv,scipy.special.binom(nactv,nb_act)))
        num_dets = min(
            3,
            int(
                scipy.special.binom(nactv, na_act) *
                scipy.special.binom(nactv, nb_act)))
        #         print(k, num_dets)
        twice_ms = na - nb + 2 * k
        multiplicity = twice_ms + 1
        state = forte.StateInfo(na=na + k,
                                nb=nb - k,
                                multiplicity=multiplicity,
                                twice_ms=twice_ms,
                                irrep=0)
        #         state_vec.append((,num_dets))
        if num_dets > 0:
            state_map[state] = num_dets


#     state_1 = forte.StateInfo(na=na+1,nb=nb-1,multiplicity=3,twice_ms=1,irrep=0)
#     num_states = 3
#     if nact <= 2:
#         num_states = 1
#     state_map = {state_vec[0][0] : state_vec[0][1], state_vec[1][0] : state_vec[1][1]} #, state_1 : num_states}

    as_solver = forte.make_active_space_solver('ASCI', state_map, scf_info,
                                               mo_space_info, as_ints,
                                               forte.forte_options)
    en = as_solver.compute_energy()

    energies = []
    occs = []
    for key, val in en.items():
        for n, energy in enumerate(val):
            ref = as_solver.rdms({(key, key): [(n, n)]}, 1)
            #            ordm = forte.get_rdm_data(ref[0], 1) # TODO-PR reintroduce
            #            occs.append(get_occ(ordm, nact)) # TODO-PR reintroduce
            occs.append(0.0)
            label = "{}-{}".format(n + 1, key.multiplicity_label())
            energies.append((en[key][n], label, key.multiplicity()))

    return energies, occs


def make_mo_plot(energies, occs):
    ### Colors and formatting

    rgb_colors = [
        (0, 0, 0),  # Black  
        (
            0, 78, 139
        ),  # Dark blue                                                                                 
        (89, 171, 131),  # green          
        (225, 27, 46),  # Reddish                
        (255, 205, 65),  # Dark yellow    
        (
            97, 202, 228
        ),  # Light blue                                                                                                                                                                           
        (
            0, 130, 183
        ),  # Blue                                                                                      
    ]

    colors = []
    for c in rgb_colors:
        colors.append(tuple([float(p) / 255.0 for p in c]))
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['figure.figsize'] = 18.0, 5.0
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 18

    ax = plt.gca()

    ax.tick_params(direction='out',
                   width=1.0,
                   length=3.0,
                   axis='x',
                   which='both',
                   bottom=False,
                   top=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axes.xaxis.set_ticklabels([])

    x_vals = [1.0, 2.0]
    yvals = []

    energies2 = []
    for n, energy_label in enumerate(energies):
        energies2.append((energy_label, n))

    energies2 = sorted(energies2)
    energy_values = []
    for n, energy_label in enumerate(energies):
        energy, label, multp = energy_label
        energy_values.append(energy)

    min_e = min(energy_values)
    max_e = max(energy_values)
    delta_e = (max_e - min_e) / 10.0

    last_energy = -10000.0
    for energy_label, n in energies2:
        energy, label, multp = energy_label
        y_vals = [energy, energy]

        occ_str = " | ".join(["%4.2f" % occ for occ in occs[n]])

        #         print(energy)
        #         print(last_energy + delta_e)
        #         print("State " + label + "  occ = |" + occ_str + "|")
        if (energy < last_energy + delta_e):
            label_y = last_energy + delta_e
        else:
            label_y = energy

        plt.plot(x_vals, y_vals, color=colors[multp], linewidth=2.0)
        plt.plot([x_vals[1] + 0.1, 2.5], [energy, label_y], color='k')
        plt.text(2.6, label_y, "State " + label + "  occ = |" + occ_str + "|")
        last_energy = label_y

    ax.set_xlim(0.0, 5.0)

    return plt


#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Selector():
    def __init__(self, ints, scf_info, wfn, cube_dir='.', run=False):
        self.box_width = 200
        self.box_height = 200
        self.wfn = wfn
        self.ints = ints
        self.scf_info = scf_info
        self.cube_dir = cube_dir
        self.run = run
        self.plot = None
        self.debug = False
        self.active_space_selector()

    def list_cube_files(self):
        # convert tga image to png and store state information
        cube_list = []
        for file in glob.glob(self.cube_dir + '/Psi_a*.cube'):
            file = os.path.basename(file)
            file_prefix = os.path.splitext(file)[0]
            cube_list.append(file_prefix)
        return cube_list

    def get_mo_occ(self, na, nb, mo_num):
        occ = 0
        mo_num = int(mo_num)
        if mo_num < na:
            occ += 1
        if mo_num < nb:
            occ += 1
        return occ

    def get_mo_info(self, cube_list, na, nb, epsilon_a):
        mo_list = []
        for file in cube_list:
            mo_type, spin, mo_num, index_irrep, irrep = re.split('-|_', file)
            mo_list.append({
                'mo_num': int(mo_num) - 1,
                'mo_type': mo_type,
                'spin': spin,
                'symmetry': (index_irrep, irrep),
                'file_name': file,
                'occ': self.get_mo_occ(na, nb,
                                       int(mo_num) - 1),
                'epsilon': epsilon_a.get(int(mo_num) - 1)
            })

        return sorted(mo_list, key=lambda k: k['mo_num'])

    def make_mo_widget(self, mo_list):
        ncubes = len(mo_list)
        maxwidgets_per_row = 4
        widget_rows = []

        box_layout = widgets.Layout(border='0px solid black',
                                    width='',
                                    height='',
                                    flex_direction='row',
                                    display='flex')

        rows = []
        current_row = []
        button_list = []
        for mo in mo_list:
            cube_file_name = self.cube_dir + '/' + mo['file_name'] + '.cube'
            mol_renderer = FortePy3JSRenderer(width=self.box_width,
                                              height=self.box_height)
            cubefile = CubeFile(cube_file_name)
            mol_renderer.add_cubefile(cubefile)
            image = mol_renderer.renderer

            label = '{}) {} {} (eps={:.3f},occ={})'.format(
                mo['mo_num'] + 1, mo['symmetry'][0], mo['symmetry'][1],
                mo['epsilon'], mo['occ'])
            button = widgets.ToggleButton(value=False,
                                          description=label,
                                          layout={'width': '200px'})

            hb = widgets.VBox([image, button], layout=box_layout)
            button_list.append(button)
            if len(current_row) >= maxwidgets_per_row:
                rows.append(widgets.HBox(current_row))
                current_row = []
            current_row.append(hb)

        if len(current_row) > 0:
            rows.append(widgets.HBox(current_row))

        if self.debug:
            self.occ_w = widgets.Textarea(value='',
                                          placeholder='',
                                          description='Orbital Spaces Size:',
                                          disabled=True,
                                          layout={
                                              'height': '100px',
                                              'width': '800px',
                                              'margin': '0px 0px 0px 0px'
                                          })
            rows.append(self.occ_w)

            self.plot = widgets.Textarea(value='',
                                         placeholder='',
                                         description='CI Results',
                                         disabled=True,
                                         layout={
                                             'height': '100px',
                                             'width': '800px',
                                             'margin': '0px 0px 0px 0px'
                                         })
            rows.append(self.plot)

        self.out = widgets.Output()
        rows.append(self.out)

        return (widgets.VBox(rows), button_list)

    def active_space_selector(self):
        wfn = self.wfn
        nmo = wfn.nmo()
        na = wfn.nalpha()
        nb = wfn.nbeta()
        epsilon_a = wfn.epsilon_a()
        cube_list = self.list_cube_files()
        mo_list = self.get_mo_info(cube_list, na, nb, epsilon_a)
        print('Setup  MO widget')
        mo_widget, button_w = self.make_mo_widget(mo_list)
        self.mo_list = mo_list
        self.button_w = button_w
        print('Display MO widget')
        display(mo_widget)
        self.generate_occupations(False)
        for w in button_w:
            w.observe(self.generate_occupations, 'value')

    def generate_occupations(self, change):
        mo_list = self.mo_list
        button_w = self.button_w
        debug = True
        wfn = self.wfn
        nmopi = wfn.nmopi()
        ndoccpi = wfn.doccpi()
        nsoccpi = wfn.soccpi()
        nirrep = wfn.nirrep()
        char_table = wfn.molecule().point_group().char_table()
        labels = {char_table.gamma(h).symbol(): h for h in range(nirrep)}
        #         print(labels)
        mo_type_list = {}
        n = 0
        for h in range(nirrep):
            for i in range(ndoccpi[h]):
                mo_type_list[(h, i)] = [n, 'core']
                n += 1
            for i in range(ndoccpi[h], ndoccpi[h] + nsoccpi[h]):
                mo_type_list[(h, i)] = [n, 'active']
                n += 1
            for i in range(ndoccpi[h] + nsoccpi[h], nmopi[h]):
                mo_type_list[(h, i)] = [n, 'virtual']
                n += 1

        for mo, w in zip(mo_list, button_w):
            if w.value:
                irrep_index, irrep = mo['symmetry']
                h = labels[irrep.upper()]
                i = int(irrep_index) - 1
                mo_type_list[(h, i)][1] = 'active'

        self.restricted_docc = [0] * nirrep
        self.active = [0] * nirrep
        self.restricted_uocc = [0] * nirrep

        sorted_mos = []
        for h_i, mo in mo_type_list.items():
            n, mo_type = mo
            h, i = h_i
            if mo_type == 'core':
                self.restricted_docc[h] += 1
                sorted_mos.append((h, 0, i, n))
            if mo_type == 'active':
                self.active[h] += 1
                sorted_mos.append((h, 1, i, n))
            if mo_type == 'virtual':
                self.restricted_uocc[h] += 1
                sorted_mos.append((h, 2, i, n))

        self.mo_reorder = [e[3] for e in sorted(sorted_mos)]
        s = 'mo_reordering = {}\nrestricted_docc = {}\nactive = {}\nrestricted_uocc = {}'.format(
            self.mo_reorder, self.restricted_docc, self.active,
            self.restricted_uocc)
        if self.run:
            energies, occs = run_sci(self.ints, self.scf_info, self.wfn, self)
            if self.debug:
                results = ""
                for e, occ in zip(energies, occs):
                    occ_str = " ".join(["%4.2f" % float(o) for o in occ])
                    results += "{} -> {}\n".format(e, occ_str)
                self.plot.value = results
            with self.out:
                clear_output(wait=True)
                plt = make_mo_plot(energies, occs)
                plt.show()
        if self.debug:
            self.occ_w.value = s
