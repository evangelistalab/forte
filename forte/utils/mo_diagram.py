# -*- coding: utf-8 -*-

import plotly.graph_objects as go


class MODiagram:
    """
    A class to make plots of MO levels

    Methods
    -------
    MODiagram(wfn)
        Make a MO plot from a psi4 Wavefunction object
    save(filename)
        Save the plot to a figure file
    """
    def __init__(self,
                 wfn,
                 plot_range=None,
                 style='arrows',
                 title='Molecular orbital diagram',
                 height=600,
                 width=None,
                 occupied_color='rgba(0,0,0,1.0)',
                 unoccupied_color='rgba(100,100,100,1.0)'):
        self.title = title
        self.plot_range = plot_range
        self.style = style
        self.height = height
        self.width = width
        self.occupied_color = occupied_color
        self.unoccupied_color = unoccupied_color
        self.epsilon_a = wfn.epsilon_a().nph
        self.epsilon_b = wfn.epsilon_b().nph
        self.nirrep = wfn.nirrep()
        self.nmopi = wfn.nmopi()
        self.nalphapi = wfn.nalphapi()
        self.nbetapi = wfn.nbetapi()
        self.irrep_labels = wfn.molecule().irrep_labels()
        if (self.__check_plotly_is_installed()):
            self.__compute_levels_scf()

    def save(self, filename):
        if self.fig == None:
            return
        self.fig.write_image(filename)

    def __check_plotly_is_installed(self):
        import sys
        import subprocess
        import pkg_resources

        required = {'plotly'}
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        if len(missing) > 0:
            print(
                f"The following packages are missing: {','.join(missing)}. Please install them before usign MODiagram"
            )
            return False
        return True

    def __compute_levels_scf(self):
        # store data for each orbital class (epsilon,irrep,index_in_irrep)
        ea_occ_data = sorted([(self.epsilon_a[h][i], h, i, 1)
                              for h in range(self.nirrep)
                              for i in range(self.nalphapi[h])])
        ea_vir_data = sorted([(self.epsilon_a[h][i], h, i, 0)
                              for h in range(self.nirrep)
                              for i in range(self.nalphapi[h], self.nmopi[h])])
        eb_occ_data = sorted([(self.epsilon_b[h][i], h, i, 1)
                              for h in range(self.nirrep)
                              for i in range(self.nbetapi[h])])
        eb_vir_data = sorted([(self.epsilon_b[h][i], h, i, 0)
                              for h in range(self.nirrep)
                              for i in range(self.nbetapi[h], self.nmopi[h])])

        y_aocc = [e for (e, _, _, _) in ea_occ_data]
        y_avir = [e for (e, _, _, _) in ea_vir_data]
        y_bocc = [e for (e, _, _, _) in eb_occ_data]
        y_bvir = [e for (e, _, _, _) in eb_vir_data]

        self.restricted = True
        for (ea, eb) in zip(y_aocc, y_bocc):
            if abs(ea - eb) > 0.0001:
                self.restricted = False
        for (ea, eb) in zip(y_avir, y_bvir):
            if abs(ea - eb) > 0.0001:
                self.restricted = False

        a_spin = '' if self.restricted else 'a'
        aocc_labels = [
            f'MO{k + 1}{a_spin} {i+1}{self.irrep_labels[h]} (e = {e:.3f} Eh)'
            for (k, (e, h, i, occ)) in enumerate(ea_occ_data)
        ]
        naocc = len(aocc_labels)
        avir_labels = [
            f'MO{k + 1 + naocc}{a_spin} {i+1}{self.irrep_labels[h]} (e = {e:.3f} Eh)'
            for (k, (e, h, i, occ)) in enumerate(ea_vir_data)
        ]

        b_spin = '' if self.restricted else 'b'
        bocc_labels = [
            f'MO{k + 1}{b_spin} {i+1}{self.irrep_labels[h]} (e = {e:.3f} Eh)'
            for (k, (e, h, i, occ)) in enumerate(eb_occ_data)
        ]
        nbocc = len(bocc_labels)
        bvir_labels = [
            f'MO{k + 1 + naocc}{b_spin} {i+1}{self.irrep_labels[h]} (e = {e:.3f} Eh)'
            for (k, (e, h, i, occ)) in enumerate(eb_vir_data)
        ]

        self.fig = go.Figure()

        if self.plot_range == None:
            min_e = min(y_aocc)
            max_e = max(y_avir)
            e_padding = 0.05 * (max_e - min_e)
            y_min = min_e - e_padding
            y_max = max_e + e_padding
            self.fig.update_yaxes(range=[y_min, y_max])
            y_range = y_max - y_min
        else:
            self.fig.update_yaxes(range=self.plot_range)
            y_range = self.plot_range[1] - self.plot_range[0]

        if self.restricted:
            self.width = 350 if self.width == None else self.width
            self.fig.update_xaxes(range=[0.0, 1.0])
            spacing = 0.02
            x_occ_levels = self.__compute_level_position(ea_occ_data, 0.5)
            x_vir_levels = self.__compute_level_position(ea_vir_data, 0.5)
            x_aocc = self.__compute_level_position(ea_occ_data, 0.5 - spacing)
            x_bocc = self.__compute_level_position(eb_occ_data, 0.5 + spacing)
            self.__draw_levels(self.fig, x_occ_levels, y_aocc, aocc_labels,
                               self.occupied_color)
            self.__draw_levels(self.fig, x_vir_levels, y_avir, avir_labels,
                               self.unoccupied_color)
            self.__draw_electrons(self.fig, x_aocc, y_aocc, True, self.height,
                                  y_range)
            self.__draw_electrons(self.fig, x_bocc, y_bocc, False, self.height,
                                  y_range)
        else:
            self.width = 600 if self.width == None else self.width
            self.fig.update_xaxes(range=[0.0, 2.0])
            x_aocc = self.__compute_level_position(ea_occ_data, 0.5)
            x_avir = self.__compute_level_position(ea_vir_data, 0.5)
            x_bocc = self.__compute_level_position(eb_occ_data, 1.5)
            x_bvir = self.__compute_level_position(eb_vir_data, 1.5)
            self.__draw_levels(self.fig, x_aocc, y_aocc, aocc_labels,
                               self.occupied_color)
            self.__draw_levels(self.fig, x_avir, y_avir, avir_labels,
                               self.unoccupied_color)
            self.__draw_levels(self.fig, x_bocc, y_bocc, bocc_labels,
                               self.occupied_color)
            self.__draw_levels(self.fig, x_bvir, y_bvir, bvir_labels,
                               self.unoccupied_color)
            self.__draw_electrons(self.fig, x_aocc, y_aocc, True, self.height,
                                  y_range)
            self.__draw_electrons(self.fig, x_bocc, y_bocc, False, self.height,
                                  y_range)

        self.fig.update_xaxes(visible=False)
        self.fig.update_yaxes(title_text='Energy (Eh)',
                              gridcolor='rgba(200, 200, 200, 1.0)',
                              showline=True,
                              linewidth=1,
                              linecolor='black')
        self.fig.update_layout(title=self.title,
                               autosize=False,
                               width=self.width,
                               height=self.height,
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False,
                               hoverlabel_align='right',
                               yaxis_zeroline=False,
                               xaxis_zeroline=False)
        self.fig.show()

    def __compute_level_position(self,
                                 data,
                                 pos_offset,
                                 spacing=0.3,
                                 degeneracy_threshold=0.005):
        #find groups of degenerate orbitals
        groups = []
        last_group = []
        for e, h, i, occ in data:
            if len(last_group) == 0:
                last_group.append(e)
            else:
                if abs(e - last_group[0]) < degeneracy_threshold:
                    last_group.append(e)
                else:
                    groups.append(last_group)
                    last_group = [e]
        if len(last_group) > 0:
            groups.append(last_group)

        # now assign the position of the orbitals
        # orbitals are centered around the pos_offset and spaced according to spacing
        x = []
        for group in groups:
            shift = pos_offset - spacing * float(len(group) - 1) / 2.0
            for i in range(len(group)):
                x.append(shift + spacing * i)
        return x

    def __draw_levels(self, fig, x, y, labels, color):
        levels = go.Scatter(x=x,
                            y=y,
                            name='',
                            mode='markers',
                            marker_symbol=41,
                            marker_line_width=2,
                            marker_size=18,
                            hoverinfo='text',
                            hovertext=labels,
                            hoverlabel_align='right',
                            marker_color=color,
                            marker_line_color=color)
        fig.add_trace(levels)

    def __draw_electrons(self, fig, x, y, alpha, height, y_range):
        if self.style == 'arrows':
            dots = go.Scatter(x=x,
                              y=y,
                              name='',
                              mode='markers',
                              marker_line_width=2,
                              marker_size=10,
                              marker_symbol=42,
                              hoverinfo='none',
                              marker_color='rgba(255, 0, 0, 1.0)')
            fig.add_trace(dots)

            scale_factor = y_range * 600 / (3.5 * height)
            arrow_shift = 0.05 if alpha else -0.05
            yarrow = [yel + scale_factor * arrow_shift for yel in y]
            arrow = go.Scatter(x=x,
                               y=yarrow,
                               name='',
                               mode='markers',
                               marker_line_width=0,
                               marker_size=6,
                               marker_symbol=5 if alpha else 6,
                               hoverinfo='none',
                               marker_color='rgba(0, 0, 0, 1.0)')
            fig.add_trace(arrow)
        else:
            dots = go.Scatter(x=x,
                              y=y,
                              name='',
                              mode='markers',
                              marker_line_width=0,
                              marker_size=6,
                              hoverinfo='none',
                              marker_color='rgba(0, 0, 0, 1.0)')
            fig.add_trace(dots)
