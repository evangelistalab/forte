#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ipywidgets as widgets
import re
from .py3js_renderer import Py3JSRenderer

def cube_viewer(cubes,
                scale=1.0,
                font_size=16,
                font_family='Helvetica',
                width=400,
                height=400,
                show_text=True):
    """
    A simple widget for viewing cube files

    Parameters
    ----------
    cubes : dict
        a dictionary of CubeFile objects
    scale : float
        the scale factor used to make a molecule smaller or bigger (default = 1.0)
    font_size : int
        the font size (default = 16)
    font_family : str
        the font used to label the orbitals (default = Helvetica)
    width : int
        the width of the plot in pixels (default = 400)
    height : int
        the height of the plot in pixels (default = 400)
    show_text : bool
        show the name of the cube file under the plot? (default = True)
    """
    # convert cube file names into human readable text
    labels_to_filename = {}
    psi4_mo_label_re = r'Psi_([a|b])_(\d+)_(\d+)-([\w\d]*)\.cube'
    psi4_density_label_re = r'D(\w)\.cube'
    for k in cubes.keys():
        m1 = re.match(psi4_mo_label_re, k)
        m2 = re.match(psi4_density_label_re, k)
        if m1:
            label = f'MO {int(m1.groups()[1]):4d}{m1.groups()[0]} ({m1.groups()[2]}-{m1.groups()[3]})'
        elif m2:
            if m2.groups()[0] == 'a':
                label = 'Density (alpha)'
            if m2.groups()[0] == 'b':
                label = 'Density (beta)'
            if m2.groups()[0] == 's':
                label = 'Density (spin)'
            if m2.groups()[0] == 't':
                label = 'Density (total)'
        else:
            label = k
        labels_to_filename[label] = k
    sorted_labels = sorted(labels_to_filename.keys())

    box_layout = widgets.Layout(border='0px solid black',
                                width=f'{width + 50}px',
                                height=f'{height + 100}px')

    def f(label, cubes, labels_to_filename):
        filename = labels_to_filename[label]
        cube = cubes[filename]
        renderer = Py3JSRenderer(width=width, height=height)
        type = 'mo'
        if label[0] == 'D':
            type = 'density'
        renderer.add_cubefile(cube, scale=scale, type=type)
        style = f'font-size:{font_size}px;font-family:{font_family};font-weight: bold;'
        mo_label = widgets.HTML(
            value=f'<div align="center" style="{style}">{label}</div>')
        file_label = widgets.HTML(
            value=f'<div align="center">({filename})</div>')
        display(
            widgets.VBox([mo_label, renderer.renderer, file_label],
                         layout=box_layout))

    ws = widgets.Select(options=sorted_labels, description='Cube files:')
    interactive_widget = widgets.interactive(
        f,
        label=ws,
        cubes=widgets.fixed(cubes),
        labels_to_filename=widgets.fixed(labels_to_filename))
    output = interactive_widget.children[-1]
    output.layout.height = f'{height + 100}px'
    style = """
    <style>
       .jupyter-widgets-output-area .output_scroll {
            height: unset !important;
            border-radius: unset !important;
            -webkit-box-shadow: unset !important;
            box-shadow: unset !important;
        }
        .jupyter-widgets-output-area  {
            height: auto !important;
        }
    </style>
    """
    display(widgets.HTML(style))
    display(interactive_widget)
