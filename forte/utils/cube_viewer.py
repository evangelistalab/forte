#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
import importlib

import ipywidgets as widgets
from .py3js_renderer import Py3JSRenderer

found_forte = importlib.util.find_spec("forte") is not None
if found_forte:
    import forte
else:
    from .cube_file import *


def list_cubes(path='.'):
    """
    List all the cubefiles (suffix ".cube" ) in a given path

    Parameters
    ----------
    path : str
        The path of the directory that will contain the cube files
    """

    import os
    cube_files = []
    isdir = os.path.isdir(path)
    if isdir:
        for file in os.listdir(path):
            if file.endswith('.cube'):
                cube_files.append(os.path.join(path, file))
        if len(cube_files) == 0:
            print(f'load_cubes: no cube files found in directory {path}')
    else:
        print(f'load_cubes: directory {path} does not exist')

    return cube_files


class CubeLoader():
    """
    A simple class to load and cache cube files

    Parameters
    ----------
    path : str
        The path used to load cube files (default = '.')
    cubes : list
        List of cube files to be plotted
    width : int
        the width of the plot in pixels (default = 400)
    height : int
        the height of the plot in pixels (default = 400)
    scale : float
        the scale factor used to make a molecule smaller or bigger (default = 1.0)
    font_size : int
        the font size (default = 16)
    font_family : str
        the font used to label the orbitals (default = Helvetica)
    show_text : bool
        show the name of the cube file under the plot? (default = True)
    """
    def __init__(self, cubes=None):
        self.cached_cubes = {}
        self.debug = False

    def load(self, filename):
        # if the file is cached return it
        if filename in self.cached_cubes:
            cf = self.cached_cubes[filename]
        # otherwise load it
        else:
            start_time = time.perf_counter()

            if found_forte:
                cf = forte.CubeFile(filename)
            else:
                cf = CubeFile(filename)

            self.cached_cubes[filename] = cf

            end_time = time.perf_counter()
            if self.debug:
                print(f'Time to read cube file: {end_time-start_time}')
        return {filename: cf}


class CubeViewer():
    """
    A simple widget for viewing cube files. Cube files are loaded from the current
    directory. Alternatively, the user can pass a path or a dictionary containing CubeFile objects

    Parameters
    ----------
    path : str
        The path used to load cube files (default = '.')
    cubes : list
        List of cube files to be plotted
    width : int
        the width of the plot in pixels (default = 400)
    height : int
        the height of the plot in pixels (default = 400)
    scale : float
        the scale factor used to make a molecule smaller or bigger (default = 1.0)
    font_size : int
        the font size (default = 16)
    font_family : str
        the font used to label the orbitals (default = Helvetica)
    show_text : bool
        show the name of the cube file under the plot? (default = True)
    """
    def __init__(self,
                 cubes=None,
                 path='.',
                 width=400,
                 height=400,
                 font_size=16,
                 font_family='Helvetica',
                 levels=None,
                 colors=None,
                 colorscheme='emory',
                 opacity=1.0,
                 sumlevel=0.85,
                 show_text=True):

        start_time = time.perf_counter()

        if cubes == None:
            print(f'CubeViewer: listing cube files from the directory {path}')
            cubes = list_cubes(path)

        if (len(cubes) == 0):
            print(
                f'CubeViewer: no cube files provided. The widget will not be displayed'
            )
            return

        self.debug = False
        self.cubes = cubes
        self.path = path
        self.width = width
        self.height = height
        self.scale = 1.0
        self.font_size = font_size
        self.font_family = font_family
        self.show_text = show_text

        labels_to_filename, sorted_labels = self.parse_cube_files()

        box_layout = widgets.Layout(border='0px solid black',
                                    width=f'{width + 50}px',
                                    height=f'{height + 100}px')

        # start a CubeLoader
        cube_loader = CubeLoader(cubes=cubes)
        # start a Py3JSRenderer
        renderer = Py3JSRenderer(width=width, height=height)

        make_objs_time = time.perf_counter()

        if self.debug:
            print(f'Time to make objects: {make_objs_time-start_time}')

        make_meshes_time = time.perf_counter()

        print(
            f'Reading {len(sorted_labels)} cube file{"s" if len(sorted_labels) > 1 else ""}'
        )

        first = True
        for label in sorted_labels:
            filename = labels_to_filename[label]
            cube = cube_loader.load(filename)
            type = 'density' if label[0] == 'D' else 'mo'
            renderer.add_cubefiles(cube,
                                   type=type,
                                   colorscheme=colorscheme,
                                   levels=levels,
                                   colors=colors,
                                   opacity=opacity,
                                   sumlevel=sumlevel,
                                   add_geom=first)
            if first: first = False

        style = f'font-size:{font_size}px;font-family:{font_family};font-weight: bold;'
        mo_label = widgets.HTML()
        file_label = widgets.HTML()

        def update_renderer(label, objects):
            """This function updates the rendeder once the user has selected a new cube file to plot"""
            start_update_time = time.perf_counter()

            renderer, style, labels_to_filename = objects

            filename = labels_to_filename[label]

            # update the renderer
            renderer.set_active_cubes(filename)

            # update the labels
            file_label.value = f'<div align="center">({filename})</div>'
            mo_label.value = f'<div align="center" style="{style}">{label}</div>'

            end_update_time = time.perf_counter()
            if self.debug:
                print(
                    f'Time to update objects ({label}): {end_update_time-start_update_time}'
                )

        ws = widgets.Select(options=sorted_labels, description='Cube files:')
        interactive_widget = widgets.interactive(update_renderer,
                                                 label=ws,
                                                 objects=widgets.fixed(
                                                     (renderer, style,
                                                      labels_to_filename)))

        output = interactive_widget.children[-1]
        # output.layout.height = f'{height + 100}px'
        widget_style = """
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

        # display the rendered and the labels
        display(
            widgets.VBox([mo_label, renderer.renderer, file_label],
                         layout=box_layout))

        # disable scroll and display the selection widget
        display(widgets.HTML(widget_style))
        display(interactive_widget)
        display_time = time.perf_counter()
        if self.debug:
            print(f'Time to prepare renderer: {display_time-start_time}')

    def parse_cube_files(self):
        # convert cube file names into human readable text
        labels_to_filename = {}
        # parser for MO cube files generated by psi4
        psi4_mo_label_re = r'.*Psi_([a|b])_(\d+)_(\d+)-([\w\d\'\"]*)\.cube'
        # parser for MO density files generated by psi4
        psi4_density_label_re = r'.*D(\w)\.cube'
        for key in self.cubes:
            m1 = re.match(psi4_mo_label_re, key)
            m2 = re.match(psi4_density_label_re, key)
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
                label = key
            labels_to_filename[label] = key
        sorted_labels = sorted(labels_to_filename.keys())
        return (labels_to_filename, sorted_labels)
