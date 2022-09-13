#
# @BEGIN LICENSE
#
# Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
# that implements a variety of quantum chemistry methods for strongly
# correlated electrons.
#
# Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import logging
import psi4
import forte

logging_depth = 0


def clean_options():
    """
    A function to clear the options object

    This function does the following
    1. clears the psi4 and Forte options object
    2. re-registers all of Forte options (in their default values)
    3. pushes the value of Forte options to the psi4 options object
    """
    # clear options and allocate a fresh forte.forte_options object
    psi4.core.clean_options()
    forte.forte_options = forte.ForteOptions()

    # register options defined in Forte in the forte_options object
    forte.register_forte_options(forte.forte_options)

    # push the options defined in forte_options to psi
    psi_options = psi4.core.get_options()
    psi_options.set_current_module('FORTE')
    forte.forte_options.push_options_to_psi4(psi_options)


class ForteManager(object):
    """Singleton class to handle startup and cleanup of forte (mostly ambit)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = True
            # Put any initialization here.
            my_proc, n_nodes = forte.startup()
        return cls._instance

    def __del__(cls):
        forte.cleanup()


def start_logging(filename='forte.log', level=logging.DEBUG):
    """
    This function sets the output of logs to ``filename`` (default = forte.log)
    and sets the log level to all information.

    Parameters
    ----------
    filename: str
        the name of the log file (default = 'forte.log')
    level: {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}
        the level of severity of the events tracked (default = logging.DEBUG, which means track everything)
    """
    logging.basicConfig(filename=filename, level=level, format='# %(asctime)s | %(levelname)s | %(message)s')
    logging.info('Starting the Forte logger')


def flog(level, msg):
    """
    Log the message ``msg`` with logging level ``level``.

    ``level`` should be chosen in this way:
    debug: for detailed ouput mostly for diagnostic purpose
    info: for information produced during normal operation of the program
    warning: for a warning regarding a particular runtime event
    error: for an error that does not raise an exception

    Parameters
    ----------
    level: str
        the level of the message logged. Can be any of ('debug','info','warning','error')
    msg: str
        the text to be logged

    This function calls the logging module and puts some spaces before
    the message that we want to log.
    """
    global logging_depth
    level = level.lower()
    spaces = ' ' * max((logging_depth - 1), 0) * 2
    s = f"{spaces}{msg}"
    if level == 'info':
        logging.info(s)
    elif level == 'warning':
        logging.warning(s)
    elif level == 'debug':
        logging.debug(s)
    elif level == 'error':
        logging.error(s)
    else:
        raise ValueError(f'forte.core.flog was called with an unrecognized level ({level})')


def increase_log_depth(func):
    """
    This is a decorator used to increase the depth of forte's log.

    It is used to decorate the run() function of solvers:
        @increase_log_dept
            def run(self):
                ...
    """
    def wrapper(*args, **kwargs):
        global logging_depth
        logging_depth += 1
        func(*args, **kwargs)
        logging_depth += -1

    return wrapper
