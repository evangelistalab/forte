# import forte
from typing import List
from .module import Module
from forte.data import ForteData
from forte._forte import to_state_nroots_map, make_active_space_solver, make_mcscf_two_step, MOSpaceInfo, make_mo_space_info_from_map


class MCSCF(Module):

    """
    A module to perform MCSCF calculations.
    """

    def __init__(self, solver_type: str = "FCI"):
        """
        Parameters
        ----------
        solver_type: str
            The type of the active space solver.
        """
        super().__init__()
        self.solver_type = solver_type
        self.freeze_core = False

    def _run(self, data: ForteData) -> ForteData:
        state_map = to_state_nroots_map(data.state_weights_map)

        mcscf_mo_space_info = self.make_mcscf_mo_space_info(data)
        
        data.active_space_solver = make_active_space_solver(
            self.solver_type, state_map, data.scf_info, mcscf_mo_space_info, data.options
        )
        casscf = make_mcscf_two_step(
            data.active_space_solver, data.state_weights_map, data.scf_info, data.options, mcscf_mo_space_info, data.ints
        )
        energy = casscf.compute_energy()
        data.results.add("energy", energy, "MCSCF energy", "hartree")

        return data


    def make_mcscf_mo_space_info(self, data: ForteData) -> MOSpaceInfo:
        """This function prepares a MOSpaceInfo info object for a MCSCF computation
        """
        if self.freeze_core:
            return data.mo_space_info

        # build a dictionary of the MO spaces
        mo_space_dict = {}
        for space in data.mo_space_info.space_names():
            mo_space_dict[space] = list(data.mo_space_info.dimension(space).to_tuple())

        # combine the frozen and restricted spaces 
        docc = [ x + y for x, y in zip(mo_space_dict['FROZEN_DOCC'], mo_space_dict['RESTRICTED_DOCC'])]
        uocc = [ x + y for x, y in zip(mo_space_dict['FROZEN_UOCC'], mo_space_dict['RESTRICTED_UOCC'])]

        # remove the frozen spaces and add the restricted spaces
        mo_space_dict.pop('FROZEN_DOCC')
        mo_space_dict.pop('FROZEN_UOCC')
        mo_space_dict['RESTRICTED_DOCC'] = docc
        mo_space_dict['RESTRICTED_UOCC'] = uocc

        # build a MOSpaceInfo object with frozen MOs merged with the restricted MOs
        nmopi = data.scf_info.nmopi()
        point_group = data.mo_space_info.point_group_label()
        mcscf_mo_info = make_mo_space_info_from_map(nmopi, point_group, mo_space_dict, data.mo_space_info.reorder())

        return mcscf_mo_info