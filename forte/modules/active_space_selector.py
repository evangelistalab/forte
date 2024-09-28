from typing import List, Union, Dict
from .module import Module
from forte.data import ForteData
from forte.modules.validators import Feature, module_validation
from forte._forte import make_mo_space_info_from_map


class ActiveSpaceSelector(Module):
    """
    A module to prepare an ActiveSpaceIntegral
    """

    # accept as input a dictionary of string,int list pairs or a list of integers
    def __init__(self, active_space: Union[Dict[str, List[int]], List[int]] = None):
        """
        Parameters
        ----------
        active_space : dict
            The active space to be used for the calculation
        """
        super().__init__()
        # assert isinstance(active_space, dict) or isinstance(active_space, list)

        self.active_space = active_space

    @module_validation(needs=[Feature.SCF_INFO])
    def _run(self, data: ForteData) -> ForteData:
        """
        Selects the active space based on the input
        """

        # in the case where we have no active space selection, we just assemble this object from the options
        self.active_space = self.active_space or self.make_dict_from_data(data)

        # select the active space using the information in the active_space dictionary
        if isinstance(self.active_space, dict):
            if "nel" in self.active_space and "norb" in self.active_space:
                self.select_norb_nel(data, self.active_space["nel"], self.active_space["norb"])
            elif "nel" in self.active_space and "active_orbitals" in self.active_space:
                self.select_active_orbitals_nel(data, self.active_space["nel"], self.active_space["active_orbitals"])
            elif "active" in self.active_space:
                self.select_mos_active(data, self.active_space)
            else:
                raise ValueError("Invalid active space selection")
        else:
            raise ValueError("Invalid active space selection is not a dictionary")
        return data

    def make_dict_from_data(self, data: ForteData) -> Dict:
        options = data.options
        active_space = {}
        if options.get_int("NACT_EL") and options.get_int("NACT_ORB"):
            active_space["nel"] = options.get_int("NACT_EL")
            active_space["norb"] = options.get_int("NACT_ORB")
        elif options.get_int("NACT_EL") and options.get("ACTIVE_ORBITALS"):
            active_space["nel"] = options.get_int("NACT_EL")
            active_space["active_orbitals"] = options.get("ACTIVE_ORBITALS")
        elif options.get_int("ACTIVE"):
            active_space["active"] = options.get_int("ACTIVE")
        return active_space

    def select_norb_nel(self, data, nel, norb):
        """
        Selects the active space based on the number of electrons and orbitals
        """
        # figure out how many electrons we have
        if len(data.state_weights_map) != 1:
            raise ValueError("Active space selection based on (nel,norb) requires a single state")

        state = list(data.state_weights_map.items())[0][0]
        na = state.na()
        nb = state.nb()
        nel_restricted = na + nb - nel  # number of electrons in the active space
        # check that we have an even number of restricted docc orbitals
        if nel_restricted % 2 != 0:
            raise ValueError("The number of restricted doubly occupied orbitals must be even")
        num_restricted_docc = nel_restricted // 2

        # selection based on the alpha orbital energies
        epsilon = data.scf_info.epsilon_a().nph
        nirrep = len(epsilon)
        # sort the orbitals by energy
        orbital_energies = []
        for h, eps_h in enumerate(epsilon):
            for k, eps in enumerate(eps_h):
                orbital_energies.append((eps, h, len(orbital_energies)))
        orbital_energies = sorted(orbital_energies)
        print(f"Selecting {num_restricted_docc} restricted docc orbitals")

        restricted_docc_per_irrep = [0] * nirrep
        restricted_docc_list = []
        for i in range(num_restricted_docc):
            eps, h, k = orbital_energies[i]
            restricted_docc_per_irrep[h] += 1
            restricted_docc_list.append(k)

        active_per_irrep = [0] * nirrep
        active_list = []
        for u in range(num_restricted_docc, num_restricted_docc + norb):
            eps, h, k = orbital_energies[u]
            active_per_irrep[h] += 1
            active_list.append(k)

        print(f"Restricted orbitals: {restricted_docc_list}")
        print(f"Active orbitals:     {active_list}")

        data.options.set_int_list("RESTRICTED_DOCC", restricted_docc_per_irrep)
        data.options.set_int_list("ACTIVE", active_per_irrep)

    def select_active_orbitals_nel(self, data, nel, active_orbitals_str):
        """
        Selects the active space based on the number of electrons and a list of active orbitals
        """
        # figure out how many electrons we have
        if len(data.state_weights_map) != 1:
            raise ValueError("Active space selection based on (nel,norb) requires a single state")

        import itertools

        nmopi = list(data.scf_info.nmopi().to_tuple())
        cumulative_offset = list(itertools.accumulate([0] + nmopi[:-1]))
        print(f"Number of MOs per irrep: {nmopi}")
        print(f"Cumulative offset: {cumulative_offset}")

        # parse the active orbitals
        active_orbital_indices = []
        for x in active_orbitals_str:
            if len(x.split()) != 2:
                raise ValueError("Invalid active orbital format")
            index_in_irrep, irrep_label = x.split()
            irrep = data.symmetry.irrep_label_to_index(irrep_label)
            abs_index = int(index_in_irrep) - 1
            active_orbital_indices.append((irrep, abs_index))

        state = list(data.state_weights_map.items())[0][0]
        na = state.na()
        nb = state.nb()
        nel_docc = na + nb - nel  # number of electrons in the active space
        # check that we have an even number of restricted docc orbitals
        if nel_docc % 2 != 0:
            raise ValueError("The number of restricted doubly occupied orbitals must be even")
        num_docc = nel_docc // 2

        # selection based on the alpha orbital energies
        epsilon = data.scf_info.epsilon_a().nph
        nirrep = len(epsilon)
        # sort the orbitals by energy
        nonactv_orbital_information = []
        actv_orbital_information = []
        for h, eps_h in enumerate(epsilon):
            for k, eps in enumerate(eps_h):
                if (h, k) in active_orbital_indices:
                    actv_orbital_information.append((eps, h, k))
                else:
                    nonactv_orbital_information.append((eps, h, k))

        nonactv_orbital_information = sorted(nonactv_orbital_information)
        print(f"Selecting {num_docc} restricted docc orbitals")

        docc_per_irrep = [0] * nirrep
        docc_list = []
        for eps, h, k in nonactv_orbital_information[:num_docc]:
            docc_per_irrep[h] += 1
            docc_list.append((h, k))

        active_per_irrep = [0] * nirrep
        active_list = []
        for eps, h, k in actv_orbital_information:
            active_per_irrep[h] += 1
            active_list.append((h, k))

        uocc_per_irrep = [0] * nirrep
        uocc_list = []
        for eps, h, k in nonactv_orbital_information[num_docc:]:
            uocc_per_irrep[h] += 1
            uocc_list.append((h, k))

        print(f"Restricted orbitals: {docc_list}")
        print(f"Active orbitals:     {active_list}")
        print(f"Unrestricted orbitals: {uocc_list}")

        reorder = [[] for i in nmopi]
        for h, k in docc_list:
            reorder[h].append(k)
        for h, k in active_list:
            reorder[h].append(k)
        for h, k in uocc_list:
            reorder[h].append(k)

        print(f"Reordering: {reorder}")

        data.options.set_int_list("RESTRICTED_DOCC", docc_per_irrep)
        data.options.set_int_list("ACTIVE", active_per_irrep)
        data.scf_info.reorder_orbitals(reorder)

    def select_mos_active(self, data, active_space):
        """
        Selects the active space based on the number of orbitals per irrep
        """

        def add_space(name):
            if name in active_space:
                orbs_per_irrep = active_space[name]
                data.options.set_int_list(name.upper(), orbs_per_irrep)
                print(f"{name.upper()} orbitals per irrep: {orbs_per_irrep}")

        add_space("active")
        add_space("gas1")
        add_space("gas2")
        add_space("gas3")
        add_space("gas4")
        add_space("gas5")
        add_space("gas6")
        add_space("restricted_docc")
        add_space("frozen_docc")
        add_space("restricted_uocc")
        add_space("frozen_uocc")
