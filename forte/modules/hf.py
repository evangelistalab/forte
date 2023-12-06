from typing import List
from dataclasses import dataclass

from psi4 import geometry

from .module import Module
from forte.data import ForteData
from forte.core import clean_options
from forte._forte import SCFInfo


@dataclass
class HF(Module):
    """
    A module to run Hartree-Fock computation
    """

    basis: str
    restricted: bool = True
    e_convergence: float = 1.0e-10
    d_convergence: float = 1.0e-6
    docc: List[int] = None
    socc: List[int] = None
    output_file: str = "output.dat"

    def __post_init__(self):
        """
        Parameters
        ----------
        solver_type: str
            The type of the active space solver.
        """
        super().__init__()

    def _run(self, data: ForteData) -> ForteData:
        """Run a Hartree-Fock computation"""
        import psi4

        # reset psi4's options to avoid pollution
        clean_options()

        molecule = data.molecule
        if len(data.state_weights_map.keys()) != 1:
            # raise and print the content of data.state_weights_map
            raise ValueError(
                f"(HF) The state_weights_map should contain only one state. "
                f"Instead, it contains {data.state_weights_map}."
            )

        state = list(data.state_weights_map.keys())[0]
        molecule.set_molecular_charge(0)  # TODO: should be state.charge())
        molecule.set_multiplicity(state.multiplicity())

        # # prepare options for psi4
        # scf_type_dict = {
        #     "CONVENTIONAL": "PK",
        #     "STD": "PK",
        # }
        # # convert to psi4 terminology
        # int_type = self.model.int_type.upper()
        # if int_type in scf_type_dict:
        #     scf_type = scf_type_dict[int_type]
        # else:
        #     scf_type = int_type
        scf_type = "PK"

        if self.restricted:
            ref = "RHF" if state.multiplicity() == 1 else "ROHF"
        else:
            ref = "UHF"

        options = {
            "BASIS": self.basis,
            "REFERENCE": ref,
            "SCF_TYPE": scf_type,
            "E_CONVERGENCE": self.e_convergence,
            "D_CONVERGENCE": self.d_convergence,
        }

        # optionally specify docc/socc
        if self.docc is not None:
            options["DOCC"] = self.docc
        if self.socc is not None:
            options["SOCC"] = self.socc

        # if self.data.model.jkfit_aux_basis is not None:
        #     options["DF_BASIS_SCF"] = self.data.model.jkfit_aux_basis

        full_options = {**options}

        # send user options to psi4
        psi4.set_options(full_options)

        # pipe output to the file self.output_file
        psi4.core.set_output_file(self.output_file, True)

        # # pre hf callback
        # self._cbh.call("pre hf", self)

        # run scf and return the energy and a wavefunction object
        # flog("info", "HF: calling psi4.energy().")
        energy, psi_wfn = psi4.energy("scf", molecule=molecule, return_wfn=True)
        # flog("info", "HF: psi4.energy() done")

        # check symmetry
        # flog("info", "HF: checking symmetry of the HF solution")
        self.check_symmetry(data, psi_wfn)

        # add the energy to the results
        # flog("info", f"HF: hf energy = {energy}")
        data.results.add("hf energy", energy, "Hartree-Fock energy", "Eh")

        # store calculation outputs in the Data object
        data.psi_wfn = psi_wfn
        data.scf_info = SCFInfo(psi_wfn)

        # # post hf callback
        # self._cbh.call("post hf", self)

        # flog("info", "HF: calling psi4.core.clean()")
        psi4.core.clean()

        return data

    def check_symmetry(self, data, psi_wfn):
        state = list(data.state_weights_map.keys())[0]
        socc = psi_wfn.soccpi()
        sym = 0
        for h in range(socc.n()):
            if socc[h] % 2 == 1:
                sym = sym ^ h
        if state.irrep() != sym:
            symmetry = data.symmetry
            target = symmetry.irrep_label(state.irrep())
            actual = symmetry.irrep_label(sym)
            raise RuntimeError(
                f"(HF) The HF equations converged on a state with a symmetry ({actual}) different from the one requested ({target})."
                "\nPass the docc and socc options to converge to a solution with the correct symmetry."
            )
