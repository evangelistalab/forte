from pyscf import gto, scf, mcscf

molecule ="""
  F
  H 1 1.5
  """
mol = gto.M(atom = molecule, basis = 'cc-pvdz', symmetry = 'c2v')
mf = scf.RHF(mol)
mf.kernel()
mc = mcscf.CASSCF(mf, ncas=2, nelecas=(1,1))
mc.mc2step()
