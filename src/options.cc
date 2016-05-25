
void forte_options(std::string name, Options &options)
{
    if (name == "FORTE" || options.read_globals()) {

    /*- The basis used to define an orbital subspace -*/
    options.add_str("MIN_BASIS","STO-3G");

    /*- Selects a subspace of atomic orbitals
    *
    *  Syntax: ["<element1><range1><ao set1>","<element2><range2><ao set2>",...]
    *
    *  Each list entry of the form "<element><range><ao set>" specifies a set
    *  of atomic orbitals for atoms of a give type.
    *
    *  <element> - the symbol of the element, e.g. 'Fe', 'C'
    *
    *  <range>   - the range of the atoms selected.  Possible choices are:
    *              1) '' (empty): all atoms that match <element> are selected
    *              2) 'i'       : select the i-th atom of type <element>
    *              3) 'i-j'     : select atoms i through j (included) of type <element>
    *
    *  <ao set>  - the set of atomic orbitals to select.  Possible choices are:
 *              1) '' (empty): select all basis functions
 *              2) '(nl)'    : select the n-th level with angular momentum l
 *                             e.g. '(1s)', '(2s)', '(2p)',...
 *                             n = 1, 2, 3, ...
 *                             l = 's', 'p', 'd', 'f', 'g', ...
 *              3) '(nlm)'   : select the n-th level with angular momentum l and component m
 *                             e.g. '(2pz)', '(3dzz)', '(3dxx-yy)'
 *                             n = 1, 2, 3, ...
 *                             l = 's', 'p', 'd', 'f', 'g', ...
 *                             m = 'x', 'y', 'z', 'xy', 'xz', 'yz', 'zz', 'xx-yy'
 *
 *  Valid options include:
 *
 *  ["C"] - all carbon atoms
 *  ["C","N"] - all carbon and nitrogen atoms
 *  ["C1"] - carbon atom #1
 *  ["C1-3"] - carbon atoms #1, #2, #3
 *  ["C(2p)"] - the 2p subset of all carbon atoms
 *  ["C(1s,2s)"] - the 1s/2s subsets of all carbon atoms
 *  ["C1-3(2s)"] - the 2s subsets of carbon atoms #1, #2, #3
 *
 * -*/
    options.add("SUBSPACE",new ArrayType());
    }
}
