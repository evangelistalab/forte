from forte._forte import MOSpaceInfo, ForteOptions


def make_mo_spaces_from_options(options: ForteOptions) -> dict:
    """
    Create a dictionary of mo_space_info objects from the options stored in the data object
    """
    mo_spaces = {}
    for space in MOSpaceInfo.elementary_spaces + ["ACTIVE"]:
        if options.is_none(space):
            continue

        occupation = options.get_int_list(space)
        if len(occupation) != 0:
            mo_spaces[space] = occupation
    return mo_spaces
