def parse_state(state):
    # throw error if state is not a dictionary
    assert isinstance(state, dict)
    # throw error if state does not contain the required keys
    assert "charge" in state
    assert "multiplicity" in state
    assert "sym" in state
    # return the values of the keys
    return state["charge"], state["multiplicity"], state["sym"]


def parse_active_space(active_space):
    """Accepts inputs of the form {"norb": 2, "nel": 2}"""
    # throw error if active_space is not a dictionary
    assert isinstance(active_space, dict)
    # throw error if active_space does not contain the required keys
    assert "norb" in active_space
    assert "nel" in active_space
    # return the values of the keys
    return active_space["norb"], active_space["nel"]
