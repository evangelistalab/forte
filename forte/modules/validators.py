from enum import Enum, auto
from forte.data import ForteData


class Feature(Enum):
    """
    This enum class is used to store all possible Features needed by a module
    """

    OPTIONS = "options"
    STATE_WEIGHTS_MAP = "state_weights_map"
    SYMMETRY = "symmetry"
    MOLECULE = "molecule"
    MODEL = "model"
    SCF_INFO = "scf_info"
    MO_SPACE_INFO = "mo_space_info"
    INTS = "ints"
    RESULTS = "results"
    AS_INTS = "as_ints"
    # options = auto()
    # state_weights_map = auto()
    # symmetry = auto()
    # molecule = auto()
    # model = auto()
    # scf_info = auto()
    # mo_space_info = auto()
    # ints = auto()
    # as_ints = auto()
    # psi_wfn = auto()
    # active_space_solver = auto()
    # rdms = auto()
    # results = auto()


def module_validation(needs=None):
    needs = needs or []

    def decorator(func):
        def wrapper(self, data: ForteData):
            # Validate inputs
            missing_fields = [field.value for field in needs if getattr(data, field.value, None) is None]
            if missing_fields:
                missing_feature_error = f"\nThe module {self.__class__.__name__} requires the following missing input data: {missing_fields}"
                missing_feature_error += "\n\nDefined data fields:"
                field_names = data.__dataclass_fields__.keys()
                for field_name in field_names:
                    value = getattr(data, field_name)
                    if value is not None:
                        missing_feature_error += f"\n - {field_name}: {value}"
                raise AssertionError(missing_feature_error)

            result = func(self, data)

            return result

        return wrapper

    return decorator
