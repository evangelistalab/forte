from pathlib import Path
import pytest

from forte.core import flog, start_logging


def test_core():
    """Test logging and failure to set logging level."""
    filename = Path(__file__).parent / 'forte.log'
    start_logging(filename)
    flog('INFO', 'testing logging')

    # check the last message logged in
    with open(filename) as f:
        line = f.readlines()[-1].split('|')
        level = line[-2].strip()
        msg = line[-1].strip()
        assert level == 'INFO'
        assert msg == 'testing logging'

    with pytest.raises(ValueError):
        flog('alert', 'something went wrong!')


if __name__ == "__main__":
    test_core()
