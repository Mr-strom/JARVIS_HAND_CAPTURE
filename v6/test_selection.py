import pytest
from v6.launch import command

def test_default_is_original_v5():
    assert command()[1].endswith('neural_live.py')
    assert 'v6.live' not in command()

@pytest.mark.parametrize('mode', ['optimizer', 'hybrid'])
def test_optional_dispatch(mode):
    cmd = command(mode)
    assert cmd[1:3] == ['-m', 'v6.live']
    assert cmd[cmd.index('--controller') + 1] == mode

def test_reject_unknown():
    with pytest.raises(ValueError):
        command('unknown')
