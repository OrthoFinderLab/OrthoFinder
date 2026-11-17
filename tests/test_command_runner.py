"""utils/command_runner.py: commands run from a list or taken from an iterator as they are started."""
import pytest

from orthofinder.utils import command_runner


class LazyCommands(object):
    """As search.commands.SearchCommands: a length, and each command made when it is reached."""

    def __init__(self, commands):
        self.commands = commands
        self.made = 0

    def __len__(self):
        return len(self.commands)

    def __iter__(self):
        for c in self.commands:
            self.made += 1
            yield c


@pytest.mark.parametrize("lazy", [False, True])
def test_commands_run_and_recorded(tmp_path, lazy):
    commands = ["echo %d > %s" % (i, tmp_path / ("f%d" % i)) for i in range(25)]
    done = []
    command_runner.run_parallel_commands(3, LazyCommands(commands) if lazy else commands, q_list_of_list=False,
                                         on_success=done.append)
    assert sorted(done) == sorted(commands)
    assert all((tmp_path / ("f%d" % i)).read_text() == "%d\n" % i for i in range(25))


def test_lazy_commands_stop_after_a_failure(tmp_path):
    # (the rest are not even made once one has failed)
    commands = LazyCommands(["true"] * 5 + ["exit 3"] + ["true"] * 200)
    with pytest.raises(Exception):
        command_runner.run_parallel_commands(2, commands, q_list_of_list=False, q_print_on_error=False)
    assert commands.made < 50
