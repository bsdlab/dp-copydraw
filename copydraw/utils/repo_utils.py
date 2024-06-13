import re

from git import Repo
from copydraw.utils.config_loading import load_paradigm_config


class ThisRepo():
    """
    A repo class collecting some information about the repo status
    to be used for logging information
    """

    def __init__(self):
        paths = load_paradigm_config()
        self.repo = Repo(paths['SCRIPT_ROOT'])

    def get_curr_status(self):
        """ Get current repo version and not committed diffs
        for logging purpose

        Returns
        -------
        state_and_diff : str
            current state and description of non commited diffs

        """

        changed_files = re.findall(
            r'modified:\s+([^\n]*)',
            self.repo.git.status()
        )

        if changed_files != []:
            cfiles = "\n".join(changed_files)
            changed_files_str = f"Changes in files: {cfiles}"
        elif not self.repo.is_dirty():
            changed_files_str = "Repo is clean"
        else:
            changed_files_str = "Repo is dirty!"

        diff = self.repo.git.diff()

        git_info = [" git ".center(80, "="),
                    f"{self.repo.head.commit}",
                    changed_files_str,
                    f"Git diff:\n{diff}",
                    " git end ".center(80, "=")
                    ]

        return "\n".join(git_info)

    def get_commit_hexsha(self):
        return self.repo.head.object.hexsha

    def status(self):
        return self.repo.git.status()

