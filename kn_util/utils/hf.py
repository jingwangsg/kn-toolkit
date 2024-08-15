from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitInfo, validate_hf_hub_args, future_compatible
from concurrent.futures import Future
from typing import List, Optional, Union
from pathlib import Path


class HfApiExtended(HfApi):
    @validate_hf_hub_args
    @future_compatible
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Union[str, bool, None] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        multi_commits: bool = False,
        multi_commits_verbose: bool = False,
        run_as_future: bool = False,
    ) -> Union[CommitInfo, str, Future[CommitInfo], Future[str]]:
        pass
