from joblib import Parallel
from pydrive.apiattr import ApiAttributeMixin
from pydrive.files import GoogleDriveFile
from pydrive.files import GoogleDriveFileList
from pydrive.auth import LoadAuth
from pydrive.drive import GoogleDrive
from tqdm.contrib.concurrent import process_map

class BoostedGoogleDriveFileList(GoogleDriveFileList):
    def __init__(self, auth=None, param=None,
                 max_workers=128, chunksize=100):
        super().__init__(auth, param)
        self.max_workers = max_workers
        self.chunksize = chunksize
    
    @LoadAuth
    def _GetList(self):
        self.metadata = self.auth.service.metadata.files().list(**dict(self)).execute(http=self.http)
        result = []
        def get_tmp_file(file_metadata):
            tmp_file = GoogleDriveFile(
                auth=self.auth,
                metadata=file_metadata,
                uploaded=True
            )
            return tmp_file
        if (len(result) > 128):
            result = process_map(fn=get_tmp_file,
                                iterables=self.metadata["items"],
                                max_worker=self.max_workers,
                                chunksize=self.chunksize
                                )
        else:
            for file_metadata in self.metadata["items"]:
                result.append(get_tmp_file(file_metadata))
        return result



class BoostedGoogleDrive(GoogleDrive):
    def __init__(self, auth=None):
        super().__init__(auth)
    
    def ListFile(self, param=None, *args, **kwargs):
        return BoostedGoogleDriveFileList(auth=self.auth, param=param, *args, **kwargs)

