from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
from ..binary import save_pickle
from ..text import save_json

gauth = GoogleAuth(settings_file="/export/home/kningtg/.gdrive/settings.yaml")
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)
TEAM_DRIVE_ID = "0AEgpDWBrB9EKUk9PVA"

def get_file_id(file_path, root=TEAM_DRIVE_ID):
    # ipdb.set_trace()
    if file_path[-1] == "/":  # delete / at the end for standard format
        file_path = file_path[:-1]

    file_list = [{'title': '','id': root}]
    filename_list = file_path.split("/")
    for name in filename_list:
        file_id = None
        for filename in file_list:
            if filename["title"] == name:
                file_id = filename["id"]
                file_list = list_file_by_id(file_id)
                break

        if not file_id:
            raise Exception(f"{name} File Not Found")
    
    return file_id
    

def list_file_by_id(file_id, drive_id=TEAM_DRIVE_ID):
    file_list =  drive.ListFile({'q':f"'{file_id}' in parents and trashed=false",
                                'corpora': 'teamDrive',
                                'teamDriveId': f'{drive_id}',
                                'includeTeamDriveItems': True,
                                'supportsTeamDrives': True
                                }).GetList()
    # import ipdb; ipdb.set_trace()
    file_list = [{"title": x["title"], "id": x["id"]} for x in file_list]
    # print(file_list[:10])
    return file_list

def list_file_by_path(file_path, drive_id=TEAM_DRIVE_ID):
    file_id = get_file_id(file_path)
    file_list = list_file_by_id(file_id)
    return file_list


    