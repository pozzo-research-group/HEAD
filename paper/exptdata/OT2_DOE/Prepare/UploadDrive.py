from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
# gauth.LoadCredentialsFile(path_to_credentials) 
drive = GoogleDrive(gauth)
def initialize_connection(path_to_credentials): # would need to make these globally availble or return the drive 
    gauth = GoogleAuth()
    # gauth.LocalWebserverAuth()
    gauth.LoadCredentialsFile(path_to_credentials) 
    drive = GoogleDrive(gauth)

def team_drive_dict(path_to_credentials): # for now lets keep this static, will prompt to login for the team drive. 
    """team_drive_id must be formatted with single quotations in the string, with the string datatype coming from double 
    quotation marks i.e. "'team_drive_id'" """ 
    gauth = GoogleAuth()
    # gauth.LocalWebserverAuth()
    gauth.LoadCredentialsFile(path_to_credentials) 
    drive = GoogleDrive(gauth)
    team_drive_folder_list = drive.ListFile({'q':"'0AHSxuxDy84zYUk9PVA' in parents and trashed=false", 
                                'corpora': 'teamDrive', 
                                'teamDriveId': '0AHSxuxDy84zYUk9PVA', 
                                'includeTeamDriveItems': True, 
                                'supportsTeamDrives': True}).GetList()

    team_drive_id_dict = {}    
    for file in team_drive_folder_list: # hmm the fact that this is static ID we can make into dictioanry
        team_drive_id_dict[file['title']] =  file['id']
    
    return team_drive_id_dict

def file_and_folder_navi(folder_id): # for now lets keep this static, will prompt to login for the team drive. 
    folder_id = '"' + folder_id  + '"'
    
    drive_list = drive.ListFile({'q':folder_id + " in parents and trashed=false", 
                                'corpora': 'teamDrive', 
                                'teamDriveId': '0AHSxuxDy84zYUk9PVA', 
                                'includeTeamDriveItems': True, 
                                'supportsTeamDrives': True}).GetList()

    drive_dict = {}    
    for file in drive_list: # hmm the fact that this is static ID we can make into dictioanry
        drive_dict[file['title']] =  file['id']
    
    return drive_dict

def upload_to_team_drive_folder(folder_id, file_path, file_name):    
    # dont set title or mimetype in order to make it default to actual files name and dtype
    f = drive.CreateFile({
        'title': file_name,
        'parents': [{
            'kind': 'drive#fileLink',
            'teamDriveId': '0AHSxuxDy84zYUk9PVA',
            'id': folder_id
        }]
    })
    f.SetContentFile(file_path)

    f.Upload(param={'supportsTeamDrives': True})
