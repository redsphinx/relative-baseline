from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from datetime import date, datetime


SPREADSHEET_ID = '1p8S73Li52kqmi9NO-eJMOjnLbkVud9jMQ7PXxoTI2jA'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
CREDENTIALS_JSON = '/home/gabras/credentials.json'
CREDENTIALS_PKL = '/home/gabras/token.pickle'
CREDS = None
SERVICE = None
VALUE_INPUT_OPTION = 'RAW'


def initialize():
    global CREDS, SERVICE

    if os.path.exists(CREDENTIALS_PKL):
        with open(CREDENTIALS_PKL, 'rb') as token:
            CREDS = pickle.load(token)
            
    if not CREDS or not CREDS.valid:
        if CREDS and CREDS.expired and CREDS.refresh_token:
            CREDS.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_JSON, SCOPES)
            CREDS = flow.run_local_server(port=5006)
        with open(CREDENTIALS_PKL, 'wb') as token:
            pickle.dump(CREDS, token)

    SERVICE = build('sheets', 'v4', credentials=CREDS)


def get_specific_row(experiment_number, sheet_number):
    initialize()

    if sheet_number == 1:
        start = 16
    elif sheet_number == 2:
        start = 11
    elif sheet_number == 0:
        start = 13
    else:
        print('ERROR: Sheet number %d not supported' % sheet_number)
        return None

    range_name = 'Sheet%d!D%d:D1000' % (sheet_number, start)

    result = SERVICE.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
    rows = result.get('values', [])
    specific_row = start + rows.index([str(experiment_number)])

    return specific_row


def get_next_row(sheet_number):
    range_name = 'Sheet%d!A1:A1000' % sheet_number

    result = SERVICE.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=range_name).execute()
    rows = result.get('values', [])
    next_row = len(rows) + 1
    return next_row


def write_settings(project_variable):
    # ONLY for implementation---------------------------------------------
    # from relative_baseline.omg_emotion.settings import ProjectVariable
    # project_variable = ProjectVariable()
    # ONLY for implementation---------------------------------------------

    initialize()

    if project_variable.sheet_number in [1, 2, 3]:

        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '',  # mean accuracy                                            #E
            '',  # std                                                      #F
            '',  # best run                                                 #G
            str(project_variable.data_points),                              # H
            str(project_variable.num_out_channels)                          # I
        ]]
        end_letter = 'I'
    elif project_variable.sheet_number in [0]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # mean accuracy                                            #E
            '',  # std                                                      #F
            '',  # best run                                                 #G
            str(project_variable.theta_init),  # H
            str(project_variable.srxy_init),  # I
            str(project_variable.srxy_smoothness),  # J
            project_variable.weight_transform  # K
        ]]
        end_letter = 'K'
    else:
        print('Error: sheet_number not supported')
        return None

    row = get_next_row(project_variable.sheet_number)
    range_name = 'Sheet%d!A%d:%s%d' % (project_variable.sheet_number, row, end_letter, row)

    data = [
        {
            'range': range_name,
            'values': values
        }  # ,
    ]
    body = {
        'valueInputOption': VALUE_INPUT_OPTION,
        'data': data
    }
    result = SERVICE.spreadsheets().values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID, body=body).execute()
    print('{0} cells updated.'.format(result.get('totalUpdatedCells')))
    
    return row


def write_results(accuracy, std, best_run, row, sheet_number):
    initialize()

    values = [[
        datetime.now().strftime('%H:%M:%S'),  # end time experiment     #C
    ]]
    range_name = 'Sheet%d!C%d' % (sheet_number, row)
    body = {
        'values': values
    }
    result = SERVICE.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID, range=range_name,
        valueInputOption=VALUE_INPUT_OPTION, body=body).execute()

    values = [[
        accuracy,  # mean accuracy                                      #E
        std,  # std                                                     #F
        best_run,  # best run                                           #G
    ]]
    range_name = 'Sheet%d!E%d:G%d' % (sheet_number, row, row)
    data = [
        {
            'range': range_name,
            'values': values
        }  # ,
    ]
    body = {
        'valueInputOption': VALUE_INPUT_OPTION,
        'data': data
    }
    result = SERVICE.spreadsheets().values().batchUpdate(
        spreadsheetId=SPREADSHEET_ID, body=body).execute()


# from relative_baseline.omg_emotion.settings import ProjectVariable
# project_variable = ProjectVariable()
#
# r = write_settings(project_variable)
# write_results(10, 1, 2, r)
