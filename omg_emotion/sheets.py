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

    SERVICE = build('sheets', 'v4', credentials=CREDS, cache_discovery=False)


def get_specific_row(experiment_number, sheet_number):
    initialize()

    if sheet_number in [1, 3, 5, 9, 13]:
        start = 16
    elif sheet_number in [2, 17]:
        start = 11
    elif sheet_number in [0, 10, 15, 18, 22]:
        start = 13
    elif sheet_number in [4, 6, 12]:
        start = 17
    elif sheet_number in [7, 11, 14, 20, 21]:
        start = 14
    elif sheet_number == 666:
        start = 10
    elif sheet_number in [8, 16]:
        start = 15
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

    if project_variable.sheet_number in [1, 2, 3, 5]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.data_points),                              # I
            str(project_variable.num_out_channels)                          # J
        ]]
        end_letter = 'J'
    elif project_variable.sheet_number in [0]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '', # parameters                                                # E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.theta_init),                               # I
            str(project_variable.srxy_init),                                # J
            str(project_variable.srxy_smoothness),                          # K
            project_variable.weight_transform                               # L
        ]]
        end_letter = 'L'
    elif project_variable.sheet_number in [4, 6]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '', # parameters                                                # E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.num_out_channels),                         # I
            str(project_variable.transformation_groups),                    # J
            str(project_variable.k0_groups)                                 # K
        ]]
        end_letter = 'K'
    elif project_variable.sheet_number in [7]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '', # parameters                                                # E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.model_number),                         # I
            str(project_variable.load_model),                    # J
            str(project_variable.data_points),                                 # K
            str(project_variable.k0_init)  # L
        ]]
        end_letter = 'L'
    elif project_variable.sheet_number in [666]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            ''  # best run                                                 # H
        ]]
        end_letter = 'H'
    elif project_variable.sheet_number in [8]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '', # parameters                                                # E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.model_number),                         # I
            str(project_variable.load_model),                    # J
            str(project_variable.data_points)                                 # K
        ]]
        end_letter = 'K'
    elif project_variable.sheet_number in [9, 12]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.data_points),  # I
            str(project_variable.num_out_channels),  # J
            project_variable.model_number  # K
        ]]
        end_letter = 'K'
    elif project_variable.sheet_number in [10]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.load_num_frames),  # I
            str(project_variable.num_out_channels),  # J
            project_variable.learning_rate,  # K
            project_variable.model_number # L
        ]]
        end_letter = 'L'
    elif project_variable.sheet_number in [11]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.num_out_channels),  # I
            str(project_variable.conv1_k_t),  # J
            str(project_variable.do_batchnorm)  # batchnorm # K
        ]]
        end_letter = 'K'
    elif project_variable.sheet_number in [13]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            project_variable.theta_init,  # I
            project_variable.weight_transform,  # J
            str(project_variable.k_shape), # K
            str(project_variable.load_num_frames), #L
            project_variable.model_number #M
        ]]
        end_letter = 'M'
    elif project_variable.sheet_number in [14]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            project_variable.learning_rate,  # I
            project_variable.batch_size,  # J
            str(project_variable.num_out_channels), # K
            str(project_variable.k_shape), #L
            project_variable.k0_init, #M
            project_variable.transformations_per_filter, # N
            project_variable.load_num_frames # O
        ]]
        end_letter = 'O'
    elif project_variable.sheet_number in [15]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.data_points),  # I
            project_variable.learning_rate,  # J
            project_variable.adapt_eval_on, # K
            project_variable.decrease_after_num_epochs, #L
            project_variable.reduction_factor, #M
            str(project_variable.num_out_channels), # N
            project_variable.model_number # O
        ]]
        end_letter = 'O'
    elif project_variable.sheet_number in [16]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,                             # D
            '',  # parameters                                               # E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.load_model),                    # J
            str(project_variable.data_points),                                 # K
            project_variable.learning_rate,     # L
            project_variable.batch_size,        # M
            project_variable.decrease_after_num_epochs,      # N
            project_variable.model_number,  # I
        ]]
        end_letter = 'N'
    elif project_variable.sheet_number in [17]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            str(project_variable.experiment_number),                             # D
            '',  # parameters                                               # E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.load_model),                    # J
            project_variable.model_number,  # I
        ]]
        end_letter = 'J'
    elif project_variable.sheet_number in [18]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            str(project_variable.data_points),  # I
            project_variable.learning_rate,  # J
            project_variable.theta_learning_rate, # K
            project_variable.decrease_after_num_epochs, #L
            project_variable.reduction_factor, #M
            str(project_variable.num_out_channels), # N
            project_variable.repeat_experiments, # O
            project_variable.model_number # P
        ]]
        end_letter = 'P'
    elif project_variable.sheet_number in [20]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            project_variable.learning_rate,
            str(project_variable.num_out_channels),
            str(project_variable.use_adaptive_lr),
            project_variable.model_number
        ]]
        end_letter = 'L'
    elif project_variable.sheet_number in [21]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            project_variable.learning_rate,
            str(project_variable.num_out_channels),
            str(project_variable.use_adaptive_lr),
            project_variable.optimizer,
            project_variable.model_number
        ]]
        end_letter = 'M'
    elif project_variable.sheet_number in [22]:
        values = [[
            date.today().strftime('%d-%m-%Y'),  # date                      #A
            datetime.now().strftime('%H:%M:%S'),  # start time experiment   #B
            '',  # end time experiment                                      #C
            project_variable.experiment_number,  # D
            '',  # parameters                                               #E
            '',  # mean accuracy                                            #F
            '',  # std                                                      #G
            '',  # best run                                                 # H
            project_variable.learning_rate,
            str(project_variable.num_out_channels),
            str(project_variable.use_adaptive_lr),
            project_variable.optimizer,
            project_variable.model_number,
            str(project_variable.data_points),
            project_variable.repeat_experiments,
            project_variable.end_epoch,
            project_variable.batch_size,
            project_variable.stop_at_collapse,
            project_variable.early_stopping,
            '',  # best run stop epoch
            ''  # number of collapses

        ]]
        end_letter = 'U'
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


def write_parameters(parameters, row, sheet_number):
    initialize()

    values = [[
        parameters,  # end time experiment     #E
    ]]
    range_name = 'Sheet%d!E%d' % (sheet_number, row)
    body = {
        'values': values
    }
    result = SERVICE.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID, range=range_name,
        valueInputOption=VALUE_INPUT_OPTION, body=body).execute()


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
        accuracy,  # mean accuracy                                      #F
        std,  # std                                                     #G
        best_run,  # best run                                           #H
    ]]
    range_name = 'Sheet%d!F%d:H%d' % (sheet_number, row, row)
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


def extra_write_results(best_run_stop, num_runs_collapsed, row, sheet_number):
    initialize()

    values = [[
        best_run_stop,  # mean accuracy                                      #T
        num_runs_collapsed  # std                                            #U
    ]]
    range_name = 'Sheet%d!T%d:U%d' % (sheet_number, row, row)
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
