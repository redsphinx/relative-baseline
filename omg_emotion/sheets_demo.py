from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'
# SAMPLE_RANGE_NAME = 'Class Data!A2:E'

pick = '/home/gabras/token.pickle'

def main():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(pick):
        with open(pick, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '/home/gabras/credentials.json', SCOPES)

            creds = flow.run_local_server(port=5006)
        # Save the credentials for the next run
        with open(pick, 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    # sheet = service.spreadsheets()
    # result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
    #                             range=SAMPLE_RANGE_NAME).execute()
    # values = result.get('values', [])
    #
    # if not values:
    #     print('No data found.')
    # else:
    #     print('Name, Major:')
    #     for row in values:
    #         # Print columns A and E, which correspond to indices 0 and 4.
    #         print('%s, %s' % (row[0], row[4]))

    spreadsheet_id = '1p8S73Li52kqmi9NO-eJMOjnLbkVud9jMQ7PXxoTI2jA'
    range_name = 'A2:F2'
    value_input_option = 'RAW'

    values = [
        [
            'gabi',
            'erdi',
            1,
            2,
            3,
            4
        ]#,
    ]
    data = [
        {
            'range': range_name,
            'values': values
        }#,
    ]
    body = {
        'valueInputOption': value_input_option,
        'data': data
    }
    result = service.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id, body=body).execute()
    print('{0} cells updated.'.format(result.get('totalUpdatedCells')))



    # values = [
    #     ['gabi'], ['erdi'], [133], [2], [3], [4]
    # ]
    # body = {
    #     'values': values
    # }
    # result = service.spreadsheets().values().update(
    #     spreadsheetId=spreadsheet_id, range=range_name,
    #     valueInputOption=value_input_option, body=body).execute()
    # print('{0} cells updated.'.format(result.get('updatedCells')))


if __name__ == '__main__':
    main()