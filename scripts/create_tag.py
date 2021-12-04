import pandas as pd
import numpy as np
import datetime as dt
from RetailChurnTemplateUtility import *


def azureml_main(dataframe1 = None, dataframe2 = None):
    # The entry point function can contain up to two input arguments:
    #    
    # Param:
    #   dataframe1: JoinedActivityUserInfor DataFrame
    #   dataframe2: Churn Start Period 
    print "----- Dataset to handler -----\n"
    print dataframe1.columns
    print dataframe1.info()
    print "----- Dataset to handler -----\n"

    key_column='UserId'
    activity_column='TransactionId'
    
    churn_period = 21
    churn_threshold = 0
  
    dataframe1['Timestamp'] = dataframe1\
        .apply(lambda x : dt.datetime.fromtimestamp(x['Timestamp']).strftime('%Y-%m-%d'), axis=1)
    
    # Convert Column
    dataframe1['Parsed_Date'] = pd.to_datetime(dataframe1['Timestamp'])
    
    # Assigning Churn Status
    churnUtil = RetailChurnTemplateUtility(churn_period,churn_threshold)
    outdataframe = churnUtil.assign_churn_status(dataframe1,
                                                 key_column=key_column,
                                                 activity_column=activity_column) 
    print('outdataframe.head()', outdataframe.head())
    print('outdataframe.dtypes\n', outdataframe.dtypes)
    
    # Missing values
    outdataframe.fillna('Unknown', inplace = True)
    outdataframe['churn_period'] = churn_period
    
    to_keep_list = [row for row in outdataframe.columns if row != 'Parsed_Date']
    
    # Return value must be of a sequence of pandas.DataFrame
    return outdataframe[to_keep_list]
