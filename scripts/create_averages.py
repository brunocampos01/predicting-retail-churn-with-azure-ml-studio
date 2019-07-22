import pandas as pd
import numpy as np
from RetailChurnTemplateUtility import *

def azureml_main(dataframe1 = None, dataframe2 = None):
    key_column='UserId'
    IsDevelopment=True

    # Feature Engineering
    churnUtil=RetailChurnTemplateUtility()
    df=churnUtil.calculateAverages(dataframe1,
                                   dataframe2,
                                   key_column,
                                   uniquable_columns=dataframe2.columns, 
                                   summable_columns=dataframe1.columns)
    
    # Return value must be of a sequence of pandas.DataFrame
    return df
