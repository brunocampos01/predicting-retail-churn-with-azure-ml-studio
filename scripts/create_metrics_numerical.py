import pandas as pd
import numpy as np
from RetailChurnTemplateUtility import *

def azureml_main(dataframe1=None, dataframe2=None):
    key_column='UserId'
    Is_development=True

    # Specifying churn start period
    churn_period=dataframe1.iloc[0]['churn_period']   

    # Feature Engineering
    churnUtil = RetailChurnTemplateUtility(churn_period=churn_period)
    print(churnUtil)
    
    # Feature Engineering
    return churnUtil.calculateNumericalDataFeatures(dataframe1,
                                                    key_column, 
                                                    summable_columns=dataframe1.columns,
                                                    rename_label='overall',
                                                    IsDevelopment=Is_development)
    
