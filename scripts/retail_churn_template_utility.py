# -*- coding: utf-8 -*-

import datetime as dt

import numpy as np
import pandas as pd
from pandas._libs.reduction import reduce


class RetailChurnTemplateUtility:

    def __init__(self, churn_period=10, churn_threshold=0):
        self.churn_period = churn_period
        self.churn_threshold = churn_threshold

    @staticmethod
    def calculateDateDiff(date, max_date):
        """
        Method to calculate last day of current month
        Parameters
        ----------
        date:Datetime object
        """
        return (max_date - date).days

    @staticmethod
    def calculateDistinct_X(df1, key_column, X_column):
        """
        Method to calculate distinct entries in a column X
        Parameters
        ----------
        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        X_column: string
            The column for which unique entries are being calculated
        """
        return pd.DataFrame(
            df1.groupby(key_column)[X_column].nunique()).reset_index()

    @staticmethod
    def calculateSum_X(df1, key_column, X_column):
        """
        Method to calculate sum of all entries in a column X
        Parameters
        ----------

        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        X_column: string    
            The column for which unique sum is being calculated
        """
        return pd.DataFrame(
            df1.groupby(key_column)[X_column].sum()).reset_index()

    @staticmethod
    def calculateStDev_X(df1, key_column, X_column):
        """
        Method to calculate standard deviation a column X

        Parameters
        ----------

        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        X_column: string
            The column for standard deviation is being calculated
        """
        return pd.DataFrame(
            df1.groupby(key_column)[X_column].std()).reset_index()

    def calculateNumericalDataFeaturesForGroup(self, df1, key_column,
                                               summable_columns, max_date,
                                               rename_label):
        """
        Method to calculate features for Numerical fields in the data frame for a particular group
        Parameters
        ----------

        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        summable_columns: list[string]
            The list of columns for which number of unique/distinct entries are to be calculated

        rename_label:string
            Prefix to be preprended to features

        Returns 
        -------
            A dataframe containing the numeric features for a particular group
        """
        print('calculateNumericalDataFeaturesForGroup: max_date is {}, rename_label is {}'.format(max_date, rename_label))
        exclude_list = [key_column, 'month', 'ChurnPeriod', 'Timestamp']
        feat_dfs = []

        # Generating Features corresponding to the summable list
        for feat in summable_columns:

            if feat not in exclude_list:
                df_temp = self.calculateSum_X(df1, key_column, feat)
                print('Calculating Sum for ', feat, ' returned a data frame of shape', df_temp.shape)
                df_temp.columns = [key_column, 'Total_' + feat]
                feat_dfs.append(df_temp)

        # Calculating Standard Deviations for summable features    
        for feat in summable_columns:

            if feat not in exclude_list:
                df_temp = self.calculateStDev_X(df1, key_column, feat)
                print('Calculating Stdev for ', feat, ' returned a data frame of shape', df_temp.shape)
                df_temp.columns = [key_column, 'StDev_' + feat]
                feat_dfs.append(df_temp)

        df_final = reduce(lambda left,
                                 right: pd.merge(left,
                                                 right,
                                                 on=key_column,
                                                 how='outer'), feat_dfs)
        renamed_columns_list = []
        df_recency = self.calculateRecency(df1, key_column, max_date)
        df_timedelta = self.calculateTimeDelta(df1, key_column)
        feat_dfs.append(df_timedelta)
        feat_dfs.append(df_recency)
        df_final = reduce(lambda left,
                                 right: pd.merge(left,
                                                 right,
                                                 on=key_column,
                                                 how='outer'), feat_dfs)
        print('CalculateNumericalDataFeaturesForGroup returned a dataframe of shape', df_final.shape)

        return df_final

    def calculateNumericalDataFeatures(self, df1, key_column, summable_columns,
                                       rename_label, IsDevelopment=True):
        """
        Method to calculate features for Numerical fields in the data frame
        Parameters
        ----------
        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        summable_columns: list[string]
            The list of columns for which number of unique/distinct entries are to be calculated

        rename_label:string
            Prefix to be prepended to features

        Returns
        -------
            A dataframe containing the numeric features for all of the groups
        """
        #    Extracting Timestamp
        df1.Timestamp = df1.apply(
            lambda x: dt.datetime.fromtimestamp(x['Timestamp']).strftime(
                '%Y-%m-%d'), axis=1)
        print(df1.Timestamp.head())
        churn_period = self.churn_period

        df1['Parsed_Date'] = pd.to_datetime(df1['Timestamp'])
        min_date = df1.Parsed_Date.min()
        max_date = df1.Parsed_Date.max()

        print('min_date: {},max_date: {}'.format(min_date, max_date))

        base_features_list = [key_column]
        df_profile = df1[base_features_list].drop_duplicates()
        print('Shape of the df_profile is ', df_profile.shape)

        # Apply feature engineering for the data of each of the group

        # Exclude the dates >=churn period data
        ## Churn period is defined by the group id. 
        days_to_exclude = int(churn_period)
        start_date = min_date

        if IsDevelopment == True:
            end_date = max_date - np.timedelta64(days_to_exclude, 'D')
        else:
            end_date = max_date

        print('Training Period Start Date:{}, End Date: {}, Days Excluded:{}'.format(start_date, end_date, days_to_exclude))
        df_temp = df1

        df_current = df_temp.ix[df_temp['Parsed_Date'].between(start_date, end_date)]
        print('CalculateNumericalDataFeatures :  shape:{}'.format(df_current.shape))
        df_num = self.calculateNumericalDataFeaturesForGroup(df_current,
                                                             key_column,
                                                             summable_columns,
                                                             end_date,
                                                             rename_label='Overall-')
        df_num.fillna(0, inplace=True)
        print('df_numeric.shape ', df_num.shape)

        return df_num

    def calculateStringDataFeaturesForGroup(self, df1, key_column,
                                            uniquable_columns, rename_label):
        """
        Method to calculate features for String/Textual fields in the data frame for a particular group

        Parameters
        ----------

        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        uniquable_columns: list[string]
            The list of columns for which number of unique/distinct entries are to be calculated

        rename_label:string
            Prefix to be prepended to features
        Returns
        -------
            A dataframe containing the textual features for a group

        """
        exclude_list = [key_column, 'month', 'ChurnPeriod', 'Timestamp']

        # Generating Features corresponding to the uniquable list
        feat_dfs = []

        for feat in uniquable_columns:

            if feat not in exclude_list:
                print('Calculating number of unique entries for ', feat)
                df_temp = self.calculateDistinct_X(df1, key_column, feat)
                df_temp.columns = [key_column, 'Unique_' + feat]
                feat_dfs.append(df_temp)

        # Generating Features corresponding to the summable list
        df_final = reduce(lambda left,
                                 right: pd.merge(left,
                                                 right,
                                                 on=key_column,
                                                 how='outer'), feat_dfs)
        renamed_columns_list = []

        return df_final

    def calculateStringDataFeatures(self, df1, key_column, uniquable_columns,
                                    rename_label, IsDevelopment=True):
        """
        Method to calculate features for String/Textual fields in the data frame

        Parameters
        ----------

        df1: pandas.DataFrame
            dataframe containg the data

        key_column: string
            The column having the User ID

        uniquable_columns: list[string]
            The list of columns for which number of unique/distinct entries are to be calculated

        categorical_columns: list[string]
            The list of categorical variables

        rename_label:string
            Prefix to be preprended to features

        Returns
        -------
            A dataframe containing the textual features for all of the groups
        """
        base_features_list = [key_column]
        df_profile = df1[base_features_list].drop_duplicates()
        churn_period = self.churn_period

        df1.Timestamp = df1.apply(
            lambda x: dt.datetime.fromtimestamp(x['Timestamp']).strftime(
                '%Y-%m-%d'), axis=1)
        print(
        df1.Timestamp.head())

        #   Important to have year in the month as well, it helps in ordering of months
        df1['Parsed_Date'] = pd.to_datetime(df1['Timestamp'])
        min_date = df1.Parsed_Date.min()
        max_date = df1.Parsed_Date.max()

        base_features_list = [key_column]
        df_profile = df1[base_features_list].drop_duplicates()

        # Calculating recency related features 1) Sort the time stamps 2) Get min and max date 3) Extend the max date to the end of the month as it is the current unit right now        
        # Apply feature engineering for the data of each of the group        
        # Exclude the dates >=churn period data

        days_to_exclude = int(churn_period)
        df_temp = df1
        start_date = df_temp.Parsed_Date.min()
        if IsDevelopment == True:
            end_date = max_date - np.timedelta64(days_to_exclude, 'D')
        else:
            end_date = max_date
        print(
        'Training  Start Date:{} ,  end date: {}'.format(start_date, end_date))
        # df_current=df_temp
        df_current = df_temp.ix[
            df1['Parsed_Date'].between(start_date, end_date)]
        df_str = self.calculateStringDataFeaturesForGroup(df_current,
                                                          key_column,
                                                          uniquable_columns,
                                                          rename_label='Overall-')
        df_str.fillna(0, inplace=True)
        print(
        'df_str.shape ', df_str.shape)

        return df_str

    def calculateRecency(self, df1, key_column, max_date):
        """
        # Calculating recency related features 
            1) Sort the time stamps 
            2) Get min and max date 
            3) Extend the max date to the end of the month as it is the current unit right now
        
        Method to calculate difference between the last day of the period and the last day on which the transaction was made by each of the user.

        Parameters
        ----------

        df1: pandas.DataFrame
            List of pandas dataframes such that each dataframe has user activity for a different time interval.
            If there are months like 01 02 03 04 then the list should contain four data frames in the order 01,02,03,04

        key_column: string
            The column having the User ID

        max_date:pandas.Datetime
            Max possible date time in the time span of the current group.

        Returns
        ---------
            A dataframe containing the Recency for each of the users in the dataframe.
        """
        if 'Timestamp' in df1.columns:
            df_recency = df1[[key_column, 'Timestamp']]
            df_recency.Timestamp = pd.to_datetime(df_recency.Timestamp)
            df_recency = df_recency.groupby(key_column).Timestamp.max().apply(
                lambda x: self.calculateDateDiff(x, max_date)).reset_index()
            df_recency.columns = [key_column, 'Recency']
            print(df_recency.head())
            return df_recency

        else:
            raise Exception("Timestamp not in the input data frame")

    def timedelta(self, x):
        """
        Utility method to calculate average time ( in terms of number of days) between transactions

        Parameters
        ----------

        x: pandas.Series(Timestamp)
            Series containing timestamps of the transactions

        Returns
        --------
            Average time between transactions for a user
        """
        x = sorted(x)
        deltasum = 0.0
        for i in range(1, len(x)):
            deltasum += (x[i] - x[i - 1]).days

        return deltasum / float(len(x))

    def calculateTimeDelta(self, df, key_column='UserId'):
        """
        Method to calculate average time ( in terms of number of days) between transactions  for all of the users

        Parameters
        ----------

        df: pandas.DataFrame
            DataFrame containing the timestamps and other information about the transactions
        key_column:String
            the column on which aggregation has to be made. ( UserId )

        Returns
        --------
            A dataframe containing the userid and average time between transactions for each of the user.
        """
        df.Timestamp = pd.to_datetime(df.Timestamp)
        df2 = pd.DataFrame(df.groupby(key_column).Timestamp.apply(list).apply(
            self.timedelta)).reset_index()
        df2.columns = [key_column, 'AvgTimeDelta']

        return df2

    def calculateAverages(self, df1, df2, key_column, uniquable_columns,
                          summable_columns):
        """
        Method to calculate averages which are basically the ratio of the features returned by the calculateStringDataFeatures to the features returned by the calculateNumericalDataFeatures

        Parameters
        ----------

        df1: pandas.DataFrame
            DataFrame containing the Numerical Features
        df2: pandas.DataFrame
            DataFrame containing the String Features
        key_column:String
            the column on which aggregation has to be made. ( UserId )
        uniquable_columns:list[String]
            name of the columns in the dataframe2
        summable_columns:list[String]
            name of the columns in the dataframe1
        Returns
        --------
            A dataframe containing averages.
        """
        base_features_list = [key_column]
        df_profile = df1[base_features_list].drop_duplicates()
        exclude_list = [key_column, 'month', 'ChurnPeriod', 'Timestamp',
                        'Group']

        df_final = reduce(lambda left,
                                 right: pd.merge(left,
                                                 right,
                                                 on=[key_column],
                                                 how='outer'), [df1, df2])

        print('Uniquable columns', uniquable_columns)
        print('Summable columns', summable_columns)
        print('df_final.shape', df_final.shape)

        # Calculating normalized stats
        for num_feat in summable_columns:

            if num_feat not in exclude_list:

                # numerator feature
                for denom_feat in uniquable_columns:

                    if denom_feat not in exclude_list:

                        if 'Period' in num_feat or 'Period' in denom_feat:

                            if 'Period' in num_feat and 'Period' in denom_feat:

                                print('Calculating {}'.format(
                                    num_feat + '_ratio_' + denom_feat))
                                num_prefix = num_feat.split('_')[0].split('-')[
                                    1]
                                denom_prefix = \
                                    denom_feat.split('_')[0].split('-')[1]

                                if num_prefix == denom_prefix:
                                    print(
                                    'Calculating {}'.format(
                                        num_feat + '_ratio_' + denom_feat))
                                    df_final[num_feat + '_per_' + denom_feat] = \
                                        df_final[num_feat] / (
                                                df_final[denom_feat] + 1.0)

                        elif 'StDev' in num_feat 
                            or 'StDev' in denom_feat 
                            or 'Recency' in num_feat 
                            or 'Recency' in denom_feat 
                            or 'AvgTimeDelta' in num_feat 
                            or 'AvgTimeDelta' in denom_feat:
                            pass
                        else:
                            print('Calculating {}'.format(num_feat + '_ratio_' + denom_feat))
                            df_final[num_feat + '_per_' + denom_feat] = \
                                df_final[num_feat] / (df_final[denom_feat] + 1.0)

        df_final = df_final.reset_index()
        df_final.fillna(0, inplace=True)
        renamed_columns_list = []
        to_keep_list = [key_column]

        for each in df_final.columns:

            if 'per' in each:
                to_keep_list.append(each)

        return df_final[to_keep_list]

    def assign_churn_status_to_group(self, df, min_date, max_date, key_column,
                                     activity_column):
        """
            Functions used to label Churn
        """

        """ This function is internally called by the assign_churn_status function. 
        This function can be used to assign Churn status to all the users or the users in some specific group.
        This function does not do any sort of grouping and assumes that the input dataframe consists of only one group.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the activity data

        min_date: datetime
            Minimum date in the dataframe

        max_date: datetime
            Maximum date in the dataframe

        key_column:string
            The column specifying the user id

        activity_column:
            The column identifying each unique transaction i.e. TransctionId
        """

        churn_period = self.churn_period
        churn_threshold = self.churn_threshold

        # Calculate the activity for each subscriber in the overall period
        df_overall = pd.DataFrame(df.groupby(key_column)[
                                      activity_column].count()).reset_index().copy()
        df_overall.columns = [key_column, 'OverallProductsPurchased']

        # Calculating days to subtract from the max date to calculate end of the pre churn period
        days = int(churn_period)

        # Calculate the activity for each subscriber in the prechurn period
        df_prechurn = pd.DataFrame(df[df['Parsed_Date'].between(min_date,
                                                                max_date - np.timedelta64(
                                                                    days,
                                                                    'D'))].groupby(
            key_column)[activity_column].count()).reset_index().copy()
        df_prechurn.columns = [key_column, 'PrechurnProductsPurchased']

        print(
        'Overall max date', df.Parsed_Date.max())
        print(
        'PreChurn max date', max_date - np.timedelta64(days, 'D'))
        print(
        'df_overall.shape ', df_overall.shape)
        print(
        'df_prechurn.shape ', df_prechurn.shape)

        # Merging the prechurn and overall dataframes
        df_prechurn = pd.merge(df_overall, df_prechurn, on=key_column,
                               how='outer')
        df_prechurn[
            'diff_products_purchased'] = df_prechurn.OverallProductsPurchased - df_prechurn.PrechurnProductsPurchased

        # Assigning the Churner and Nonchurner labels
        df_prechurn['Label'] = np.where(
            df_prechurn.diff_products_purchased <= churn_threshold, 'Churner',
            'Nonchurner')

        # Removing extra columns from the dataframe
        to_keep = [each for each in df_prechurn.columns if
                   each not in ['OverallProductsPurchased',
                                'PrechurnProductsPurchased',
                                'diff_products_purchased']]

        print('Overall max date', df.Parsed_Date.max())
        print('PreChurn max date', max_date - np.timedelta64(days, 'D'))
        print('df_overall.shape ', df_overall.shape)
        print('df_prechurn.shape ', df_prechurn.shape)
        print(df_prechurn.head())

        return df_prechurn[to_keep]

    def assign_churn_status(self, df, key_column, activity_column):
        """ Method to assign churn status to each individual in the data
        Parameters
        ----------
        df : pandas.DataFrame
        DataFrame containing the activity data

        key_column:string
            The column specifying the user id

        activity_column:
            The column identifying each unique transaction i.e. TransctionId

        """
        churn_period = self.churn_period

        # Calculating the min and max date in the dataframe
        min_date = df.Parsed_Date.min()
        max_date = df.Parsed_Date.max()

        print(min_date, max_date, (max_date - min_date).days)

        interval_start = min_date
        interval_end = max_date
        min_reqd_period = int(churn_period * 2)

        if (interval_end - np.timedelta64(min_reqd_period,'D')) >= interval_start:
            df_labeled = self.assign_churn_status_to_group(df, interval_start,
                                                           interval_end,
                                                           key_column,
                                                           activity_column)
            return df_labeled
        else:
            raise Exception( "The training period should be at least two times longer than the churn period.")
