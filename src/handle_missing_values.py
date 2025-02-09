import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


# Concrete Strategy for Dropping Specific Columns
class DropSpecificColumnsStrategy(MissingValueHandlingStrategy):
    def __init__(self, columns):
        """
        Initializes the DropSpecificColumnsStrategy with specific columns to drop.

        Parameters:
        columns (list): List of column names to drop.
        """
        self.columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified columns from the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
        """
        logging.info(f"Dropping columns: {self.columns}")
        df_cleaned = df.drop(columns=self.columns, errors="ignore")  # errors="ignore" ensures it won't fail if column is missing
        logging.info("Specified columns dropped successfully.")
        return df_cleaned


# Concrete Strategy for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned


# Concrete Strategy for Filling Missing Values with Median for Specific Columns
class FillMissingValuesWithMedianStrategy(MissingValueHandlingStrategy):
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in Power, Engine, Mileage, and Seats with their median values.
        Fills missing values in Price with the median price per brand and model.
        Drops rows where Price is still missing after filling.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Filling missing values with median for specific columns.")

        # Fill missing values in Power, Engine, Mileage, and Seats with their median values
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        numeric_columns.remove('Price')  # Exclude the dependent variable
        medianFiller = lambda x: x.fillna(x.median())
        df[numeric_columns] = df[numeric_columns].apply(medianFiller, axis=0)

        # Fill missing values in Price with median price per brand and model
        Brand_name = df['Car_Brand'].unique()
        Model = df['Model'].unique()

        Median1 = []  # Median Price per Brand
        for brand in Brand_name:
            median_price = df['Price'][df['Car_Brand'] == brand].median()
            Median1.append(median_price)

        Median2 = []  # Median Price per Model
        for model in Model:
            median_price = df['Price'][df['Model'] == model].median()
            Median2.append(median_price)

        # Replace missing values in Price with 0.0
        df['Price'] = df['Price'].fillna(0.0)

        # Replace 0.0 with median price per brand
        for i in range(len(df)):  # Loop through each row
            if df.loc[i, 'Price'] == 0.00:
                for j in range(len(Brand_name)):
                    if df.loc[i, 'Car_Brand'] == Brand_name[j]:  # Match the brand
                        df.loc[i, 'Price'] = Median1[j]  # Replace with median price of the brand

        # Drop rows where Price is still missing (NaN)
        logging.info("Dropping rows where Price is still missing after filling.")
        df.dropna(subset=['Price'], axis=0, inplace=True)

        logging.info("Missing values filled for Power, Engine, Mileage, Seats, and Price.")
        return df


# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    df = pd.read_csv(r"D:\Github\Linear_Regression_A-Z\extracted_data\used_cars_data.csv")

    # Step 1: Drop 'S.No.' explicitly
    drop_columns_handler = MissingValueHandler(DropSpecificColumnsStrategy(columns=["S.No."]))
    df = drop_columns_handler.handle_missing_values(df)

    # Step 2: Drop 'New_Price' based on missing values (since it has too many NaNs)
    drop_missing_handler = MissingValueHandler(DropMissingValuesStrategy(axis=1, thresh=len(df) * 0.5))
    df = drop_missing_handler.handle_missing_values(df)  # Drops columns with >50% missing values

    # Save the cleaned dataset
    df.to_csv("../extracted_data/cleaned_used_cars_data.csv", index=False)

    # Example dataframe
    processed_df = pd.read_csv(r"D:\Github\Linear_Regression_A-Z\extracted_data\after_data_types_fixing.csv")

    # Step 3: Fill missing values in Power, Engine, Mileage, Seats, and Price
    fill_missing_handler = MissingValueHandler(FillMissingValuesWithMedianStrategy())
    final_df= fill_missing_handler.handle_missing_values(processed_df)

    # Save the cleaned dataset
    df.to_csv("../extracted_data/filled_data.csv", index=False)