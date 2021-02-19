"""
Preprocessing steps done on the Delhi weather dataset.

Written by Tanmay Patil
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_csv(file_path: str) -> pd.DataFrame:
    """Read csv file and returns it as a pandas dataframe."""
    return pd.read_csv(file_path)

def remove_underscores(data_frame: pd.DataFrame) -> pd.Index:
    """Remove the undercores in the column names."""
    return data_frame.columns.str.replace("_","")

def remove_spaces(data_frame: pd.DataFrame) -> pd.Index:
    """Remove the spaces in the column names."""
    return data_frame.columns.str.replace(" ","")

def generalize_conditions(data_frame:pd.DataFrame, cond: pd.Series, weather_cond: dict) -> pd.DataFrame:
    """Consolidates multiple related weather conditions into one label."""
    data_frame['conds'] = cond.replace(weather_cond["Dust"], "Dust")
    data_frame['conds'] = cond.replace(weather_cond["Fog"], "Fog")
    data_frame['conds'] = cond.replace(weather_cond["Cloudy"], "Cloudy")
    data_frame['conds'] = cond.replace(weather_cond["Rain"], "Rain")
    data_frame['conds'] = cond.replace(weather_cond["Thunderstorms"], "Thunderstorms")
    data_frame['conds'] = cond.replace(weather_cond["Others"], "Others")
    return data_frame

def cond_to_numeric(data_frame: pd.DataFrame, col: str = "conds") -> pd.DataFrame:
    """Encode the weather column using LabelEncoder."""
    label_encoder = LabelEncoder()
    data_frame[col] = data_frame.apply(lambda x: label_encoder.fit_transform(data_frame[col].astype(str)),axis=0, result_type='expand')
    return data_frame

def zero_to_nan(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert 0 data to nan."""
    return data_frame.replace(0, np.nan)

def replace_nan(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert nan to either median or mode of the column depending on whether the column has more than 50% nan values."""
    half_rows= 0.5 * data_frame.shape[0]
    for i in data_frame.columns:
        total_nan = data_frame[i].isnull().sum()
        if total_nan < half_rows:
            if data_frame[i].dtypes == "object":
                temp_mode = data_frame[i].mode()[0]
                data_frame[i].fillna(temp_mode, inplace=True)
            else:
                temp_median = data_frame[i].median()
                data_frame[i].fillna(temp_median, inplace=True)
        else:
            data_frame.drop([i], axis=1, inplace=True)
    return data_frame

def wind_dir_to_deg(data_frame: pd.DataFrame, deg: int = 45) -> pd.DataFrame:
    """Convert the wind direction column into numerical degree values."""
    deg_list = [deg * i for i in range(8)]
    data_frame["wdire"] = data_frame["wdire"].replace(["WNW", "WSW", "ESE", "ENE", "NNW", "SSE", "NNE" ,"SSW", "Variable"],
                                    ["West", "West", "East", "East", "North", "South", "North", "South", "North"])
    data_frame["wdire"]= data_frame["wdire"].replace(["North","NE", "East","SE", "South","SW", "West", "NW"], deg_list)
    return data_frame

if __name__ == "__main__":
    df = read_csv("../data/raw/delhi_weather_data.csv")
    df.columns = remove_underscores(df)
    df.columns = remove_spaces(df)
    weather_conditons = {
        "Dust":["Widespread Dust", "Blowing Sand","Sandstorm", "Volcanic Ash" , "Light Sandstorm"],
        "Fog": ["Fog", "Shallow Fog", "Partial Fog", "Light Fog", "Mist", "Heavy Fog", "Light Haze", "Patches of Fog"],
        "Cloudy": ["Scattered Clouds", "Partly Cloudy", "Mostly Cloudy" ,"Overcast", "Funnel Cloud"],
        "Rain": ["Light Rain", "Light Drizzle","Rain", "Drizzle", "Light Rain Showers", "Drizzle" ,"Rain Showers"],
        "Thunderstorms": ["Thunderstorms and Rain", "Light Thunderstorms and Rain", "Light Thunderstorm" ,"Heavy Thunderstorms and Rain", "Heavy Rain"],
        "Others": ["Thunderstorms with Hail", "Squalls", "Light Hail Showers" ,"Light Freezing Rain", "Heavy Thunderstorms with Hail", "Unknown"]
    }
    df = generalize_conditions(df, df["conds"], weather_conditons)
    df = cond_to_numeric(df)
    df = zero_to_nan(df)
    df = replace_nan(df)
    df = wind_dir_to_deg(df)
    df.to_csv('../data/processed/delhi_weather_data_processed.csv')
    print(df.head())
