import tsdb
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass



@dataclass
class DataProcessor:
    df: pd.DataFrame
    df_label: pd.DataFrame
    id_col: str
    time_col: str
    global_features: list
    cols_exclude_z_norm: list
    max_wanted_len: int
    use_padding: bool = True
    data_tensor: torch.tensor = None
    label_tensor: torch.tensor = None
    taget_name: str = None
    event_to_token : dict = None


    def replace_global_features_with_nan(self):
        """
        Replaces all numeric values for specified global features with NaN except for the first observation per id.
        """
        # Sort the DataFrame by id and timestamp
        self.df = self.df.sort_values(by=[self.id_col, self.time_col]).reset_index(drop=True)

        # Create a mask for the first occurrence of each ID
        # first_occurrence = self.df.groupby(self.id_col).cumcount() == 0

        # # Iterate through global features and set non-first occurrences to NaN
        # for feature in self.global_features:
        #     self.df[feature] = np.where(first_occurrence, self.df[feature], np.nan)

        mask = self.df.groupby(self.id_col).cumcount() == 0
        self.df.loc[~mask, self.global_features] = np.nan


    def zscore_transformation(self):
        """
        Apply z-score normalization to all columns except those specified in cols_exclude_z_norm.
        """
        feature_columns = [col for col in self.df.columns if col not in self.cols_exclude_z_norm]

        # Apply z-score transformation
        self.df[feature_columns] = self.df[feature_columns].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )


    def melt_dataframe(
            self,
            feature_name="event",
            value_name="value"
                       ):
        """
        Reshapes the DataFrame to a long format using melt, excluding NaN values for the value column.
        """
        self.df = pd.melt(
            self.df, 
            id_vars=[self.id_col, self.time_col], 
            var_name=feature_name, 
            value_name=value_name
        ).dropna(subset=["value"]).sort_values(by=[self.id_col, self.time_col])

    def merge_label(
        self,
        # target,
    ):
        # self.taget_name = target
        self.df = pd.merge(self.df, self.df_label, on=self.id_col,how='inner')
        # self.label_tensor = torch.tensor(self.df.drop_duplicates(subset=self.id_col, keep='first')[target].values)


    
    def tokenizer(
            self,
            col_event='event'
        ):
        # Step 1: Map unique events to integer tokens
        unique_events = self.df[col_event].unique()
        self.event_to_token = {event: idx for idx, event in enumerate(unique_events)}

        # Step 2: Apply the mapping to create tokenized data
        self.df[col_event] = self.df[col_event].map(self.event_to_token)


    def df_to_3dtensor(
            self,
    ):

        id = self.id_col
        date = self.time_col

        grouped = self.df.groupby(id)
        max_length = min(self.max_wanted_len, max(grouped.size()))
        if self.max_wanted_len > max_length:
            print(f"max_wanted_len is langer dan de de aantal timestamps in de data,namelijk: {self.max_wanted_len}. data heeft max van: {max_length}")
        
        tensors = []
        for _, group in grouped:
            if len(group) > max_length:
                # cut of the oldest record to given sequence length
                group = group.iloc[-max_length:]
            elif self.use_padding:
                # fill up wit 0 paddings if shorten than given seuqence length:
                padding = pd.DataFrame([{date: None, **{feat: 0 for feat in group.columns if feat not in [self.id_col]}}] * (
                        max_length - len(group)))
                group = pd.concat([group, padding], ignore_index=True)
                group[id] = group[id].max()  # fill the empty id with id number

            group_features = group.values
            group_tensor = torch.from_numpy(group_features)
            tensors.append(group_tensor)

        data_tensor = torch.stack(tensors)
        
        self.data_tensor = data_tensor[:,:,1:] # remove id
        self.data_tensor = data_tensor[:,:,:-1] # remove target
        self.label_tensor = data_tensor[:,:,-1].max(dim=1).values # get target
        # return self.tensor

    def return_(
            self,
            name="df"
            ):
        if name == "df":
            return self.df
        if name == "3dtensor":
            return self.data_tensor.float()
        if name == "label_tensor":
            return self.label_tensor.long()
    
    
    def apply_steps(self, steps: list):
        for step in steps:
            if hasattr(self, step):
                getattr(self, step)()
            else:
                print(f"Step '{step}' not found in DataProcessor.")
        
    def get_data(self, steps=None):
        if steps is None:
            steps = [
                "replace_global_features_with_nan",
                "zscore_transformation",
                "melt_dataframe",
                "merge_label",
                "tokenizer",
                "df_to_3dtensor"
            ]
        self.apply_steps(steps)
        return self.return_("3dtensor"), self.return_("label_tensor")