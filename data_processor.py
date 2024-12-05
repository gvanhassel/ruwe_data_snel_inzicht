from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import torch

@dataclass
class DataProcessor:
    df: pd.DataFrame
    id_col: str
    time_col: str
    global_features: list
    max_wanted_len: int
    use_padding: bool = True
    data_tensor: torch.tensor = field(default_factory=lambda: torch.empty(0))
    label_tensor: torch.tensor = field(default_factory=lambda: torch.empty(0))
    taget_name: str = field(default_factory=str)
    event_to_token: dict = field(default_factory=dict)

    def replace_global_features_with_nan(self):
        """
        Replaces all numeric values for specified global features with NaN except for the first observation per id.
        """
        self.df = self.df.sort_values(by=[self.id_col, self.time_col]).reset_index(drop=True)
        first_occurrence = self.df.groupby(self.id_col).cumcount() == 0
        for feature in self.global_features:
            self.df[feature] = np.where(first_occurrence, self.df[feature], np.nan)

    def zscore_transformation(self, cols_exclude):
        """
        Apply z-score normalization to all columns except those specified in cols_exclude.
        """
        feature_columns = [col for col in self.df.columns if col not in cols_exclude]
        self.df[feature_columns] = self.df[feature_columns].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

    def melt_dataframe(self, feature_name="event", value_name="value"):
        """
        Reshapes the DataFrame to a long format using melt, excluding NaN values for the value column.
        """
        self.df = pd.melt(
            self.df,
            id_vars=[self.id_col, self.time_col],
            var_name=feature_name,
            value_name=value_name
        ).dropna(subset=["value"]).sort_values(by=[self.id_col, self.time_col])

    def merge_label(self, df_label, target):
        """
        Merges the label DataFrame with the main DataFrame.
        """
        self.taget_name = target
        self.df = pd.merge(self.df, df_label, on=self.id_col, how='inner')

    def tokenizer(self):
        """
        Maps unique events to integer tokens and applies the mapping.
        """
        unique_events = self.df['event'].unique()
        self.event_to_token = {event: idx for idx, event in enumerate(unique_events)}
        self.df['event'] = self.df['event'].map(self.event_to_token)

    def df_to_3dtensor(self):
        """
        Converts the DataFrame into a 3D tensor representation.
        """
        grouped = self.df.groupby(self.id_col)
        max_length = min(self.max_wanted_len, max(grouped.size()))
        if self.max_wanted_len > max_length:
            print(f"max_wanted_len is larger than the number of timestamps in the data: {self.max_wanted_len}. Data has a max of: {max_length}")
        
        tensors = []
        for _, group in grouped:
            if len(group) > max_length:
                group = group.iloc[-max_length:]
            elif self.use_padding:
                padding = pd.DataFrame([{self.time_col: None, **{feat: 0 for feat in group.columns if feat not in [self.id_col]}}] * (
                    max_length - len(group)))
                group = pd.concat([group, padding], ignore_index=True)
                group[self.id_col] = group[self.id_col].max()
            group_features = group.values
            group_tensor = torch.from_numpy(group_features)
            tensors.append(group_tensor)

        data_tensor = torch.stack(tensors)
        self.data_tensor = data_tensor[:, :, 1:4]  # remove id:target
        self.label_tensor = data_tensor[:, :, 4].max(dim=1).values  # get target

    def return_(self, name="df"):
        """
        Returns the requested data: DataFrame, 3D tensor, or label tensor.
        """
        if name == "df":
            return self.df
        if name == "3dtensor":
            return self.data_tensor
        if name == "label_tensor":
            return self.label_tensor
