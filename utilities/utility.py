# Preprocessor class that will be the main pipeline for our project

import pandas as pd
import os

from pathlib import Path
from typing import Literal

class AMRDataPipeline:
    def __init__(self, rtab_path: str, 
                 metadata_path: str, 
                 antibiotic_col: Literal['cfx_sr','azm_sr','cip_sr'],
                 mode: Literal['auto','custom'],
                 cols_to_dropped: list[str] = None,
                 index_name: str = "Sample_ID",
                 ):
        self.rtab_path = Path(rtab_path)
        self.metadata_path = Path(metadata_path)
        self.antibiotic_col = antibiotic_col
        self.index_name = index_name
        self.unitig_df = None
        self.metadata_df = None
        self.final_df = None
        self.mode = mode
        self.preserved_metadata_cols = [antibiotic_col, index_name]

        if cols_to_dropped is not None:
            self.cols_to_dropped = [cols for cols in cols_to_dropped if cols not in self.preserved_metadata_cols]

    def _load_transpose_rtab(self) -> pd.DataFrame | None:
        # Read the path and set 'pattern_id' as their structural anchor with automatic seperator detector
        self.unitig_df = pd.read_csv(self.rtab_path, index_col=0, sep=None, engine='python')

        # Transpose the data
        self.unitig_df = self.unitig_df.T

        # Set the anchor (index) as 'Sample_ID' to match the meta dataset
        self.unitig_df.index.name = self.index_name

        return self.unitig_df

    def _load_clean_metadata(self) -> pd.DataFrame | None:
        # Obtain metadata
        self.metadata_df = pd.read_csv(self.metadata_path)
        
        # Auto dropping cols and cleaning based on pipeline initialization
        if self.mode == 'auto':
            self.cols_to_dropped = [cols for cols in self.metadata_df.columns.tolist() if cols not in self.preserved_metadata_cols]
            self.metadata_df = self.metadata_df.drop(columns=self.cols_to_dropped)
            self.metadata_df = self.metadata_df.dropna(subset=[self.antibiotic_col], axis=0)

            # Preserves true and throws false based on boolean masking
            self.metadata_df = self.metadata_df[self.metadata_df[self.antibiotic_col].isin([0,1])]
            self.metadata_df[self.antibiotic_col] = self.metadata_df[self.antibiotic_col].astype('int8')
            self.metadata_df.set_index(self.index_name, inplace=True)
        elif self.mode == 'custom':
            # Belom jadi
            self.metadata_df.drop(columns=self.cols_to_dropped, inplace=True)

        return self.metadata_df
        
    def _merge(self) -> pd.DataFrame | None:
        self.final_df = self.unitig_df.join(self.metadata_df, how="inner")        

        return self.final_df

    def export_result(self) -> None:
        if self.final_df is None:
            raise ValueError("Pipeline haven't started preprocessing. Call preprocess first.")
        
        base_dir = Path( os.getcwd() + "/data")
        base_dir.mkdir(parents=True, exist_ok=True)

        y = self.final_df[self.antibiotic_col]
        X = self.final_df.drop(columns=[self.antibiotic_col])

        y_out = base_dir / f"{self.antibiotic_col}_labels.csv"
        X_out = base_dir / f"{self.antibiotic_col}_features.csv"

        y.to_csv(y_out, sep="\t")
        X.to_csv(X_out, sep="\t")

        print(f"Saved output to : {base_dir}")

    def preprocess(self) -> pd.DataFrame:
        self._load_transpose_rtab()
        self._load_clean_metadata()
        return self._merge()


if __name__ == "__main__":
    pipeline = AMRDataPipeline("./data/azm_sr_gwas_filtered_unitigs.Rtab", "./data/metadata.csv", antibiotic_col='azm_sr', mode='auto')

    test = pipeline.preprocess()
    print(test.shape)
    test.to_csv("test.csv")
    pipeline.export_result()