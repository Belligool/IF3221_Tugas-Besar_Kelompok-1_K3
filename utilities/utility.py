# Preprocessor class that will be the main pipeline for our project
# This pipeline is limited to processing just 3 drugs (azm, cip, cfx)

import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb

from pathlib import Path
from typing import Literal
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from scipy.spatial.distance import squareform, pdist
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor , DistanceMatrix
from Bio import Phylo

class AMRDataPipeline:
    def __init__(self, rtab_path: str, 
                 metadata_path: str, 
                 antibiotic_col: Literal['cfx','azm','cip'],
                 mode: Literal['auto','preserve'],
                 selection_mode: Literal['rf','chi','xgb'] = None,
                 alpha_threshold: float = 0.05,
                 n_features: int = 200,
                 index_name: str = "Sample_ID",
                 ):
        self.rtab_path = Path(rtab_path)
        self.metadata_path = Path(metadata_path)
        self.antibiotic_col = antibiotic_col
        self.index_name = index_name
        self.mode = mode
        self.selection_mode = selection_mode

        self.unitig_df = None
        self.metadata_df = None
        self.final_df = None
        self.selected_unitigs = None
        self.preserved_metadata_cols = [f'{antibiotic_col}_sr', index_name]

        self._validate_pipeline(n_features=n_features, alpha_threshold=alpha_threshold)

    def _validate_pipeline(self, n_features: int = None, alpha_threshold: float = None) -> None:
        if self.selection_mode is not None:
            if self.selection_mode == 'rf' or self.selection_mode == 'xgb':
                if n_features is None:
                    raise ValueError("When tree based feature selection is selected the number of features must be above 0.")
                else:
                    self.n_features = n_features
            elif self.selection_mode == 'chi':
                if alpha_threshold is None:
                    raise ValueError("When statistical feature selection is applied the significance must be above 0.")
                else:
                    self.alpha_threshold = alpha_threshold
        else:
            return

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

        elif self.mode == 'preserve':

            # Clean MIC cols
            mic_cols = [c for c in self.metadata_df.columns if c.endswith("_mic") and not c.startswith("log2_")]
            for col in mic_cols:
                self.metadata_df[col] = self.metadata_df[col].astype(str).str.replace(r'[<>=]', '', regex=True)
                self.metadata_df[col] = pd.to_numeric(self.metadata_df[col], errors='coerce')

            # Clean Beta.lactamase
            self.metadata_df['Beta.lactamase'] = self.metadata_df['Beta.lactamase'].astype(str).str.replace(r'r[<>=]','', regex=True)
            self.metadata_df['Beta.lactamase'] = pd.to_numeric(self.metadata_df['Beta.lactamase'], errors='coerce')

            # Filter columns
            preserved_cols = []
            variants = [f'{self.antibiotic_col}_mic',f'log2_{self.antibiotic_col}_mic',f'{self.antibiotic_col}_sr']
            preserved_cols.extend([c for c in variants])
            preserved_cols.extend(['Year','Country','Continent','Group',self.index_name,'Beta.lactamase'])

            self.metadata_df = self.metadata_df[preserved_cols]
            
        # Drop any row that doesnt have a value for the target variable [1,0]
        self.metadata_df = self.metadata_df.dropna(subset=[f'{self.antibiotic_col}_sr'], axis=0)

        # Preserves true and throws false based on boolean masking
        self.metadata_df = self.metadata_df[self.metadata_df[f'{self.antibiotic_col}_sr'].isin([0,1])]
        self.metadata_df[f'{self.antibiotic_col}_sr'] = self.metadata_df[f'{self.antibiotic_col}_sr'].astype('int8')
        self.metadata_df.set_index(self.index_name, inplace=True)

        return self.metadata_df
    
    def _rf_feature_selection(self) -> list[str]:
        X_temp = self.unitig_df.join(self.metadata_df, how="inner")  
        y_temp = X_temp[f"{self.antibiotic_col}_sr"]
        X_temp = X_temp.drop(columns=[f"{self.antibiotic_col}_sr"] + self.metadata_df.columns.tolist(), errors='ignore')


        selector = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        selector.fit(X_temp, y_temp)

        importances = pd.Series(selector.feature_importances_, index=X_temp.columns)
        self.selected_unitigs = importances.nlargest(self.n_features).index.to_list()

        return self.selected_unitigs
    
    def _xgb_feature_selection(self) -> list[str]:
        X_temp = self.unitig_df.join(self.metadata_df, how="inner")  
        y_temp = X_temp[f"{self.antibiotic_col}_sr"]
        X_temp = X_temp.drop(columns=[f"{self.antibiotic_col}_sr"] + self.metadata_df.columns.tolist(), errors='ignore')


        selector = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42)
        selector.fit(X_temp, y_temp)

        importances = pd.Series(selector.feature_importances_, index=X_temp.columns)
        self.selected_unitigs = importances.nlargest(self.n_features).index.to_list()

        return self.selected_unitigs
    
    def _chi_feature_selection(self) -> list[str]:
        X_temp = self.unitig_df.join(self.metadata_df, how="inner")
        y_temp = X_temp[f"{self.antibiotic_col}_sr"]
        X_temp = X_temp.drop(columns=[f"{self.antibiotic_col}_sr"] + self.metadata_df.columns.to_list(), errors='ignore')

        num_test = X_temp.shape[1]

        # Bonforreni Correction to make significane more appropriate and doesnt accumulate
        adjusted_alpha = self.alpha_threshold / num_test

        passed_unitigs = []

        for unitig in X_temp.columns:
            vector = X_temp[unitig].values
            contigency_table = pd.crosstab(vector, y_temp)

            # Unitig is present across all samples
            if contigency_table.shape != (2,2):
                continue
            
            # Yates correction
            chi_2_stat, p_val, dof, expected = chi2_contingency(contigency_table, correction=True)

            if p_val < adjusted_alpha:
                passed_unitigs.append(unitig)

        self.selected_unitigs = passed_unitigs
        if len(passed_unitigs) == 0:
            raise ValueError("Zero unitigs passed during statistical threshold using chi square.")
        
        return self.selected_unitigs

    def _merge(self) -> pd.DataFrame | None:
        self.final_df = self.unitig_df.join(self.metadata_df, how="inner")        

        return self.final_df

    def export_result(self) -> None:
        if self.final_df is None:
            raise ValueError("Pipeline haven't started preprocessing. Call preprocess first.")
        
        base_dir = Path(os.getcwd() + "/data")
        base_dir.mkdir(parents=True, exist_ok=True)

        y = self.final_df[f'{self.antibiotic_col}_sr']
        X = self.final_df.drop(columns=[f'{self.antibiotic_col}_sr'])

        y_out = base_dir / f"{self.antibiotic_col}_real_labels.csv"
        X_out = base_dir / f"{self.antibiotic_col}_real_features.csv"

        y.to_csv(y_out, sep="\t")
        X.to_csv(X_out, sep="\t")

        print(f"Saved output to : {base_dir}")

    def preprocess(self) -> pd.DataFrame:
        self._load_transpose_rtab()
        self._load_clean_metadata()
        top_unitigs = []
        if self.selection_mode is not None:
            if self.selection_mode == 'rf':
                top_unitigs = self._rf_feature_selection()
            elif self.selection_mode == 'chi':
                top_unitigs = self._chi_feature_selection()
            elif self.selection_mode == 'xgb':
                top_unitigs = self._xgb_feature_selection()

        if len(top_unitigs) == 0:
            columns_to_keep = self.unitig_df.columns.to_list() + self.metadata_df.columns.to_list()
        else:
            columns_to_keep = top_unitigs + self.metadata_df.columns.to_list()
            
        self.final_df = self._merge()
        
        if self.final_df is None:
            self.final_df = self.unitig_df.join(self.metadata_df, how="inner")
        self.final_df = self.final_df[[c for c in columns_to_keep if c in self.final_df.columns]]
        return self.final_df

    def visualize_unitigs_over_time(self, unitigs: int = 10) -> None:
        if self.selected_unitigs is None:
            raise ValueError("Pipeline haven't started preprocessing. Call preprocess first.")

        unitigs_sequences = self.selected_unitigs[:unitigs]
        unitig_time_df = self.unitig_df[unitigs_sequences].join(self.metadata_df[['Year',f'{self.antibiotic_col}_sr']], how="inner")
        unitig_filtered_resistance_df = unitig_time_df[unitig_time_df[f'{self.antibiotic_col}_sr'] == 1]
        unitig_filtered_succeptible_df = unitig_time_df[unitig_time_df[f'{self.antibiotic_col}_sr'] == 0]

        self.unitig_mean_resistance = unitig_filtered_resistance_df.groupby('Year').mean().sort_index()
        self.unitig_mean_succeptible = unitig_filtered_succeptible_df.groupby('Year').mean().sort_index()

        plt.figure(figsize=(12,6))
        plt.title("Unitig Sequence Trajectory of Gonorhea")

        for unitig in unitigs_sequences:
            line, = plt.plot(self.unitig_mean_resistance.index, self.unitig_mean_resistance[unitig], marker='o', linestyle='-', alpha=1.0, label=f"{unitig[:10]}... (Resistant)")
            shared_color = line.get_color()
            plt.plot(self.unitig_mean_succeptible.index, self.unitig_mean_succeptible[unitig], marker='x', linestyle='--', alpha=0.3, color=shared_color,label=f"{unitig[:10]}... (Susceptible)")

        plt.grid(True, alpha=0.5, linestyle='--')
        plt.ylabel("Unitig Frequency")
        plt.xlabel("Timeline")
        plt.tight_layout()
        plt.show()

    def visualize_phylo_tree(self, unitigs: int = 10, mode: Literal['ascii','plt'] = 'ascii', type: Literal['succeptible','resistant'] = 'resistant') -> None:
        biopython_matrix = []
        
        if type == 'resistant':
            df = self.final_df[self.final_df[f'{self.antibiotic_col}_sr'] == 1].iloc[:unitigs]
        elif type == 'succeptible':
            df = self.final_df[self.final_df[f'{self.antibiotic_col}_sr'] == 0].iloc[:unitigs]
        else:
            raise ValueError(f"{type} is not supported.")

        names = df.index.tolist()

        # Jaccard distance calculation
        dense_matrix = squareform(pdist(df.values, metric='jaccard'))

        # Transform to adhere Biopython's Triangular matrix for Distance Matrix 
        for i in range (len(names)):
            row = list(dense_matrix[i, :i+1])
            biopython_matrix.append(row)

        dm = DistanceMatrix(names=names, matrix=biopython_matrix)
        constructor = DistanceTreeConstructor()
        tree = constructor.nj(dm)

        if mode == 'plt':
            Phylo.draw(tree)
            plt.show()
        elif mode == 'ascii':
            Phylo.draw_ascii(tree)
        else:
            raise ValueError(f"{mode} is not supported.")

if __name__ == "__main__":
    pipeline = AMRDataPipeline("./data/cip_sr_gwas_filtered_unitigs.Rtab", 
                               "./data/metadata.csv", 
                               antibiotic_col='cip', 
                               mode='auto', 
                               selection_mode=None,
                               n_features=500)

    test = pipeline.preprocess()
    print(test.shape)
    test.to_csv("test.csv")
    pipeline.export_result()

    # pipeline.visualize_alignment_matrix()
    pipeline.visualize_phylo_tree(15, type='resistant')

    # pipeline.unitig_mean_resistance.to_csv("test.csv")
