
import logging

import knime.extension as knext
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

LOGGER = logging.getLogger(__name__)

ICONS_PATH = "icons/"


multicollinearity_category = knext.category(
    path="/community",
    level_id="multicollinearity",
    name="Multicollinearity Tools",
    description="Nodes for multicollinearity and influence diagnostics.",
    icon=ICONS_PATH + "viftol.png",
)


@knext.node(
    name="VIF & TOL",  
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=ICONS_PATH + "viftol.png",
    category=multicollinearity_category,
)
@knext.input_table(
    name="Input Data",
    description="Any input table. Select numeric columns to compute VIF/TOL.",
)
@knext.output_table(
    name="VIF/TOL Table",
    description="Feature-level VIF and TOL (1/VIF).",
)
class VIFNode:
    """Computes VIF and TOL for selected numeric columns (or all numeric if none selected)."""

    selected_columns = knext.MultiColumnParameter(
        label="Columns for VIF/TOL",
        description="Select predictors. If empty, all numeric columns are used.",
        port_index=0,
    )

    def configure(self, configure_context, input_schema_1):
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Feature"),
                knext.Column(knext.double(), "VIF"),
                knext.Column(knext.double(), "TOL"),
            ]
        )

    def execute(self, exec_context, input_1):
        df = input_1.to_pandas()

        chosen = list(self.selected_columns) if self.selected_columns else []
        if len(chosen) == 0:
            chosen = df.select_dtypes(include=["number"]).columns.tolist()

        if len(chosen) < 2:
            raise ValueError(
                "VIF/TOL requires at least 2 numeric columns. "
                "Select ≥2 columns (or ensure the input has ≥2 numeric columns)."
            )

        X = df[chosen].copy().dropna()

        if X.shape[0] < 2:
            raise ValueError("Not enough rows after removing missing values (dropna).")

        # remove constant columns (variance 0)
        constant_cols = X.nunique(dropna=True)[lambda s: s <= 1].index.tolist()
        if constant_cols:
            X = X.drop(columns=constant_cols)

        if X.shape[1] < 2:
            raise ValueError(
                "After removing constant columns, fewer than 2 columns remain. "
                "Select different columns."
            )

        X_const = sm.add_constant(X, has_constant="add")

        rows = []
        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue
            try:
                vif = float(variance_inflation_factor(X_const.values, i))
            except Exception:
                vif = float("inf")
            tol = 1.0 / vif if np.isfinite(vif) and vif != 0 else 0.0
            rows.append((col, vif, tol))

        out = pd.DataFrame(rows, columns=["Feature", "VIF", "TOL"])
        return knext.Table.from_pandas(out)


@knext.node(
    name="Eigenvalues & Condition Index (CN)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=ICONS_PATH + "eigenvalues.png",
    category=multicollinearity_category,
)
@knext.input_table(
    name="Input Data",
    description="Select ONLY the predictor variables (independent variables).",
)
@knext.output_table(
    name="Eigenvalues/CN Table",
    description="Standard collinearity diagnostics (Dimension, Eigenvalue, Condition Index).",
)
class EigenvaluesCNNode:
    selected_columns = knext.MultiColumnParameter(
        label="Predictor Columns",
        description="Select the independent variables. DO NOT select the target (sales).",
        port_index=0,
    )

    def configure(self, configure_context, input_schema_1):
        
        return knext.Schema.from_columns(
            [
                knext.Column(knext.int64(), "Dimension"),
                knext.Column(knext.double(), "Eigenvalue"),
                knext.Column(knext.double(), "Condition_Index"),
            ]
        )

    def execute(self, exec_context, input_1):
        df = input_1.to_pandas()
        
        
        chosen = list(self.selected_columns) if self.selected_columns else df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[chosen].copy().dropna()



        X.insert(0, 'Intercept', 1.0)


        X_vals = X.values
        norms = np.sqrt(np.sum(X_vals**2, axis=0))
        X_scaled = X_vals / norms


        xtx = np.dot(X_scaled.T, X_scaled)
        eigvals = np.linalg.eigvals(xtx)
        

        eigvals = np.sort(eigvals)[::-1]


        max_eig = eigvals[0]
        cond_index = np.sqrt(max_eig / eigvals)

        out = pd.DataFrame({
            "Dimension": [i + 1 for i in range(len(eigvals))],
            "Eigenvalue": eigvals.astype(float),
            "Condition_Index": cond_index.astype(float)
        })

        return knext.Table.from_pandas(out)

@knext.node(
    name="DFFITS (Influence Diagnostics)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=ICONS_PATH + "dffits.png",
    category=multicollinearity_category,
)
@knext.input_table(
    name="Input Data",
    description="Any input table. Select target and predictor columns for OLS and DFFITS computation.",
)
@knext.output_table(
    name="DFFITS Table",
    description="Observation-level DFFITS, threshold and flag (sorted by absolute DFFITS).",
)
class DFFITSNode:
    """Fits an OLS model and computes DFFITS for each observation."""

    target_column = knext.ColumnParameter(
        label="Target (dependent variable)",
        description="Select the dependent variable (numeric).",
        port_index=0,
    )

    feature_columns = knext.MultiColumnParameter(
        label="Predictors (independent variables)",
        description="Select predictor columns. If empty, all numeric columns except target are used.",
        port_index=0,
    )

    def configure(self, configure_context, input_schema_1):
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Row_Index"),
                knext.Column(knext.double(), "DFFITS_Score"),
                knext.Column(knext.double(), "Abs_DFFITS"),
                knext.Column(knext.double(), "Critical_Threshold"),
                knext.Column(knext.string(), "Influence_Status"),
            ]
        )

    def execute(self, exec_context, input_1):
        df = input_1.to_pandas()

        out_columns = [
            "Row_Index",
            "DFFITS_Score",
            "Abs_DFFITS",
            "Critical_Threshold",
            "Influence_Status",
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        target = self.target_column
        if target is None or target not in df.columns:
            if not numeric_cols:
                raise ValueError("No numeric columns available to use as target.")
            target = numeric_cols[0]

        raw_features = list(self.feature_columns) if self.feature_columns else []
        features = [
            col for col in raw_features
            if col is not None and col in df.columns and col != target
        ]

        if not features:
            features = [col for col in numeric_cols if col != target]

        selected_cols = [target] + features
        data = df[selected_cols].copy()

        for col in selected_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna()

        if data.empty:
            return knext.Table.from_pandas(pd.DataFrame(columns=out_columns))

        y = data[target]

        if features:
            X = data[features]
        else:
            X = pd.DataFrame(index=data.index)

        X_const = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X_const).fit()
        influence = model.get_influence()
        dffits_values = influence.dffits[0]

        n = len(data)
        p = X_const.shape[1]
        threshold = 2 * np.sqrt(p / n) if n > 0 else np.nan

        out = pd.DataFrame({
            "Row_Index": data.index.astype(str),
            "DFFITS_Score": dffits_values,
            "Abs_DFFITS": np.abs(dffits_values),
            "Critical_Threshold": threshold,
        })

        out["Influence_Status"] = np.where(
            out["Abs_DFFITS"] > threshold,
            "High Influence",
            "Normal"
        )

        out = out.sort_values("Abs_DFFITS", ascending=False).reset_index(drop=True)

        return knext.Table.from_pandas(out)



@knext.node(
    name="DFBETAS (Influence Diagnostics)",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path=ICONS_PATH + "dfbetas.png",
    category=multicollinearity_category,
)
@knext.input_table(
    name="Input Data",
    description="Any input table. Select target and predictor columns for OLS and DFBETAS computation.",
)
@knext.output_table(
    name="DFBETAS Table",
    description="Observation-parameter DFBETA values, threshold and flag (sorted by absolute DFBETA).",
)
class DFBETASNode:
    """Fits an OLS model and computes DFBETAS for each observation and parameter."""

    target_column = knext.ColumnParameter(
        label="Target (dependent variable)",
        description="Select the dependent variable (numeric).",
        port_index=0,
    )

    feature_columns = knext.MultiColumnParameter(
        label="Predictors (independent variables)",
        description="Select predictor columns. If empty, all numeric columns except target are used.",
        port_index=0,
    )

    def configure(self, configure_context, input_schema_1):
        return knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Original_Row_ID"),
                knext.Column(knext.string(), "Variable_Name"),
                knext.Column(knext.double(), "DFBETA_Value"),
                knext.Column(knext.double(), "Abs_DFBETA"),
                knext.Column(knext.double(), "Threshold_Limit"),
                knext.Column(knext.string(), "Influence_Flag"),
            ]
        )

    def execute(self, exec_context, input_1):
        df = input_1.to_pandas()

        out_columns = [
            "Original_Row_ID",
            "Variable_Name",
            "DFBETA_Value",
            "Abs_DFBETA",
            "Threshold_Limit",
            "Influence_Flag",
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        target = self.target_column
        if target is None or target not in df.columns:
            if not numeric_cols:
                raise ValueError("No numeric columns available to use as target.")
            target = numeric_cols[0]

        raw_features = list(self.feature_columns) if self.feature_columns else []
        features = [
            col for col in raw_features
            if col is not None and col in df.columns and col != target
        ]

        if not features:
            features = [col for col in numeric_cols if col != target]

        selected_cols = [target] + features
        data = df[selected_cols].copy()

        for col in selected_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna()

        if data.empty:
            return knext.Table.from_pandas(pd.DataFrame(columns=out_columns))

        if len(features) == 0:
            raise ValueError("At least one predictor column is required for DFBETAS.")

        if data.shape[0] < len(features) + 2:
            raise ValueError("Insufficient data points for the number of predictors.")

        y = data[target]
        X = data[features]

        X_const = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_const).fit()

        influence = model.get_influence()
        dfbetas_matrix = influence.dfbetas
        param_names = list(model.params.index)

        dfbetas_df = pd.DataFrame(dfbetas_matrix, columns=param_names)
        dfbetas_df.insert(0, "Original_Row_ID", data.index.astype(str))

        out = dfbetas_df.melt(
            id_vars="Original_Row_ID",
            var_name="Variable_Name",
            value_name="DFBETA_Value"
        )

        n = len(data)
        calc_threshold = 2 / np.sqrt(n)

        out["Abs_DFBETA"] = out["DFBETA_Value"].abs()
        out["Threshold_Limit"] = calc_threshold
        out["Influence_Flag"] = np.where(
            out["Abs_DFBETA"] > calc_threshold,
            "HIGH",
            "Normal"
        )

        out = out.sort_values("Abs_DFBETA", ascending=False).reset_index(drop=True)

        return knext.Table.from_pandas(out)
