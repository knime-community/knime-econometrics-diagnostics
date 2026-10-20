# KNIME Econometric Diagnostics Extension

This KNIME Extension provides custom Python-based nodes designed for **Multiple Linear Regression (OLS) Diagnostics**, bridging low-code data analytics with advanced econometrics.

Developed as part of a Master's Thesis research at the **Department of Economics, University of Thessaly, Volos, Greece**.

---

## 🛠️ Included Features & Diagnostics

* **Multicollinearity Assessment:**
  * **VIF & Tolerance:** Automated calculation of Variance Inflation Factors and Tolerance metrics.
  * **Eigenvalues & Condition Index (CN):** Severe multicollinearity detection based on matrix condition thresholds.

* **Influence & Outlier Analysis:**
  * **DFFITS Analysis:** Identifies influential data points affecting model fit.
  * **DFBETAS Analysis:** Measures the impact of individual observations on specific regression coefficients.

* **Cross-Platform Validation:**
  * Computations are internally executed using Python (`statsmodels`, `knime.api`) and fully cross-validated for exact mathematical alignment against **IBM SPSS**.

---

## 🏛️ Academic Context & Peer Review

Portions of this research and package design have been peer-reviewed and presented at:
* **The 22nd International Conference on Mobile Learning** (Zagreb)
* **Panhellenic Statistics Conference** by the Greek Statistical Institute (ESI)
* **Greek Mathematical Society Conference**

**Academic Supervisor:** Assoc. Professor Dr. Kiriaki Tsilika

---

## 📄 License
This project is open-source and licensed under the MIT License.
