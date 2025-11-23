## 🖥️ Streamlit App

### Tab 1 — Suspicious Articles Explorer
This tab shows articles that are flagged as **suspicious** based on:
- **Anomaly detection** (IsolationForest → `anomaly_label == -1`)
- **Location mismatch** (predicted location from content ≠ reported location)

It displays:
- Heading  
- Full article content  
- Reported Location  
- Extracted Location  
- Predicted Location  
- Anomaly Score  
- Suspicious Flag (`is_suspicious`)

---

### Tab 2 — Analyse New Article

Here, the user can:
- Enter **Heading**
- Enter **Full Article Text**
- (Optional) Enter **Reported Location** (e.g., `US`, `India`, `Pakistan`)

The app will:
- **Predict location** from the content using Logistic Regression  
- **Compute anomaly score** using IsolationForest  
- Show message:
  - whether the article is **normal / anomalous**
  - whether **reported location matches** predicted location or not

---

## 🤖 ML Model Summary

| Feature             | Details                             |
|--------------------|-------------------------------------|
| Vectorizer         | TF-IDF (3000 features)              |
| Anomaly Detection  | IsolationForest (contamination=0.05)|
| Location Model     | Logistic Regression (max_iter=1000) |
| Output Suspicious  | `anomaly OR location_mismatch`      |

---


git clone https://github.com/Ganesh-alt1807/News-Anomaly-Detection-Project-Ganesh.git
cd News-Anomaly-Detection-Project-Ganesh
pip install -r requirements.txt



**Deployed app link**

https://news-anomaly-detection-project-ganesh-gw5khzsrbzpfu2qtdmajw4.streamlit.app/





👤 Author
Name	  Track
Ganesh	MDTM46B (Final Project)
