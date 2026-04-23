# 🌍 Smart Environmental Monitoring System

A Machine Learning project to **predict Air Quality Index (AQI)** and classify pollution levels using environmental sensor data.

---

## 📌 Objective

Build an ML-based system to:
- Predict **AQI** from environmental features
- Classify pollution levels as **Good / Moderate / Poor**

---

## 📊 Features Used

| Feature | Description |
|--------|-------------|
| PM2.5 | Fine particulate matter |
| PM10  | Coarse particulate matter |
| NO₂   | Nitrogen Dioxide |
| SO₂   | Sulphur Dioxide |
| CO    | Carbon Monoxide |
| O₃    | Ozone |

---

## 🤖 Models Trained

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 📁 Project Structure

```
smart-environmental-monitoring/
│
├── main.py               # Main Python script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smart-environmental-monitoring.git
cd smart-environmental-monitoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python main.py
```

---

## 📈 Output

- R² Score comparison of all 3 models
- PM2.5 vs AQI scatter plot
- AQI Category distribution chart
- Sample AQI prediction

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
