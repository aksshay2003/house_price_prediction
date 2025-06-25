# 🏠 House Price Prediction - End-to-End ML Pipeline

This project demonstrates a complete MLOps-driven machine learning pipeline for predicting house prices using structured data. It includes modular pipeline orchestration, experiment tracking, version control, and automated monitoring—all designed to reflect real-world, production-grade ML systems.

---

## 🚀 Project Highlights

- 📌 **Objective**: Predict house prices using features such as location, size, number of rooms, and amenities.
- 🧱 **Pipeline Orchestration**: Managed using **ZenML** to build clean, modular, and reproducible pipelines.
- 📊 **Experiment Tracking**: Integrated **MLflow** for logging parameters, metrics, artifacts, and models.
- 🔄 **Workflow Automation**: Used **Prefect** for automating and scheduling pipeline runs.
- 📁 **Data Version Control**: Leveraged **DVC** to track datasets and model files effectively.
- 🧪 **Modeling**: Applied regression models with hyperparameter tuning and evaluation metrics like RMSE and R².
- 📈 **Monitoring**: Included model performance tracking to detect drift and retrain if needed.
- 🐳 **Containerization**: Dockerized the project for isolated and portable development environments.
- ⚙️ **CI/CD Integration**: Planned integration with **Jenkins** to enable seamless model updates and testing.

---

## 🛠️ Technologies Used

- **Languages & Frameworks**: Python, Jupyter Notebook  
- **Tools**: ZenML, MLflow, DVC, Prefect, Docker, Jenkins  
- **Cloud**: AWS (for scalability and storage)  
- **Version Control**: Git + GitHub  
- **Frontend (optional)**: Node.js, React.js (for visualizing predictions)

---
```
## 📂 Project Structure

house_price_prediction/
├── analysis/ # Exploratory Data Analysis notebooks
├── extracted_data/ # Raw and cleaned datasets
├── flow/ # Prefect flow definitions
├── pipelines/ # ZenML pipeline definitions
├── prices-predictor-system/ # Core model logic and helper scripts
├── src/ # Feature engineering, preprocessing, utils
├── steps/ # ZenML steps for each stage in pipeline
├── run_pipeline.py # Main entry point to trigger the pipeline
├── mlflow_deploy.py # MLflow integration and tracking script
├── perfect.py # Prefect job configuration
├── run_deployment.py # Pipeline deployment script
├── ingest_data.py # Initial raw data ingestion
├── .dvc/ # DVC configuration and metadata
├── .dvcignore
├── .gitignore
└── README.md


```
---

## ⚙️ How to Run

1. Clone the repository
git clone https://github.com/aksshay2003/house_price_prediction.git
cd house_price_prediction

2.Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # or use venv\Scripts\activate on Windows
pip install -r requirements.txt

3.Set up DVC and MLflow
dvc pull
mlflow ui

4.Run the ML pipeline
python run_pipeline.py

🔍 MLflow Tracking Dashboard
Use the MLflow UI to:
Log hyperparameters, metrics, models, and visualizations
Compare different experiments
Register and manage production-ready models

Launch with:
mlflow ui
Then open: http://127.0.0.1:5000

Why This Project Matters
This project combines clean pipeline design with robust experiment tracking and monitoring, providing hands-on experience with building maintainable and scalable ML systems. It reflects industry-grade practices in MLOps and emphasizes traceability, reproducibility, and automation—making it ideal for production-oriented AI workflows.
