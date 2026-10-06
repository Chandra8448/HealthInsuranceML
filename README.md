# Health Insurance Premium Prediction (ML + Flask)

A web app like the loan-approval one, but for **health insurance**: users register / log in, enter
age, sex, BMI, children, smoker and region, and a machine learning model predicts the **yearly premium**.
A dashboard shows model quality, feature importance and each user's prediction history.

```
app.py                                   Flask app (login, predict, dashboard)
templates/, static/                      UI
Health_Insurance_Model_Training.ipynb    Train the model in Google Colab
model.pkl, metrics.json                  Produced by the notebook (you add these)
requirements.txt, Procfile, render.yaml  Deployment
```

## Step 1 - Train the model in Google Colab
1. Go to https://colab.research.google.com -> **File -> Upload notebook** -> choose `Health_Insurance_Model_Training.ipynb`.
2. **Runtime -> Run all**.
3. At the end, `model.pkl` and `metrics.json` download to your computer. Copy both into this project folder (next to `app.py`).
4. Note the **scikit-learn version** printed in the first cell and set it in `requirements.txt`, e.g. `scikit-learn==1.6.1`.

If you skip this step the app still runs using a small fallback model, but real predictions need the Colab model.

## Step 2 - Run locally (optional)
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000, register an account, then use Prediction.

## Step 3 - Put it on GitHub
```bash
git init
git add .
git commit -m "Health insurance ML app"
git branch -M main
git remote add origin https://github.com/<your-username>/health-insurance-ml.git
git push -u origin main
```
(Create the empty repo first on github.com -> New repository.)

## Step 4 - Deploy on Render (free)
1. https://render.com -> **New + -> Web Service** -> connect your GitHub repo.
2. Build command: `pip install -r requirements.txt`  |  Start command: `gunicorn app:app`
   (or choose **Blueprint** and Render reads `render.yaml`).
3. Add environment variable `SECRET_KEY` = any long random string.
4. Deploy. Your link will look like `https://health-insurance-ml.onrender.com`.

## Notes
- User accounts are stored in SQLite (`app.db`). On Render's free plan the disk resets on redeploy, so accounts
  are lost. For permanent accounts, switch to a hosted Postgres database.
- The dataset is the public "insurance" dataset (1,338 rows). Predictions are estimates for learning/demo use,
  not real insurance quotes.
