import json
import os
import sqlite3
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import (Flask, flash, g, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
DB_PATH = os.path.join(BASE_DIR, "app.db")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]


# ---------------------------------------------------------------- model
def build_fallback_model():
    """Used only if model.pkl is missing/incompatible. Trains on synthetic data
    so the site never crashes. Replace by training in the Colab notebook."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(42)
    n = 1500
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "sex": rng.choice(["male", "female"], n),
        "bmi": rng.normal(30, 6, n).clip(15, 50),
        "children": rng.integers(0, 5, n),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], n),
    })
    y = (2000 + df.age * 250 + (df.bmi - 25).clip(0) * 350 + df.children * 450
         + (df.smoker == "yes") * 23000 + rng.normal(0, 2500, n)).clip(1100)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])],
        remainder="passthrough")
    model = Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(random_state=42))])
    model.fit(df[FEATURES], y)
    return model


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # missing file or sklearn version mismatch
        print("Could not load model.pkl, using fallback model:", exc)
        return build_fallback_model()


def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"model": "Fallback model", "r2": None, "mae": None, "rmse": None,
                "importances": {}, "rows": None}


model = load_model()


# ------------------------------------------------------------------- db
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age INTEGER, sex TEXT, bmi REAL, children INTEGER,
            smoker TEXT, region TEXT, premium REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    db.commit()
    db.close()


init_db()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


# --------------------------------------------------------------- routes
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if not name or not email or len(password) < 6:
            flash("Fill all fields; password needs at least 6 characters.", "error")
        else:
            try:
                db = get_db()
                db.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                           (name, email, generate_password_hash(password)))
                db.commit()
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That email is already registered.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        user = get_db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form["password"]):
            session.clear()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            return redirect(request.args.get("next") or url_for("predict"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    if request.method == "POST":
        try:
            row = {
                "age": int(request.form["age"]),
                "sex": request.form["sex"],
                "bmi": float(request.form["bmi"]),
                "children": int(request.form["children"]),
                "smoker": request.form["smoker"],
                "region": request.form["region"],
            }
            if not (18 <= row["age"] <= 100) or not (10 <= row["bmi"] <= 70) \
                    or not (0 <= row["children"] <= 10):
                raise ValueError
            premium = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
            premium = max(premium, 0)
            db = get_db()
            db.execute("""INSERT INTO predictions(user_id,age,sex,bmi,children,smoker,region,premium)
                          VALUES(?,?,?,?,?,?,?,?)""",
                       (session["user_id"], row["age"], row["sex"], row["bmi"],
                        row["children"], row["smoker"], row["region"], premium))
            db.commit()
            result = {"premium": premium, "monthly": premium / 12,
                      "risk": "High" if premium > 20000 else "Medium" if premium > 10000 else "Low",
                      "inputs": row}
        except (ValueError, KeyError):
            flash("Please enter valid values (age 18-100, BMI 10-70, children 0-10).", "error")
    return render_template("predict.html", result=result)


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    history = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 20",
        (session["user_id"],)).fetchall()
    metrics = load_metrics()
    return render_template("dashboard.html", history=history, metrics=metrics,
                           chart_labels=[f"#{h['id']}" for h in reversed(history)],
                           chart_values=[round(h["premium"], 2) for h in reversed(history)])


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
