
import os
import time
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from flask import (
    Flask, request, render_template,
    redirect, url_for, flash, jsonify
)
from werkzeug.utils import secure_filename

from inference import predict_image

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY", "change-this-before-deployment"
)

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "VisionGuard_MobileNetV2")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured.")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["MAX_CONTENT_LENGTH"] = (
    int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
)


def get_db():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def initialize_database():
    with get_db() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id BIGSERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms DOUBLE PRECISION NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                ai_probability DOUBLE PRECISION NOT NULL,
                model_name TEXT NOT NULL
            )
        """)


@app.before_request
def start_request_timer():
    request.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    try:
        duration_ms = (
            time.perf_counter() - request.start_time
        ) * 1000

        with get_db() as connection:
            connection.execute("""
                INSERT INTO requests
                (timestamp, method, path, status_code, duration_ms)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                datetime.now(timezone.utc).isoformat(),
                request.method,
                request.path,
                response.status_code,
                duration_ms
            ))

    except Exception:
        app.logger.exception("Could not log HTTP request")

    return response


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    if "image" not in request.files:
        flash("Please choose an image to upload.")
        return redirect(url_for("index"))

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        flash("Please choose an image first.")
        return redirect(url_for("index"))

    safe_filename = secure_filename(uploaded_file.filename)

    if not safe_filename or not allowed_file(safe_filename):
        flash("Unsupported file type. Upload JPG, JPEG, PNG, or WEBP.")
        return redirect(url_for("index"))

    try:
        prediction = predict_image(uploaded_file)

    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    except Exception:
        app.logger.exception("Prediction failed")
        flash("We couldn't process this image. Please try another.")
        return redirect(url_for("index"))

    with get_db() as connection:
        connection.execute("""
            INSERT INTO predictions
            (timestamp, filename, predicted_label, confidence,
             ai_probability, model_name)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            datetime.now(timezone.utc).isoformat(),
            safe_filename,
            prediction["label"],
            prediction["confidence"],
            prediction["ai_probability"],
            MODEL_NAME
        ))

    return render_template(
        "result.html",
        filename=safe_filename,
        prediction=prediction
    )


@app.route("/history")
def history():
    with get_db() as connection:
        records = connection.execute("""
            SELECT id, timestamp, filename, predicted_label,
                   confidence, ai_probability, model_name
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()

    return render_template("history.html", records=records)


@app.route("/health")
def health():
    try:
        with get_db() as connection:
            connection.execute("SELECT 1")

        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "database": "connected"
        }), 200

    except Exception:
        app.logger.exception("Health check failed")
        return jsonify({
            "status": "unhealthy",
            "database": "unavailable"
        }), 503


@app.errorhandler(413)
def file_too_large(error):
    return (
        "Image too large. Please upload an image smaller than "
        f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB.",
        413
    )


@app.errorhandler(404)
def page_not_found(error):
    return "Page not found. Please return to the home page.", 404


initialize_database()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False
    )
