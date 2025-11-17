# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os, json
# from bayes_model import Bayes_Classifier

# APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA_FILE = os.path.join(APP_ROOT, "alldata.txt")
# HISTORY_FILE = os.path.join(APP_ROOT, "history.json")

# app = Flask(__name__)
# CORS(app)

# # Global model
# model = Bayes_Classifier()


# # ---------- Helpers ----------
# def load_data(path):
#     if not os.path.exists(path):
#         return []
#     with open(path, "r", encoding="utf-8") as f:
#         return [l.strip() for l in f if "|" in l]

# def save_file(file, path):
#     file.save(path)

# def save_url_history(url, result):
#     history = []
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, "r", encoding="utf-8") as f:
#             history = json.load(f)
#     # Thêm vào đầu
#     history.insert(0, {"url": url, "result": result})
#     history = history[:50]  # giữ 50 bản ghi
#     with open(HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(history, f, ensure_ascii=False, indent=2)


# # ---------- API: Upload + Train ----------
# @app.route("/upload", methods=["POST"])
# def upload_and_train():
#     if "file" not in request.files:
#         return jsonify({"status": "error", "msg": "file missing"}), 400

#     file = request.files["file"]
#     save_file(file, DATA_FILE)

#     # load + train
#     lines = load_data(DATA_FILE)
#     if not lines:
#         return jsonify({"status": "error", "msg": "no valid data"}), 400

#     model.train(lines)

#     # metrics
#     preds = model.classify_lines(lines)
#     total = len(preds)
#     pos = preds.count("5")
#     neg = preds.count("1")
#     accuracy = round((pos + neg) / total, 4) if total > 0 else 0

#     return jsonify({
#         "status": "ok",
#         "msg": "Training completed",
#         "total": total,
#         "positive": pos,
#         "negative": neg,
#         "accuracy": accuracy
#     })


# # ---------- API: Classify single text ----------
# @app.route("/classify", methods=["POST"])
# def classify_text():
#     data = request.get_json()
#     text = data.get("text", "").strip()
#     if not text:
#         return jsonify({"status": "error", "msg": "empty text"}), 400

#     result = model.classify_text(text)
#     return jsonify({
#         "status": "ok",
#         "label": "positive" if result["label"] == "5" else "negative",
#         "score_pos": result["score_pos"],
#         "score_neg": result["score_neg"]
#     })


# # ---------- API: Classify URL ----------
# @app.route("/classify-url", methods=["POST"])
# def classify_url():
#     data = request.get_json()
#     url = data.get("url", "").strip()
#     if not url:
#         return jsonify({"status": "error", "msg": "empty url"}), 400

#     # Lấy nội dung từ URL (giả sử dùng requests + BeautifulSoup)
#     try:
#         import requests
#         from bs4 import BeautifulSoup
#         resp = requests.get(url, timeout=10)
#         soup = BeautifulSoup(resp.text, "html.parser")
#         paragraphs = soup.find_all("p")
#         content = " ".join(p.get_text() for p in paragraphs)
#         preview = content[:500]  # preview 500 ký tự
#     except Exception as e:
#         return jsonify({"status": "error", "msg": f"Cannot fetch URL: {e}"}), 400

#     result = model.classify_text(content)
#     res_data = {
#         "label": "positive" if result["label"] == "5" else "negative",
#         "score_pos": result["score_pos"],
#         "score_neg": result["score_neg"],
#         "content_preview": preview
#     }

#     save_url_history(url, res_data)
#     return jsonify(res_data)


# # ---------- API: Metrics ----------
# @app.route("/metrics", methods=["GET"])
# def metrics():
#     lines = load_data(DATA_FILE)
#     preds = model.classify_lines(lines)
#     pos = preds.count("5")
#     neg = preds.count("1")
#     return jsonify({
#         "status": "ok",
#         "positive": pos,
#         "negative": neg,
#         "total": len(lines)
#     })


# # ---------- API: History ----------
# @app.route("/history", methods=["GET"])
# def get_history():
#     if not os.path.exists(HISTORY_FILE):
#         return jsonify([])
#     with open(HISTORY_FILE, "r", encoding="utf-8") as f:
#         return jsonify(json.load(f))


# # ---------- Start Server ----------
# if __name__ == "__main__":
#     if os.path.exists(DATA_FILE):
#         model.train(load_data(DATA_FILE))
#         print("Initial model trained")
#     app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json
from bayes_model import Bayes_Classifier

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(APP_ROOT, "alldata.txt")
HISTORY_FILE = os.path.join(APP_ROOT, "history.json")

app = Flask(__name__)
CORS(app)

# Global model
model = Bayes_Classifier()

# ---------- Helpers ----------
def load_data(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if "|" in l]

def save_file(file, path):
    file.save(path)

def save_url_history(url, result):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.insert(0, {"url": url, "result": result})
    history = history[:50]  # giữ 50 bản ghi
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ---------- API: Upload + Train ----------
@app.route("/upload", methods=["POST"])
def upload_and_train():
    if "file" not in request.files:
        return jsonify({"status": "error", "msg": "file missing"}), 400

    file = request.files["file"]
    save_file(file, DATA_FILE)

    lines = load_data(DATA_FILE)
    if not lines:
        return jsonify({"status": "error", "msg": "no valid data"}), 400

    model.train(lines)

    # metrics
    preds = model.classify_lines(lines)
    total = len(preds)
    pos = preds.count("5")
    neg = preds.count("1")
    accuracy = round((pos + neg) / total, 4) if total > 0 else 0

    return jsonify({
        "status": "ok",
        "msg": "Training completed",
        "total": total,
        "positive": pos,
        "negative": neg,
        "accuracy": accuracy
    })

# ---------- API: Classify single text ----------
@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"status": "error", "msg": "empty text"}), 400

    result = model.classify_text(text)
    return jsonify({
        "status": "ok",
        "label": "positive" if result["label"] == "5" else "negative",
        "score_pos": result["score_pos"],
        "score_neg": result["score_neg"]
    })

# ---------- API: Classify URL ----------
@app.route("/classify-url", methods=["POST"])
def classify_url():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"status": "error", "msg": "empty url"}), 400

    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text() for p in paragraphs)
        preview = content[:500]
    except Exception as e:
        return jsonify({"status": "error", "msg": f"Cannot fetch URL: {e}"}), 400

    result = model.classify_text(content)
    res_data = {
        "label": "positive" if result["label"] == "5" else "negative",
        "score_pos": result["score_pos"],
        "score_neg": result["score_neg"],
        "content_preview": preview
    }

    save_url_history(url, res_data)
    return jsonify(res_data)

# ---------- API: Metrics ----------
@app.route("/metrics", methods=["GET"])
def metrics():
    lines = load_data(DATA_FILE)
    preds = model.classify_lines(lines)
    pos = preds.count("5")
    neg = preds.count("1")
    return jsonify({
        "status": "ok",
        "positive": pos,
        "negative": neg,
        "total": len(lines)
    })

# ---------- API: History ----------
@app.route("/history", methods=["GET"])
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify([])
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

# ---------- Start Server ----------
if __name__ == "__main__":
    if os.path.exists(DATA_FILE):
        model.train(load_data(DATA_FILE))
        print("Initial model trained")
    app.run(host="0.0.0.0", port=5000, debug=True)
