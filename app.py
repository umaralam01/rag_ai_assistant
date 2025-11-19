from flask_sqlalchemy import SQLAlchemy # Flask ke andar database ko handle karne ke liye ORM (Object Relational Mapper).
from datetime import datetime
from PIL import Image
import pytesseract
import os
import json
import fitz  # PyMuPDF
import whisper
import chromadb
import requests
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer

# ======================================================
#  Merge helper (5 chunks -> 1 chunk with labels)
# ======================================================
def merge_chunks_with_labels(chunks, group_size=5, is_pdf=True):
    merged = []

    for i in range(0, len(chunks), group_size):
        group = chunks[i:i+group_size]
        final_text = ""

        for item in group:
            if is_pdf:
                final_text += f"[Page {item['page']}] {item['text']}\n\n"
            else:
                final_text += f"[Segment {item['segment']} | {item['start']}s → {item['end']}s] {item['text']}\n\n"

        merged.append({"text": final_text.strip()})

    return merged


# ======================================================
#  APP SETUP
# ======================================================
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ========= DATABASE MODELS =========

class UploadLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    filetype = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class QueryLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


os.makedirs("uploads", exist_ok=True)
os.makedirs("jsons", exist_ok=True)


# ======================================================
#  LOAD MODELS
# ======================================================
print("🚀 Loading models... please wait...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = SentenceTransformer('clip-ViT-B-32')

whisper_model = whisper.load_model("base")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag_data")
print("✅ Models and Vector DB ready!\n")


# ======================================================
#  PDF PROCESSING
# ======================================================
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue

        emb = embed_model.encode(text).tolist()
        collection.add(
            ids=[f"{title}page{page_num}"],
            embeddings=[emb],
            metadatas=[{"type": "pdf", "file": title, "page": page_num}],
            documents=[text]
        )

        chunks.append({"page": page_num, "text": text})

    merged_pages = merge_chunks_with_labels(chunks, 5, is_pdf=True)

    json_path = f"jsons/{title}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"merged_pages": merged_pages}, f, ensure_ascii=False, indent=2)

    return f"✅ PDF '{title}' processed and merged!"


# ======================================================
#  AUDIO PROCESSING
# ======================================================
def process_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language="hi", task="translate")
    title = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = []

    for i, seg in enumerate(result["segments"], start=1):
        text = seg["text"].strip()
        emb = embed_model.encode(text).tolist()

        collection.add(
            ids=[f"{title}seg{i}"],
            embeddings=[emb],
            metadatas=[{
                "type": "audio",
                "file": title,
                "start": seg["start"],
                "end": seg["end"]
            }],
            documents=[text]
        )

        chunks.append({
            "segment": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })

    merged_segments = merge_chunks_with_labels(chunks, 5, is_pdf=False)

    json_path = f"jsons/{title}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"merged_segments": merged_segments}, f, ensure_ascii=False, indent=2)

    return f"✅ Audio '{title}' processed and merged!"


# ======================================================
#  FLASK ROUTES
# ======================================================

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/up", methods=["POST"])
def up():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"})

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        if file.filename.endswith(".pdf"):
            msg = process_pdf(file_path)
            log = UploadLog(filename=file.filename, filetype="pdf")
            db.session.add(log)
            db.session.commit()

        elif file.filename.endswith((".mp3", ".wav")):
            msg = process_audio(file_path)
            log = UploadLog(filename=file.filename, filetype="audio")
            db.session.add(log)
            db.session.commit()

        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            extracted_text = pytesseract.image_to_string(Image.open(file_path))
            log = UploadLog(filename=file.filename, filetype="image")
            db.session.add(log)
            db.session.commit()

            collection.add(
                documents=[extracted_text],
                metadatas=[{"source": file.filename, "type": "image"}],
                ids=[file.filename]
            )
            msg = "Image uploaded successfully!"

        else:
            return jsonify({"error": "Only PDF, MP3, WAV, PNG, JPG supported!"})

    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"message": msg})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "⚠ Enter a valid question."})

    query_embed = embed_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embed], n_results=4)

    if not results["documents"] or not results["documents"][0]:
        return jsonify({"answer": "No relevant content found."})

    contexts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if meta.get("type") == "pdf":
            tag = f"(📘 PDF: {meta['file']}, Page {meta['page']})"
        elif meta.get("type") == "audio":
            tag = f"(🎧 Audio: {meta['file']} {meta['start']}s→{meta['end']}s)"
        else:
            tag = "(Unknown source)"

        contexts.append(f"{tag}\n{doc}")

    context_text = "\n\n".join(contexts)

    prompt = f"""
Relevant context:
{context_text}

--------------------
User Question: "{query}"
"""

    # ======================================================
    # STREAMING RESPONSE — FIX APPLIED HERE
    # ======================================================
    def generate_stream():
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": True},
                stream=True
            )

            answer_full = ""

            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    answer_full += token
                    yield token

            # FIX: SAVE TO DB WITH app.app_context()
            with app.app_context():
                log = QueryLog(question=query, answer=answer_full)
                db.session.add(log)
                db.session.commit()

        except Exception as e:
            yield f"\n Stream error: {e}"

    from flask import Response
    return Response(generate_stream(), mimetype="text/plain")


# ======================================================
#  MAIN
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
