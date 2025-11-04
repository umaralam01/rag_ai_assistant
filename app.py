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

# ======================
#  APP SETUP
# ======================
app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)
os.makedirs("jsons", exist_ok=True)

# ======================
#  LOAD MODELS
# ======================
print("🚀 Loading models... please wait...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = SentenceTransformer('clip-ViT-B-32')

whisper_model = whisper.load_model("base")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("rag_data")
print("✅ Models and Vector DB ready!\n")


# ======================
#  PDF PROCESSING
# ======================
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
            ids=[f"{title}_page_{page_num}"],
            embeddings=[emb],
            metadatas=[{"type": "pdf", "file": title, "page": page_num}],
            documents=[text]
        )
        chunks.append({"page": page_num, "text": text})

    json_path = f"jsons/{title}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"pages": chunks}, f, ensure_ascii=False, indent=2)

    return f"✅ PDF '{title}' processed and embeddings stored!"


# ======================
#  AUDIO PROCESSING
# ======================
def process_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language="hi", task="translate")
    title = os.path.splitext(os.path.basename(audio_path))[0]
    chunks = []

    for i, seg in enumerate(result["segments"], start=1):
        text = seg["text"].strip()
        emb = embed_model.encode(text).tolist()
        collection.add(
            ids=[f"{title}_seg_{i}"],
            embeddings=[emb],
            metadatas=[{"type": "audio", "file": title, "start": seg["start"], "end": seg["end"]}],
            documents=[text]
        )
        chunks.append({"segment": i, "start": seg["start"], "end": seg["end"], "text": text})

    json_path = f"jsons/{title}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": chunks}, f, ensure_ascii=False, indent=2)

    return f"✅ Audio '{title}' transcribed and embeddings stored!"


# ======================
#  QUERY + LLM RESPONSE
# ======================
def query_llm(user_query):
    if not user_query.strip():
        return "⚠️ Please enter a valid question."

    # --- Search in Vector DB ---
    q_emb = embed_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=4)

    if not results["documents"] or not results["documents"][0]:
        return "😕 No relevant content found. Try uploading a PDF or audio first."

    prompt_chunks = []
    for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
        prompt_chunks.append({"meta": meta, "text": doc_text})

    # --- Prepare Prompt for Llama3.2 ---
    prompt_json = json.dumps(prompt_chunks, ensure_ascii=False, indent=2)
    prompt = f"""
You are an intelligent assistant analyzing uploaded PDFs and audio transcripts.

Below are relevant text chunks from the data:

{prompt_json}

---------------------------------
User Question: "{user_query}"

Instructions:
- Answer clearly using only the given content.
- Mention the relevant page or segment numbers.
- If question is unrelated, say: "I can only answer based on uploaded documents or audio."
"""

    # --- Query Llama3.2 (Ollama) ---
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        res = r.json()
        return res.get("response", "⚠️ No response from Llama3.2.")
    except Exception as e:
        return f"❌ Error contacting Ollama: {e}"


# ======================
#  FLASK ROUTES
# ======================
@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/up",methods=["POST"])
def up():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded!"})

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        if file.filename.endswith(".pdf"):
            msg = process_pdf(file_path)

        elif file.filename.endswith((".mp3", ".wav")):
            msg = process_audio(file_path)

        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # 🖼️ Image file handling (OCR)
            extracted_text = pytesseract.image_to_string(Image.open(file_path))

            # Tumhara ChromaDB collection yahan add hoga (example)
            collection.add(
                documents=[extracted_text],
                metadatas=[{"source": file.filename, "type": "image"}],
                ids=[file.filename]
            )

            msg = f"Image uploaded successfully! "

        else:
            return jsonify({"error": "Unsupported file type! Use PDF, MP3, WAV, or Image (PNG/JPG)."})
            
    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"message": msg})

@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image uploaded!"})

    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    try:
        # Step 1: Load image and create embedding
        image = Image.open(img_path)
        img_emb = clip_model.encode(image, convert_to_tensor=False).tolist()

        # Step 2: Search for best matching text chunks (PDF + audio)
        results = collection.query(query_embeddings=[img_emb], n_results=3)

        if not results["documents"] or not results["documents"][0]:
            return jsonify({"answer": "😕 No matching content found for this image."})

        best_doc = results["documents"][0][0]
        best_meta = results["metadatas"][0][0]

        # Step 3: Compute simple similarity threshold (optional)
        distances = results.get("distances", [[1.0]])[0]
        best_distance = distances[0] if distances else 1.0

        if best_distance > 0.5:
            response = "❌ This image doesn't appear to match any uploaded PDF or audio content."
        else:
            src_type = best_meta.get("type", "unknown")
            if src_type == "pdf":
                response = f"🖼️ This image seems related to page {best_meta.get('page', '?')} of '{best_meta.get('file', 'unknown')}.pdf'\n\n{best_doc}"
            elif src_type == "audio":
                response = f"🖼️ This image seems related to audio segment {best_meta.get('start', 0):.1f}s–{best_meta.get('end', 0):.1f}s in '{best_meta.get('file', 'unknown')}.mp3'\n\n{best_doc}"
            else:
                response = f"🖼️ This image might be related to '{best_meta.get('file', 'unknown')}'.\n\n{best_doc}"

        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"answer": "⚠️ Please enter a valid question."})

    # --- Step 1: Embed query ---
    query_embed = embed_model.encode([query]).tolist()[0]

    # --- Step 2: Search in ChromaDB ---
    results = collection.query(query_embeddings=[query_embed], n_results=4)
    if not results["documents"] or not results["documents"][0]:
        return jsonify({"answer": "😕 No relevant content found. Try uploading a PDF or audio first."})

    # --- Step 3: Build context dynamically ---
    contexts = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        src_type = meta.get("type", "unknown")
        file_name = meta.get("file", "unknown")

        if src_type == "pdf":
            page = meta.get("page", "unknown")
            tag = f"(📘 PDF: {file_name}, Page {page})"
        elif src_type == "audio":
            start = meta.get("start", 0)
            end = meta.get("end", 0)
            tag = f"(🎧 Audio: {file_name}, {start:.1f}s–{end:.1f}s)"
        else:
            tag = f"(Unknown source: {file_name})"

        contexts.append(f"{tag}\n{doc}")

    # --- Step 4: Create unified LLM prompt ---
    context_text = "\n\n".join(contexts)
    prompt = f"""
You are an intelligent RAG assistant that answers questions based on extracted data from PDFs and audio transcripts.

Relevant extracted context:
{context_text}

---------------------------------
User Question: "{query}"

Instructions:
- Use only the provided context to answer.
- Mention whether the answer came from a PDF (with page number) or an audio segment (with timestamps).
- If the information is unclear, respond: "Not found in the uploaded content."



"""

    # --- Step 5: Query Llama3.2 locally via Ollama ---
    try:
        import subprocess
        cmd = ["ollama", "run", "llama3.2", prompt]
        result = subprocess.run(
                         cmd,
                        capture_output=True,
                         text=True,
                         encoding="utf-8",     # ✅ Ensures proper Unicode handling
                        errors="replace"      # ✅ Prevents crashes if a symbol can't render
                                 )
        answer = result.stdout.encode('utf-8', 'replace').decode('utf-8').strip()

        
    except Exception as e:
        answer = f"❌ Error contacting Llama3.2: {e}"

    return jsonify({"answer": answer})


# ======================
#  MAIN
# ======================
if __name__ == "__main__":
    app.run(debug=True)




















