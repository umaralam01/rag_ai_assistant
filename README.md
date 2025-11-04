# 🔥 Multimodal RAG AI System  
**Where Fire, Water, Image, and Voice Unite to Power Knowledge Retrieval**

---

## 🚀 Overview

This project is a **Flask-based Multimodal Retrieval-Augmented Generation (RAG) System** that allows you to **upload PDFs, audio files, and images**, and then **ask natural questions** about their content.

It integrates multiple AI models to process and understand multimodal data — combining **text, image, and voice** into one intelligent retrieval system.

---

## ✨ Features

| Type | Description |
|------|--------------|
| 📄 **PDF Upload** | Extracts and embeds page-wise content for semantic search |
| 🎧 **Audio Upload** | Transcribes and translates Hindi/English audio using Whisper |
| 🖼️ **Image Upload** | Extracts text via OCR (Tesseract) and stores embeddings |
| 💬 **Ask Questions** | Query across all uploaded content using LLaMA 3.2 |
| 🔍 **Vector Search** | Uses ChromaDB for efficient embedding-based retrieval |
| ⚙️ **Local AI** | Runs completely offline using Ollama (no API keys needed) |
| 🤖 **LLM Response** | Answers intelligently with context references (page/segment) |

---

## 🧠 Tech Stack

| Category | Technology |
|-----------|-------------|
| **Backend Framework** | Flask |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`, `clip-ViT-B-32`) |
| **Audio Transcription** | OpenAI Whisper |
| **OCR Engine** | Tesseract OCR + Pillow |
| **Vector Database** | ChromaDB |
| **LLM Backend** | Ollama (LLaMA 3.2) |
| **Frontend** | HTML, CSS, JavaScript (Flask Templates) |

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/umaralam01/multimodal-rag-ai.git
cd multimodal-rag-ai