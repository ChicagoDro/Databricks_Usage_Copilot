.PHONY: setup db index app all clean

# Default DB path and index path
DB=data/usage_rag_data.db

# ------------------------------------------------------
# 1. Initialize SQLite database & seed data
# ------------------------------------------------------
db:
	@echo "🗄️  Creating SQLite Databricks usage database..."
	python database_setup.py $(DB)
	@echo "✔️  Database created: $(DB)"

# ------------------------------------------------------
# 2. Build FAISS index
# ------------------------------------------------------
index:
	@echo "📦 Building FAISS vector index..."
	python -m src.ingest_embed_index
	@echo "✔️  FAISS index built."

# ------------------------------------------------------
# 3. Launch the Streamlit UI
# ------------------------------------------------------
app:
	@echo "🚀 Launching Streamlit UI..."
	export PYTHONPATH=$(PWD) && streamlit run src/app.py


# ------------------------------------------------------
# 4. Full setup + app
# ------------------------------------------------------
all: db index app

# ------------------------------------------------------
# 5. Clean generated files
# ------------------------------------------------------
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf indexes/usage_faiss
	rm -f $(DB)
	@echo "✔️  Cleaned."
