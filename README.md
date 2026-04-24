🧠 Graph-Based Event Extraction for Low-Resource Languages

This project implements a parser-independent, graph-based event extraction framework designed for low-resource, morphologically rich languages (e.g., Manipuri).

It combines:

* Morphological analysis
* Graph Neural Networks (GCN)
* Sequence modeling (BiLSTM)
* Multi-task learning (POS, NER, Trigger, Arguments)

⸻

🚀 Features

* ✅ Parser-free architecture (no dependency parser needed)
* ✅ Graph-based modeling using token relations
* ✅ Trigger detection + argument extraction
* ✅ Multi-task learning setup
* ✅ FastText graph embeddings (optional)
* ✅ Detailed evaluation + CSV output

⸻

📂 Project Structure
.
├── graph_builder.py      # Builds sentence-level graphs
├── dataset.py            # Loads and encodes dataset
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation + prediction dump
├── config.py             # Configuration settings
├── models.py             # EventExtractor model (GCN + LSTM)
├── utils.py              # Vocab and helpers
├── morphology.py         # Morphological analyzer
├── prediction_dump.json  # Model outputs
├── token_level_results.csv

📦 Requirements
torch>=2.0
numpy
tqdm
matplotlib
gensim
scikit-learn

📊 Dataset Format

Your dataset must be JSON in this structure:
{
  "doc_id": "doc_1",
  "sentence_annotations": [
    {
      "sid": 1,
      "tokens": ["..."],
      "pos_labels": ["..."],
      "ner_labels": ["..."],
      "entities": [],
      "events": [
        {
          "event_id": "e1",
          "type": "Reporting",
          "trigger": "said",
          "trigger_span": [3, 3],
          "arguments": [
            {
              "role": "agent",
              "text": "John",
              "span": [0, 0]
            }
          ]
        }
      ]
    }
  ]
}

🏋️ Training
Run: python train.py

🧪 Evaluation
Run: python evaluate.py
