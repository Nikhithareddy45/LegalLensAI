# LegalLensAI 🏛️

An advanced AI-powered legal document analysis system that combines machine learning models with natural language processing to provide comprehensive legal contract analysis, risk detection, and intelligent querying capabilities.

## 🎯 Objective
- Provide end-to-end, document-grounded analysis for legal contracts: instant summary, risk detection with explanations, and a QA assistant that answers strictly from the contract.
- Deliver reproducible figures and tables for publication (IEEE-ready), plus a live demo UI.

## 🔧 Technologies Used
- Language Models: LegalBERT (nlpaueb/legal-bert-base-uncased), BART (facebook/bart-large-cnn)
- Python Libraries: transformers, torch, scikit-learn, pandas, numpy, seaborn, matplotlib
- App & Reports: Streamlit (web UI), ReportLab (PDF generation)
- Retrieval: Dense embeddings + cosine similarity; rule-based patterns for risk detection

## 🔄 System Flow
- Upload contract → Instant summary → Risk analysis → Legal QA (document-only chat)
- Components:
  - Summarization: Extractive + optional Abstractive fusion
  - Risk Detection: Semantic + rules → Fusion scoring
  - QA: Retrieve top clauses → constrained, context-grounded answer

## 🚀 Features

### 📋 Document Analysis
- **Clause Classification**: Automatically identifies and classifies legal clauses using BERT-based models
- **Risk Detection**: Multi-layered risk assessment using semantic analysis and rule-based patterns
- **Summarization**: Both extractive and abstractive summarization for quick document overview
- **Query Processing**: Intelligent retrieval system for finding relevant legal clauses

### 🛡️ Risk Detection System
- **Semantic Risk Analysis**: Uses neural networks to understand contextual risks
- **Rule-Based Detection**: Pattern matching for known risky legal terms
- **Fusion Approach**: Combines multiple methods for comprehensive risk assessment
- **Risk Scoring**: Quantified risk levels (LOW, MEDIUM, HIGH) with explanations

### 📝 Summarization Capabilities
- **Extractive Summarization**: TF-IDF based sentence ranking
- **Abstractive Summarization**: BART model for generative summaries
- **Fusion Summarization**: Combines both approaches for optimal results

### 🔍 Query System
- **Semantic Search**: LegalBERT embeddings for intelligent clause retrieval
- **Context-Aware Results**: Provides summaries and risk analysis for retrieved clauses
- **Similarity Scoring**: Ranked results based on semantic relevance

## ✅ Step-by-Step Procedure
1. Install and launch
   - Create venv and install requirements
   - Run the web app:
     ```bash
     streamlit run webapp/app.py
     ```
   - Open the local URL shown (e.g., http://localhost:8501/)
2. Use the web app
   - Upload or paste contract text in Upload tab → Instant Summary appears
   - Risk & Summary tab → Analyze risks and download PDF report
   - Legal QA tab → Ask questions; answers are grounded in the contract with evidence
3. Generate figures (PNG + SVG) for paper
   - Script: [scripts/generate_figures.py](file:///d:/Perug/LegalLensAi-Majorrrrrr/LegalLensAI_chat/scripts/generate_figures.py)
   - Output folder: results/figures
   - Run:
     ```bash
     python scripts/generate_figures.py
     ```
4. Generate LaTeX tables for paper
   - Script: [generate_paper_tables.py](file:///d:/Perug/LegalLensAi-Majorrrrrr/LegalLensAI_chat/results/scripts/generate_paper_tables.py)
   - Run:
     ```bash
     python results/scripts/generate_paper_tables.py
     ```
5. Reproduce evaluation
   - Ensure results/metrics CSVs exist, then re-run plots with the figure script
   - For end-to-end evaluation, use your project-specific evaluation scripts in src/evaluation and legal_cuad/

## 🏗️ Architecture

```
LegalLensAI/
├── src/
│   ├── data_loading/          # Data ingestion and processing
│   ├── data_preprocessing/    # Text cleaning and preparation
│   ├── modeling/             # ML models and training
│   ├── risk_detection/       # Risk analysis modules
│   ├── summarization/        # Document summarization
│   ├── query_system/         # Intelligent querying
│   ├── evaluation/           # Performance metrics
│   └── utils/               # Helper utilities
├── config/                   # Configuration files
├── data/                     # Data storage
├── models/                   # Trained models
├── results/                  # Analysis outputs
└── tests/                    # Test suites
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Nikhithareddy45/LegalLensAI.git
cd LegalLensAI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models** (automatically downloaded on first run):
- LegalBERT (`nlpaueb/legal-bert-base-uncased`)
- BART (`facebook/bart-large-cnn`)

## 🚀 Quick Start

### Basic Usage

```python
from src.query_system.query_engine import QueryEngine

# Initialize the system
engine = QueryEngine()

# Sample legal clauses
clauses = [
    "This Agreement may be terminated by either party upon written notice.",
    "Liability is limited to the amount paid under this Agreement.",
    "Confidential information must not be disclosed to third parties."
]

# Process a query
query = "What are the termination conditions?"
results = engine.process_query(query, clauses)

# View results
for result in results:
    print(f"Clause: {result['clause']}")
    print(f"Risk Score: {result['risk']['overall_semantic_risk']:.3f}")
    print(f"Summary: {result['summary']}")
```

### Risk Detection

```python
from src.risk_detection import SemanticRiskDetector, RuleBasedRiskDetector, RiskFusion

# Initialize risk detection
semantic = SemanticRiskDetector()
rules = RuleBasedRiskDetector()
fusion = RiskFusion(semantic, rules)

# Analyze legal text
text = "Liability is limited to the amount paid under this Agreement"
risk_analysis = fusion.fuse(text)

print(f"Overall Risk: {risk_analysis['overall_semantic_risk']:.3f}")
print(f"High Risk Categories: {risk_analysis['high_risk_categories']}")
```

### Document Summarization

```python
from src.summarization.fusion import FusionSummarizer

# Initialize summarizer
summarizer = FusionSummarizer()

# Summarize legal document
document = """
This Agreement may be terminated by either party upon written notice.
The Client shall pay all outstanding dues before termination.
Confidential information must not be disclosed.
Liability is limited to the amount paid under this Agreement.
Governing law shall be the laws of India.
"""

summary = summarizer.summarize(document)
print(f"Summary: {summary}")
```

## 🖼️ Figures and Tables
- Figures (architecture, training curves, per-category F1, ablation, confusion matrix, error distribution, demo heatmap, attention examples) generated to results/figures
- Tables (dataset statistics, classification, ROUGE, risk detection, retrieval, ablation, significance) printed as LaTeX via the tables script

## 📊 Model Performance

### Clause Classification
- **Model**: LegalBERT fine-tuned on CUAD dataset
- **Accuracy**: ~85% on validation set
- **Supported Clause Types**: 41+ categories

### Risk Detection
- **Semantic Analysis**: Keyword-based risk scoring
- **Rule-Based**: 12+ risk patterns
- **Fusion Accuracy**: Improved detection through ensemble methods

### Summarization
- **Extractive**: TF-IDF sentence ranking
- **Abstractive**: BART generative model
- **Fusion**: Combined approach for optimal results

## 🧪 Testing

### Run Checkpoints

```bash
# Test risk detection system
python -m src.checkpoint_3_risk_detection

# Test query system
python -m src.checkpoint_4_query_system

# Test individual components
python -m src.summarization.test_extractive
python -m src.summarization.test_abstractive
python -m src.summarization.test_fusion
```

### Model Training

```bash
# Train clause classification model
python -m src.modeling.train

# Run inference
python -m src.modeling.inference
```

## 📁 Data Requirements

### Input Format
Legal documents should be provided as plain text with standard legal clause formatting.

### Supported Document Types
- **Contracts**: Service agreements, NDAs, employment contracts
- **Legal Clauses**: Termination, liability, confidentiality, payment terms
- **Regulatory Documents**: Compliance documents, policy statements

## ⚙️ Configuration

### Model Settings
Edit `config/config.yaml` to adjust:
- Model paths and parameters
- Risk detection thresholds
- Summarization settings

### Data Paths
Configure data locations in `config/paths.yaml`:
- Training data directories
- Model storage paths
- Output locations

## 🔧 Advanced Usage

### Custom Risk Patterns

```python
from src.risk_detection.rules import RuleBasedRiskDetector

# Add custom risk patterns
detector = RuleBasedRiskDetector()
detector.rules["custom_risk"] = r"your_custom_regex_pattern"
```

### Custom Summarization

```python
from src.summarization.extractive import ExtractiveSummarizer

# Adjust summary length
summarizer = ExtractiveSummarizer(top_k=10)
summary = summarizer.summarize(document)
```

## 🧰 Troubleshooting
- Windows virtual memory (paging file) errors:
  - If you see “The paging file is too small” during model load, the app automatically falls back to extractive-only summarization and smaller embedding batches.
  - Optional: increase Windows paging file size for heavier models.
- Streamlit ports:
  - If the preview shows “connection refused”, restart the app and open the latest Local URL.

## 📈 Performance Metrics

### Evaluation Metrics
- **Precision/Recall**: For clause classification
- **F1 Score**: Overall model performance
- **Risk Detection Accuracy**: Pattern matching effectiveness
- **Summarization Quality**: ROUGE scores for abstractive summaries

### Benchmark Results
- **Clause Classification**: F1: 0.85
- **Risk Detection**: Precision: 0.82, Recall: 0.78
- **Summarization**: ROUGE-L: 0.72

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for code style
- Add tests for new features
- Update documentation
- Ensure all checkpoints pass

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CUAD Dataset**: Contract Understanding Atticus Dataset for training
- **Hugging Face**: Transformer models and tokenizers
- **LegalBERT**: Domain-specific BERT for legal text
- **scikit-learn**: Machine learning utilities

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

## 🔮 Future Roadmap

### Planned Features
- [ ] Multi-language support
- [ ] Web interface for easy document upload
- [ ] API endpoints for integration
- [ ] Advanced visualization of risk analysis
- [ ] Contract comparison tools
- [ ] Legal compliance checking

### Model Improvements
- [ ] Fine-tune on more specific legal domains
- [ ] Implement transformer-based risk detection
- [ ] Add few-shot learning capabilities
- [ ] Improve summarization quality

---

**LegalLensAI** - Making legal document analysis intelligent, accessible, and efficient. 🚀
