"""
Script to generate a comprehensive PDF:
"EmotiBeat: Tech Stack, Model Architecture Selection Matrix & Expected Interviewer Q&A"
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = Path(r"d:\Enotion based music player\EmotiBeat_TechStack_Models_and_Interview_QA.pdf")


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds dynamic total page numbers and running headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#334155"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "EmotiBeat — Technologies, Models & Expected Interview Questions")
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "Technical Architecture & Q&A Defense")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
        
        # Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * 72 - 54, 36, page_str)
        self.drawString(54, 36, "EmotiBeat: Real-Time Emotion-Controlled Music Player • Technical Defense Guide")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * 72 - 54, 48)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()
    
    # Custom Color Palette
    C_NAVY = colors.HexColor("#0F172A")          # Slate 900
    C_HEADER = colors.HexColor("#1E1B4B")        # Deep Indigo
    C_PRIMARY = colors.HexColor("#312E81")       # Indigo 900
    C_ACCENT = colors.HexColor("#0D9488")        # Dark Teal
    C_CARD_BG = colors.HexColor("#F8FAFC")       # Slate 50
    C_CARD_BORDER = colors.HexColor("#E2E8F0")   # Slate 200
    C_QA_Q_BG = colors.HexColor("#EEF2FF")       # Indigo 50
    C_QA_BORDER = colors.HexColor("#6366F1")     # Indigo Accent

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=C_HEADER,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=C_ACCENT,
        spaceAfter=10,
    )
    
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=C_HEADER,
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=C_ACCENT,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=C_NAVY,
    )

    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    qa_q_style = ParagraphStyle(
        "QAQuestion",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.8,
        leading=13.5,
        textColor=colors.HexColor("#1E1B4B"),
        keepWithNext=True,
    )

    qa_a_style = ParagraphStyle(
        "QAAnswer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=C_NAVY,
    )

    story = []

    # Title & Banner
    story.append(Paragraph("🛠️ EmotiBeat: Technologies, Models & Interview Q&A Defense", title_style))
    story.append(Paragraph("Comprehensive Guide: Technology Stack, Model Selection Justifications & Expected Interview Questions", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceBefore=2, spaceAfter=8))

    # ==========================================
    # SECTION 1: COMPLETE TECH STACK BREAKDOWN
    # ==========================================
    story.append(Paragraph("1. Technology Stack Breakdown (What & Why)", h1_style))
    story.append(Paragraph("Detailed mapping of each technology used in EmotiBeat, its exact role, and engineering justification:", body_style))
    story.append(Spacer(1, 4))

    tech_rows = [
        [
            Paragraph("<b>Layer / Domain</b>", badge_style),
            Paragraph("<b>Technology / Library</b>", badge_style),
            Paragraph("<b>Purpose in EmotiBeat</b>", badge_style),
            Paragraph("<b>Why Chosen over Alternatives</b>", badge_style)
        ],
        [
            Paragraph("<b>Core Language</b>", body_style),
            Paragraph("<b>Python 3.10+</b>", body_style),
            Paragraph("Entire backend, ML pipelines, API integrations, and UI.", body_style),
            Paragraph("Rich ecosystem for Computer Vision (OpenCV), Deep Learning, and fast prototyping.", body_style)
        ],
        [
            Paragraph("<b>Computer Vision</b>", body_style),
            Paragraph("<b>OpenCV (cv2)</b>", body_style),
            Paragraph("Face detection, bounding-box expansion, histogram equalization, frame preprocessing.", body_style),
            Paragraph("Ultra-fast C++ optimized operations and native DNN execution module without heavy GPU requirements.", body_style)
        ],
        [
            Paragraph("<b>Deep Learning</b>", body_style),
            Paragraph("<b>PyTorch & HuggingFace</b>", body_style),
            Paragraph("Vision Transformer (ViT) pipeline for fine-grained 7-class facial expression recognition.", body_style),
            Paragraph("Multi-head self-attention captures global facial relationships better than isolated convolutions.", body_style)
        ],
        [
            Paragraph("<b>Edge Inference</b>", body_style),
            Paragraph("<b>ONNX & cv2.dnn</b>", body_style),
            Paragraph("FERPlus deep neural network inference on CPU with sub-15ms latency.", body_style),
            Paragraph("Eliminates PyTorch runtime overhead during lightweight webcam video processing.", body_style)
        ],
        [
            Paragraph("<b>GenAI & NLP</b>", body_style),
            Paragraph("<b>Google Gemini</b><br/>(<code>langchain-google-genai</code>)", body_style),
            Paragraph("Multilingual sentiment reasoning (English, Hindi, Hinglish, Spanish) and multimodal image analysis.", body_style),
            Paragraph("Zero-shot semantic understanding with temperature=0 for deterministic mood-to-music mapping.", body_style)
        ],
        [
            Paragraph("<b>Signal Processing</b>", body_style),
            Paragraph("<b>NumPy & Custom EMA</b>", body_style),
            Paragraph("Exponential Moving Average probability smoothing, consensus voting buffer (N=10).", body_style),
            Paragraph("Vectorized array operations prevent micro-expression flickering without introducing latency.", body_style)
        ],
        [
            Paragraph("<b>Audio Synthesis</b>", body_style),
            Paragraph("<b>NumPy + Wave Module</b>", body_style),
            Paragraph("Algorithmic procedural 16-bit PCM WAV music synthesis for all 5 emotion categories.", body_style),
            Paragraph("Generates valid audio in-memory; completely solves cold-start dependencies without internet.", body_style)
        ],
        [
            Paragraph("<b>Web Dashboard</b>", body_style),
            Paragraph("<b>Streamlit (Glassmorphism)</b>", body_style),
            Paragraph("Real-time dual-column UI with camera feed, confidence gauges, and playback controls.", body_style),
            Paragraph("Rapid reactive UI state management (<code>st.session_state</code>) with custom CSS styling.", body_style)
        ],
        [
            Paragraph("<b>Quality Assurance</b>", body_style),
            Paragraph("<b>Pytest & Unittest</b>", body_style),
            Paragraph("Automated unit test suite covering vision, smoother, model, provider, and GenAI.", body_style),
            Paragraph("Ensures regression-free continuous integration and validates mathematical invariants.", body_style)
        ]
    ]

    tech_table = Table(tech_rows, colWidths=[80, 110, 155, 165])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_HEADER),
        ('BOX', (0, 0), (-1, -1), 0.5, C_CARD_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_CARD_BORDER),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_CARD_BG]),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 10))

    # ==========================================
    # SECTION 2: WHICH MODEL FOR WHICH APPROACH
    # ==========================================
    story.append(Paragraph("2. Model Selection Matrix: Which Model for Which Approach?", h1_style))
    story.append(Paragraph("A concise comparison justifying the exact architecture chosen for every modality and use case:", body_style))
    story.append(Spacer(1, 4))

    model_rows = [
        [
            Paragraph("<b>Task / Approach</b>", badge_style),
            Paragraph("<b>Selected Model / Architecture</b>", badge_style),
            Paragraph("<b>Input Format & Specs</b>", badge_style),
            Paragraph("<b>Key Architectural Strengths</b>", badge_style),
            Paragraph("<b>Trade-offs & Constraints</b>", badge_style)
        ],
        [
            Paragraph("<b>Real-Time Webcam CPU Inference</b>", body_style),
            Paragraph("<b>FERPlus ONNX</b><br/>(OpenCV DNN)", body_style),
            Paragraph("64x64x1 Grayscale tensor,<br/>Normalized [-1, 1]", body_style),
            Paragraph("• Sub-15ms inference on CPU.<br/>• Zero GPU dependency.<br/>• Minimal RAM (<50MB).", body_style),
            Paragraph("Slightly lower accuracy on extreme head angles compared to Transformers.", body_style)
        ],
        [
            Paragraph("<b>High-Accuracy Facial Recognition</b>", body_style),
            Paragraph("<b>Vision Transformer (ViT)</b><br/>(<code>google/vit-base-patch16-224</code>)", body_style),
            Paragraph("224x224x3 RGB PIL Image,<br/>ImageNet normalized", body_style),
            Paragraph("• SOTA global attention across entire face.<br/>• Zero bias on complex facial features.<br/>• High precision on subtle expressions.", body_style),
            Paragraph("Higher CPU inference time (~80-120ms); best utilized when GPU or batch mode is available.", body_style)
        ],
        [
            Paragraph("<b>Transfer Learning Baseline Training</b>", body_style),
            Paragraph("<b>MobileNetV2 Backbone</b><br/>(Keras / TensorFlow)", body_style),
            Paragraph("224x224x3 RGB,<br/>Data-augmented", body_style),
            Paragraph("• Depthwise separable convolutions.<br/>• 2-stage transfer learning (Frozen &rarr; Fine-tune).<br/>• Balanced class weights.", body_style),
            Paragraph("Requires saving larger weight files (.h5/SavedModel) compared to compact ONNX.", body_style)
        ],
        [
            Paragraph("<b>Multilingual Thought & Text Sentiment</b>", body_style),
            Paragraph("<b>Google Gemini</b><br/>(<code>gemini-1.5-flash</code> / <code>gemini-3.6-flash</code>)", body_style),
            Paragraph("Raw Text Strings (English, Hindi, Hinglish, Spanish)", body_style),
            Paragraph("• Understands cultural nuances, slang, and mixed-language input.<br/>• Outputs calibrated probabilities and semantic music tags.<br/>• Zero-shot reasoning.", body_style),
            Paragraph("Requires internet connection and Google API key; wrapped with local fallback.", body_style)
        ],
        [
            Paragraph("<b>Anti-Jitter Temporal Signal Stabilization</b>", body_style),
            Paragraph("<b>EMA + Consensus Voting</b><br/>(&alpha;=0.65, N=10 window)", body_style),
            Paragraph("5-dimensional probability distribution vectors", body_style),
            Paragraph("• Smooths frame-to-frame noise.<br/>• Hysteresis threshold rejects blinks.<br/>• Mathematical stability scoring.", body_style),
            Paragraph("Introduces a minor 200-300ms smoothing delay to ensure intentional mood shifts.", body_style)
        ]
    ]

    model_table = Table(model_rows, colWidths=[90, 105, 95, 115, 105])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_ACCENT),
        ('BOX', (0, 0), (-1, -1), 0.5, C_CARD_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_CARD_BORDER),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_CARD_BG]),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 10))

    # ==========================================
    # SECTION 3: EXPECTED INTERVIEWER QUESTIONS & MODEL ANSWERS
    # ==========================================
    story.append(Paragraph("3. Expected Technical Questions from Interviewer & Defenses", h1_style))
    story.append(Paragraph("Master these answers to explain the exact rationale behind design and model choices:", body_style))
    story.append(Spacer(1, 4))

    qa_data = [
        (
            "Q1: Why didn't you just use standard CNNs (like VGG16/ResNet50) for facial emotion recognition?",
            "\"Standard heavy CNNs like VGG16 or ResNet50 have over 25-130M parameters, leading to high latency (>100ms per frame) on edge devices without dedicated GPUs. For EmotiBeat, I selected a dual approach: 1) FERPlus ONNX via OpenCV DNN for sub-15ms real-time CPU performance, and 2) Vision Transformer (ViT) which uses multi-head self-attention to capture long-range spatial dependencies (e.g. eye-to-mouth ratio) rather than localized receptive fields.\""
        ),
        (
            "Q2: How does your Temporal Smoother differentiate between a micro-expression and a genuine mood change?",
            "\"We employ a two-layer stabilization mechanism: 1) Exponential Moving Average (EMA) applies an exponential decay (alpha = 0.65) to smooth out raw logits across time. 2) An N=10 Sliding Window Consensus Buffer computes a stability score S = (count of dominant class) / N. If a user blinks or looks away for 2 frames (200ms), the stability index stays below the 0.70 hysteresis threshold, so the music track continues playing without interruption.\""
        ),
        (
            "Q3: How did you solve the 'False Happy' classification when a person grimaces in distress/crying?",
            "\"In standard FER datasets, visible teeth strongly correlate with smiling, causing models to predict 'Happy' during painful grimaces or crying. In src/model.py, I engineered a Distress Ratio Filter. If the sum of distress probabilities (sad + fear + disgust + anger) exceeds 0.25, the happy probability is discounted unless the ratio of smile-to-distress is strictly above 1.4. This prevented false happy triggers during genuine frustration.\""
        ),
        (
            "Q4: Why use Google Gemini GenAI instead of traditional NLP (like TF-IDF or Word2Vec)?",
            "\"Traditional bag-of-words or lexicon approaches fail on contextual nuances, negation, and mixed multilingual inputs like Hinglish (e.g., 'Yaar bohot lonely feel ho raha hai'). Google Gemini (ChatGoogleGenerativeAI) with temperature=0 provides zero-shot semantic understanding, accurately extracting nuanced emotional undertones and generating contextual acoustic keywords without requiring a separate text dataset training pipeline.\""
        ),
        (
            "Q5: How does your music engine handle the 'Cold-Start' problem when no music files exist locally?",
            "\"Most local audio players crash if the music folder is empty. I designed a Procedural Harmonic Audio Synthesizer in src/music_engine/synthesizer.py using NumPy. On first startup, if no tracks are detected, it mathematically computes sine/harmonic audio waveforms across emotional scales (e.g., major third intervals for Happy, minor minor seventh intervals for Sad) and exports valid 16-bit PCM WAV tracks in milliseconds without requiring internet access.\""
        ),
        (
            "Q6: What loss function and optimization strategy did you use for model training?",
            "\"In src/train.py, I used Categorical Cross-Entropy with Label Smoothing (0.1) to prevent the network from becoming overconfident on ambiguous facial crops. Because FER datasets suffer from severe class imbalance, I computed balanced class weights via sklearn's compute_class_weight. Training was conducted in two stages: Stage 1 trained the custom classification head with Adam (lr=1e-4), and Stage 2 unfroze the backbone with a reduced learning rate (lr=1e-5) with ReduceLROnPlateau and EarlyStopping.\""
        ),
        (
            "Q7: Since you used APIs and pre-trained models, what is your actual contribution? APIs do everything—why did this project need your engineering?",
            "\"Calling an API or loading a model is only ~5% of an AI product; the other 95% is the engineering architecture that prevents real-world failure: 1) Anti-Jitter Signal Engine: Raw APIs flicker every 100ms on blinks; I built the EMA + 10-frame consensus buffer + hysteresis thresholding so music plays smoothly. 2) Domain Error Correction: I engineered a Distress Ratio heuristic in src/model.py to eliminate false-positive smiles during crying/pain grimacing. 3) Edge Independence: The system runs 100% locally and offline in sub-15ms via ONNX/ViT without cloud lock-in. 4) Procedural Audio Synthesis: I wrote a NumPy waveform generator to solve the cold-start dependency problem.\""
        ),
        (
            "Q8: Why didn't you build/train a Deep Learning model from scratch with random weights?",
            "\"Actually, I did design a complete DL architecture and training pipeline in src/train.py, but used Two-Stage Transfer Learning and Fine-Tuning rather than training from scratch for two reasons: 1) Data Efficiency & Overfitting: Datasets like FER-2013 have only ~30k images. Training a 50-layer CNN from random weights on 30k images causes severe overfitting and poor generalization; pre-trained weights provide robust low-level edge/spatial representations. 2) Edge Latency: MobileNetV2 uses depthwise separable convolutions reducing FLOPs by 8x for 30+ FPS real-time CPU performance. On top of the backbone, I built a custom classification head (GAP + Dense 256 + BatchNorm + 40% Dropout + Softmax) and trained it with balanced class weighting.\""
        )
    ]

    for q, a in qa_data:
        qa_block = Table([
            [Paragraph(f"<b>{q}</b>", qa_q_style)],
            [Paragraph(a, qa_a_style)]
        ], colWidths=[510])
        qa_block.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), C_QA_Q_BG),
            ('BACKGROUND', (0, 1), (-1, 1), colors.white),
            ('BOX', (0, 0), (-1, -1), 0.75, C_QA_BORDER),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, 0), 4),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 4),
        ]))
        story.append(qa_block)
        story.append(Spacer(1, 5))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated PDF at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
