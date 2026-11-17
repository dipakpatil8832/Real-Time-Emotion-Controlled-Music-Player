"""
Script to generate a pure spoken Interview Presentation Script PDF:
"EmotiBeat: Word-for-Word Spoken Interview Script (End-to-End Project Explanation)"
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

PDF_OUTPUT_PATH = Path(r"d:\Enotion based music player\EmotiBeat_Spoken_Interview_Script.pdf")


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
            self.drawString(54, 11 * 72 - 36, "EmotiBeat — End-to-End Spoken Interview Script")
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "Spoken Dialogue & Presentation")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)
        
        # Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * 72 - 54, 36, page_str)
        self.drawString(54, 36, "EmotiBeat • Complete Spoken Walkthrough Script for Technical Interviews")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * 72 - 54, 48)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    C_PRIMARY = colors.HexColor("#0F172A")      # Slate 900
    C_HEADER = colors.HexColor("#1E1B4B")       # Deep Indigo
    C_ACCENT = colors.HexColor("#4F46E5")       # Indigo 600
    C_SPOKEN_BG = colors.HexColor("#F8FAFC")    # Slate 50
    C_SPOKEN_BORDER = colors.HexColor("#CBD5E1")# Slate 300
    C_CUE_BG = colors.HexColor("#FEF3C7")       # Amber 100
    C_CUE_TEXT = colors.HexColor("#92400E")     # Amber 800
    C_TIP_BG = colors.HexColor("#F0FDF4")       # Emerald 50
    C_TIP_BORDER = colors.HexColor("#22C55E")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
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
        spaceAfter=12,
    )
    
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=C_HEADER,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    cue_style = ParagraphStyle(
        "StageCue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=C_CUE_TEXT,
    )

    spoken_style = ParagraphStyle(
        "SpokenDialogue",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.8,
        leading=14.5,
        textColor=C_PRIMARY,
    )

    tip_style = ParagraphStyle(
        "TipText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#166534"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("🗣️ EmotiBeat: Complete Spoken Interview Script", title_style))
    story.append(Paragraph("A Step-by-Step Word-for-Word Speaking Guide to Explain Your Project to Interviewers", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceBefore=2, spaceAfter=10))

    def make_speech_block(cue_text, speech_text, pro_tip=None):
        content = [
            Paragraph(f"🎬 <b>INTERVIEW ACTION / CUE:</b> {cue_text}", cue_style),
            Spacer(1, 4),
            Paragraph(f"<b>YOU SAY:</b><br/>\"{speech_text}\"", spoken_style)
        ]
        if pro_tip:
            content.append(Spacer(1, 4))
            content.append(Paragraph(f"💡 <i>Interview Pro-Tip: {pro_tip}</i>", tip_style))

        t = Table([[content]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), C_SPOKEN_BG),
            ('BOX', (0, 0), (-1, -1), 1, C_SPOKEN_BORDER),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        return t

    # ==========================================
    # PART 1: THE OPENING
    # ==========================================
    story.append(Paragraph("PART 1: The Opening & Hook (When they ask: 'Tell me about your project')", h1_style))
    story.append(make_speech_block(
        "Maintain eye contact, speak with energy and clarity.",
        "Good morning / afternoon! I'd love to explain my project, <b>EmotiBeat</b>.<br/><br/>"
        "EmotiBeat is an <b>AI-powered, real-time emotion-controlled music player</b>. "
        "The core idea is to eliminate manual search and playlist fatigue by allowing the system to dynamically "
        "sense the user's emotional state in real time and automatically stream music that matches or elevates their mood.<br/><br/>"
        "What makes this project unique is that it doesn't just rely on a simple face detector; it combines "
        "<b>Computer Vision, Vision Transformers (ViT), ONNX Deep Neural Networks, Google Gemini GenAI for multilingual text sentiment, "
        "and a Temporal Smoothing Engine</b> to create a seamless, jitter-free music listening experience.",
        "Notice how this opening highlights AI, GenAI, signal processing, and audio in under 45 seconds."
    ))
    story.append(Spacer(1, 10))

    # ==========================================
    # PART 2: THE REAL-WORLD PROBLEM
    # ==========================================
    story.append(Paragraph("PART 2: Why I Built This & The Problem Statement", h1_style))
    story.append(make_speech_block(
        "Show empathy for user experience and explain the technical gap in existing apps.",
        "If we look at popular music apps like Spotify or Apple Music, recommendations are largely based on "
        "past listening history or static genre tags. But human emotions are dynamic — you might be stressed during work, "
        "celebrating good news, or feeling exhausted in the evening.<br/><br/>"
        "Existing research projects in emotion-based music often fail because of two big flaws:<br/>"
        "<b>1. Emotion flickering:</b> A single blink or lighting shift causes the music to stop and switch songs abruptly.<br/>"
        "<b>2. Strict single-modality:</b> If the user turns off their webcam or is in a dark room, the entire app becomes useless.<br/><br/>"
        "I engineered EmotiBeat to solve both of these real-world challenges through <b>dual-modal inputs and temporal stabilization</b>.",
        "Interviewers love candidates who identify why standard naive implementations fail."
    ))
    story.append(Spacer(1, 10))

    # ==========================================
    # PART 3: STEP-BY-STEP END-TO-END PIPELINE
    # ==========================================
    story.append(Paragraph("PART 3: End-to-End Technical Pipeline (Step-by-Step)", h1_style))
    story.append(make_speech_block(
        "Break the architecture into 5 logical chronological steps.",
        "Let me walk you through the end-to-end data flow from the moment a user appears on camera to when the song plays:<br/><br/>"
        "<b>Step 1: Face Detection & Intelligent Margin Expansion</b><br/>"
        "The camera captures the frame. Using OpenCV, I perform histogram equalization to handle dim or bright lighting. "
        "Then, instead of taking a tight bounding box, I engineered a <b>+15% margin expansion</b>. This is crucial because "
        "emotions like sadness or surprise rely heavily on eyebrow raises, forehead wrinkles, and jawline movement which tight crops cut off.<br/><br/>"
        "<b>Step 2: Dual Deep Learning Inference (ViT & ONNX)</b><br/>"
        "The cropped face is passed into our classification engine. I provided two switchable backends: a <b>Vision Transformer (ViT)</b> "
        "for state-of-the-art attention-based expression accuracy, and an <b>OpenCV FERPlus ONNX model</b> optimized for sub-15ms inference "
        "on ordinary CPUs. It classifies 5 primary emotions: Happy, Sad, Angry, Surprise, and Neutral.<br/><br/>"
        "<b>Step 3: Multimodal GenAI & Multilingual Text Mode (Google Gemini)</b><br/>"
        "If a user prefers typing how they feel or their webcam is off, they can write their thoughts in English, Hindi, or Hinglish "
        "(e.g., <i>'Bohot stressed hu aaj exam ki wajah se'</i>). I integrated Google Gemini (ChatGoogleGenerativeAI) with zero temperature "
        "to perform semantic sentiment reasoning and extract the exact emotional mood and song theme.<br/><br/>"
        "<b>Step 4: Temporal Smoothing & Hysteresis Filtering</b><br/>"
        "This is the brain of the player. To prevent songs from constantly jumping, raw probabilities pass through an "
        "<b>Exponential Moving Average (EMA)</b> with a decay factor of 0.65, followed by an <b>N=10 Sliding Window Consensus Buffer</b>. "
        "A song only switches if the dominant emotion maintains a stability score greater than 70%.<br/><br/>"
        "<b>Step 5: Dynamic Music Recommendation & Procedural Synthesis</b><br/>"
        "Once the mood is confirmed, our Extensible Music Provider queues the matching song. We support <b>39 curated multilingual YouTube tracks</b>, "
        "local mp3/wav libraries, and a <b>built-in Procedural Harmonic Synthesizer</b> that algorithmically generates audio waveforms in memory "
        "so the project works out-of-the-box even without an internet connection.",
        "Mentioning ViT, ONNX, EMA, and Gemini shows you understand both edge computing and modern GenAI."
    ))
    story.append(Spacer(1, 10))

    # ==========================================
    # PART 4: THE HARDEST CHALLENGE & HOW I SOLVED IT
    # ==========================================
    story.append(Paragraph("PART 4: Key Technical Challenges You Solved", h1_style))
    story.append(make_speech_block(
        "Pick this story to demonstrate deep debugging and analytical problem-solving.",
        "One of the most interesting bugs I encountered during testing was the <b>'Teeth Grimace vs. Smile' false positive</b>.<br/><br/>"
        "When people are angry, frustrated, or crying in pain, they often clench or bare their teeth. Because standard FER training datasets "
        "correlate visible teeth with smiling, standard neural networks would classify intense distress as 'Happy' with 40-50% confidence!<br/><br/>"
        "To fix this, I engineered a custom <b>Distress Filter heuristic</b> in <code>src/model.py</code>: if secondary distress signals "
        "(fear, disgust, sadness) are elevated above 0.25, the happy probability is mathematically penalized unless the smile-to-distress ratio "
        "is decisively greater than 1.4. This immediately eliminated false happy triggers during genuine frustration.",
        "Interviewers love hearing about real heuristics that fix domain-specific edge cases."
    ))
    story.append(Spacer(1, 10))

    # ==========================================
    # PART 5: THE LIVE DEMO SCRIPT
    # ==========================================
    story.append(Paragraph("PART 5: What to Say While Giving a Live Demo", h1_style))
    story.append(make_speech_block(
        "Narrate your screen while running 'streamlit run app.py'.",
        "<i>(Opening the browser)</i>: 'Here is the live EmotiBeat dashboard, built with a modern dark glassmorphism theme in Streamlit.<br/><br/>"
        "On the left column, you can see our live camera feed with real-time face bounding boxes and confidence meters. "
        "Notice that as I smile, the confidence meter smoothly climbs to Happy (95%), and the player queues an upbeat track.<br/><br/>"
        "If I briefly look away or blink, watch the stability index — the Temporal Smoother prevents the song from resetting because it knows it's a micro-expression.<br/><br/>"
        "Now let's switch over to the <b>Text & Gemini GenAI Tab</b>. If I type: <i>'Feeling exhausted after studying all night'</i>, "
        "Gemini classifies this as Sad/Melancholic and automatically triggers relaxing lofi acoustic music. "
        "All components are completely tested with our automated Pytest test suite.'",
        "Always talk while clicking; never let there be awkward silence during a demo."
    ))
    story.append(Spacer(1, 10))

    # ==========================================
    # PART 6: STRONG CLOSING STATEMENT
    # ==========================================
    story.append(Paragraph("PART 6: Strong Final Closing Statement", h1_style))
    story.append(make_speech_block(
        "Conclude with confidence and invite architectural discussion.",
        "To summarize: EmotiBeat is not just a toy machine learning demo. It is a full-stack, modular audio intelligence system "
        "that brings together <b>Vision Transformers, ONNX edge inference, Multimodal GenAI, temporal signal stabilization, and extensible audio streaming</b>.<br/><br/>"
        "The code is fully modular, adhering to clean OOP principles, and 100% of unit tests pass cleanly. "
        "I would be happy to dive into any specific module, loss functions, or architecture decisions you'd like to explore!\"",
        "This invites the interviewer to ask questions in areas where you are strongest."
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated PDF at: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
