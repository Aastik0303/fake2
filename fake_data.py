import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# ---------------------------------------------------------
# Load TFLite Model
# ---------------------------------------------------------
interpreter = None
input_details = None
output_details = None

try:
    interpreter = tf.lite.Interpreter(model_path='deepfake_detector_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

def preprocess_image(image):
    image = image.resize((128, 128))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepDetect AI | Forensic Deepfake Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-base: #060810;
            --bg-surface: #0d1117;
            --bg-card: #111827;
            --bg-card2: #161d2b;
            --border: rgba(99, 179, 237, 0.12);
            --border-glow: rgba(99, 179, 237, 0.35);
            --accent: #38bdf8;
            --accent2: #818cf8;
            --accent3: #34d399;
            --danger: #f87171;
            --warn: #fbbf24;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #475569;
            --glow: 0 0 30px rgba(56, 189, 248, 0.15);
            --font-display: 'Syne', sans-serif;
            --font-mono: 'IBM Plex Mono', monospace;
            --font-body: 'DM Sans', sans-serif;
        }

        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        html { scroll-behavior: smooth; }

        body {
            font-family: var(--font-body);
            background: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ── Background grid ── */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
            z-index: 0;
        }

        /* ── Ambient orbs ── */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            pointer-events: none;
            z-index: 0;
            opacity: 0.4;
        }
        .orb-1 { width: 500px; height: 500px; background: radial-gradient(circle, rgba(56,189,248,0.2), transparent 70%); top: -150px; left: -150px; animation: float 12s ease-in-out infinite; }
        .orb-2 { width: 400px; height: 400px; background: radial-gradient(circle, rgba(129,140,248,0.15), transparent 70%); bottom: -100px; right: -100px; animation: float 15s ease-in-out infinite reverse; }
        .orb-3 { width: 300px; height: 300px; background: radial-gradient(circle, rgba(52,211,153,0.1), transparent 70%); top: 50%; left: 50%; transform: translate(-50%,-50%); animation: pulse-orb 8s ease-in-out infinite; }
        @keyframes float { 0%,100% { transform: translate(0,0); } 50% { transform: translate(30px, 20px); } }
        @keyframes pulse-orb { 0%,100% { opacity: 0.15; transform: translate(-50%,-50%) scale(1); } 50% { opacity: 0.3; transform: translate(-50%,-50%) scale(1.2); } }

        /* ── Navigation ── */
        nav {
            position: fixed;
            top: 0; left: 0; right: 0;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 48px;
            background: rgba(6,8,16,0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
        }
        .nav-logo {
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: var(--font-display);
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .nav-logo .dot { color: var(--accent); }
        .nav-logo .icon-wrap {
            width: 34px; height: 34px;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; color: #000;
        }
        .nav-links {
            display: flex;
            gap: 32px;
            list-style: none;
        }
        .nav-links a {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: color 0.2s;
            font-family: var(--font-mono);
            letter-spacing: 0.5px;
        }
        .nav-links a:hover { color: var(--accent); }
        .nav-badge {
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            color: #000;
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        }

        /* ── Page layout ── */
        .page-wrap {
            position: relative;
            z-index: 1;
            padding-top: 80px;
        }

        /* ── Hero ── */
        .hero {
            text-align: center;
            padding: 80px 24px 60px;
            max-width: 780px;
            margin: 0 auto;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.2);
            color: var(--accent);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 2px;
            text-transform: uppercase;
            padding: 7px 16px;
            border-radius: 20px;
            margin-bottom: 28px;
        }
        .hero-badge .pulse-dot {
            width: 6px; height: 6px;
            background: var(--accent3);
            border-radius: 50%;
            animation: blink 1.5s ease-in-out infinite;
        }
        @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.2; } }
        .hero h1 {
            font-family: var(--font-display);
            font-size: clamp(38px, 6vw, 68px);
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -2px;
            margin-bottom: 20px;
        }
        .hero h1 .line-accent {
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero p {
            color: var(--text-secondary);
            font-size: 17px;
            line-height: 1.7;
            max-width: 560px;
            margin: 0 auto;
        }
        .hero-stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
            padding-top: 40px;
            border-top: 1px solid var(--border);
        }
        .stat { text-align: center; }
        .stat-num {
            font-family: var(--font-display);
            font-size: 28px;
            font-weight: 800;
            color: var(--accent);
            letter-spacing: -1px;
        }
        .stat-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }

        /* ── Analyze Section ── */
        #analyze {
            max-width: 920px;
            margin: 0 auto 80px;
            padding: 0 24px;
        }

        .section-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 12px;
        }
        .section-title {
            font-family: var(--font-display);
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 32px;
        }

        /* Upload drop zone */
        .upload-zone {
            border: 1.5px dashed rgba(56,189,248,0.25);
            border-radius: 20px;
            padding: 60px 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(13,17,23,0.6);
            position: relative;
            overflow: hidden;
        }
        .upload-zone::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at center, rgba(56,189,248,0.04), transparent 70%);
            pointer-events: none;
        }
        .upload-zone:hover, .upload-zone.drag-over {
            border-color: var(--accent);
            background: rgba(56,189,248,0.05);
            box-shadow: 0 0 40px rgba(56,189,248,0.08), inset 0 0 40px rgba(56,189,248,0.03);
        }
        .upload-zone:hover .upload-icon-wrap { transform: translateY(-4px); }
        .upload-icon-wrap {
            width: 72px; height: 72px;
            margin: 0 auto 20px;
            background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(129,140,248,0.12));
            border: 1px solid rgba(56,189,248,0.2);
            border-radius: 20px;
            display: flex; align-items: center; justify-content: center;
            font-size: 28px;
            color: var(--accent);
            transition: transform 0.3s ease;
        }
        .upload-title {
            font-family: var(--font-display);
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .upload-sub {
            color: var(--text-secondary);
            font-size: 14px;
            font-family: var(--font-mono);
        }
        .upload-formats {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 20px;
        }
        .fmt-tag {
            background: rgba(56,189,248,0.08);
            border: 1px solid rgba(56,189,248,0.15);
            color: var(--accent);
            font-family: var(--font-mono);
            font-size: 10px;
            padding: 4px 10px;
            border-radius: 4px;
            letter-spacing: 1px;
        }
        input[type=file] { display: none; }

        /* ── Analysis Panel ── */
        .analysis-panel {
            display: none;
            margin-top: 24px;
            gap: 24px;
        }
        .analysis-panel.show { display: grid; grid-template-columns: 1fr 1fr; }

        .panel-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
        }

        /* Image preview */
        .img-preview-wrap {
            position: relative;
            min-height: 280px;
            display: flex; align-items: center; justify-content: center;
            background: #08090f;
        }
        #imagePreview {
            max-width: 100%;
            max-height: 340px;
            display: block;
        }

        /* Scan overlay */
        .scan-overlay {
            position: absolute;
            inset: 0;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(6,8,16,0.85);
            backdrop-filter: blur(4px);
        }
        .scan-overlay.show { display: flex; }
        .scan-line {
            position: absolute;
            left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
            box-shadow: 0 0 12px var(--accent);
            animation: scan 2.5s ease-in-out infinite;
            top: 0;
        }
        @keyframes scan { 0% { top: 0; } 100% { top: 100%; } }
        .scan-grid {
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px);
            background-size: 20px 20px;
        }
        .scan-info {
            position: relative;
            text-align: center;
            z-index: 2;
        }
        .scan-spinner {
            width: 48px; height: 48px;
            border: 2px solid rgba(56,189,248,0.2);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .scan-label {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--accent);
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        .scan-sub {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 6px;
        }
        /* Corner marks */
        .corner { position: absolute; width: 20px; height: 20px; border-color: var(--accent); border-style: solid; opacity: 0.6; }
        .corner-tl { top: 12px; left: 12px; border-width: 2px 0 0 2px; }
        .corner-tr { top: 12px; right: 12px; border-width: 2px 2px 0 0; }
        .corner-bl { bottom: 12px; left: 12px; border-width: 0 0 2px 2px; }
        .corner-br { bottom: 12px; right: 12px; border-width: 0 2px 2px 0; }

        .panel-body { padding: 24px; }

        /* Result card */
        .result-display { display: none; }
        .result-display.show { display: block; }

        .verdict-banner {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 20px;
            border-radius: 14px;
            margin-bottom: 20px;
        }
        .verdict-banner.authentic {
            background: linear-gradient(135deg, rgba(52,211,153,0.1), rgba(52,211,153,0.05));
            border: 1px solid rgba(52,211,153,0.25);
        }
        .verdict-banner.deepfake {
            background: linear-gradient(135deg, rgba(248,113,113,0.1), rgba(248,113,113,0.05));
            border: 1px solid rgba(248,113,113,0.25);
        }
        .verdict-icon {
            width: 52px; height: 52px;
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 22px;
            flex-shrink: 0;
        }
        .authentic .verdict-icon { background: rgba(52,211,153,0.15); color: var(--accent3); }
        .deepfake .verdict-icon { background: rgba(248,113,113,0.15); color: var(--danger); }
        .verdict-text h3 {
            font-family: var(--font-display);
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .authentic .verdict-text h3 { color: var(--accent3); }
        .deepfake .verdict-text h3 { color: var(--danger); }
        .verdict-text p {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            letter-spacing: 0.5px;
        }

        /* Metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 20px;
        }
        .metric-block {
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .metric-block .m-label {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 6px;
        }
        .metric-block .m-val {
            font-family: var(--font-display);
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
        }
        .metric-block .m-val.green { color: var(--accent3); }
        .metric-block .m-val.red { color: var(--danger); }
        .metric-block .m-val.blue { color: var(--accent); }

        /* Confidence bar */
        .conf-section { margin-bottom: 20px; }
        .conf-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .conf-header span:first-child {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .conf-pct {
            font-family: var(--font-display);
            font-size: 16px;
            font-weight: 700;
            color: var(--accent);
        }
        .bar-track {
            height: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 10px;
            width: 0%;
            transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);
        }
        .bar-fill.authentic-bar { background: linear-gradient(90deg, var(--accent3), #6ee7b7); box-shadow: 0 0 8px rgba(52,211,153,0.4); }
        .bar-fill.deepfake-bar { background: linear-gradient(90deg, var(--danger), #fca5a5); box-shadow: 0 0 8px rgba(248,113,113,0.4); }

        /* Risk segments */
        .risk-segments {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin-bottom: 20px;
        }
        .risk-seg {
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
            text-align: center;
        }
        .risk-seg .rs-label {
            font-family: var(--font-mono);
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .risk-seg .rs-val {
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 500;
        }

        /* Action buttons */
        .action-row {
            display: flex;
            gap: 10px;
        }
        .btn {
            flex: 1;
            padding: 12px 16px;
            border: none;
            border-radius: 12px;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.5px;
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            gap: 8px;
            transition: all 0.2s ease;
        }
        .btn-outline {
            background: transparent;
