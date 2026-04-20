import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
from PIL import Image
import numpy as np
import io
import os
import sqlite3
import base64
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Database setup
DATABASE_NAME = 'deepfake_detector.db'


def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            filename TEXT NOT NULL,
            image_data BLOB NOT NULL,
            prediction REAL NOT NULL,
            is_deepfake BOOLEAN NOT NULL,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            image_format TEXT,
            FOREIGN KEY (user_email) REFERENCES users (email)
        )
    ''')

    # Insert demo users if they don't exist
    demo_users = [
        ('demo@example.com', 'password123'),
        ('admin@deepfakedetector.com', 'admin123')
    ]

    for email, password in demo_users:
        cursor.execute('INSERT OR IGNORE INTO users (email, password) VALUES (?, ?)', (email, password))

    conn.commit()
    conn.close()


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def save_image_to_db(user_email, filename, image_data, prediction, is_deepfake, file_size, image_format):
    """Save image analysis results to database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO images (user_email, filename, image_data, prediction, is_deepfake, file_size, image_format)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_email, filename, image_data, prediction, is_deepfake, file_size, image_format))

    image_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return image_id


def get_user_images(user_email, limit=50):
    """Get user's analyzed images from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT id, filename, prediction, is_deepfake, analysis_date, file_size, image_format
        FROM images 
        WHERE user_email = ? 
        ORDER BY analysis_date DESC 
        LIMIT ?
    ''', (user_email, limit))

    images = cursor.fetchall()
    conn.close()
    return images


def get_image_data(image_id, user_email):
    """Get specific image data from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT image_data, filename, image_format 
        FROM images 
        WHERE id = ? AND user_email = ?
    ''', (image_id, user_email))

    result = cursor.fetchone()
    conn.close()
    return result


def delete_image_from_db(image_id, user_email):
    """Delete image from database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('DELETE FROM images WHERE id = ? AND user_email = ?', (image_id, user_email))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count > 0


# Initialize database on startup
init_database()

# Load the pre-trained model
try:
    model = load_model('deepfake_detector_model.tflite)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def preprocess_image(image):
    """
    Preprocess the uploaded image to match the model's input requirements.
    """
    # Resize image to 128x128 to match model input
    image = image.resize((128, 128))
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to numpy array and normalize pixel values to [0, 1]
    image_array = np.array(image) / 255.0
    # Add batch dimension for model input
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# HTML Templates
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detector Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .slide-menu {
            transform: translateX(-100%);
            transition: transform 0.3s ease-in-out;
        }

        .slide-menu.open {
            transform: translateX(0);
        }

        .overlay {
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease-in-out;
        }

        .overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .floating-animation {
            animation: floating 3s ease-in-out infinite;
        }

        @keyframes floating {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <!-- Overlay for mobile menu -->
    <div id="overlay" class="overlay fixed inset-0 bg-black bg-opacity-50 z-40"></div>

    <!-- Slide Menu -->
    <div id="slideMenu" class="slide-menu fixed left-0 top-0 h-full w-80 glass-effect z-50 p-6">
        <div class="flex justify-between items-center mb-8">
            <h2 class="text-2xl font-bold text-white">Menu</h2>
            <button id="closeMenu" class="text-white hover:text-gray-300 transition-colors">
                <i class="fas fa-times text-xl"></i>
            </button>
        </div>

        <nav class="space-y-4">
            <a href="#home" class="nav-link flex items-center space-x-3 text-white hover:text-blue-300 transition-colors p-3 rounded-lg hover:bg-white hover:bg-opacity-20">
                <i class="fas fa-home"></i>
                <span>Home</span>
            </a>
            <a href="#about" class="nav-link flex items-center space-x-3 text-white hover:text-blue-300 transition-colors p-3 rounded-lg hover:bg-white hover:bg-opacity-20">
                <i class="fas fa-info-circle"></i>
                <span>About</span>
            </a>
            <a href="#detector" class="nav-link flex items-center space-x-3 text-white hover:text-blue-300 transition-colors p-3 rounded-lg hover:bg-white hover:bg-opacity-20">
                <i class="fas fa-search"></i>
                <span>Detector</span>
            </a>
            {% if session.get('logged_in') %}
            <a href="#history" class="nav-link flex items-center space-x-3 text-white hover:text-blue-300 transition-colors p-3 rounded-lg hover:bg-white hover:bg-opacity-20">
                <i class="fas fa-history"></i>
                <span>Analysis History</span>
            </a>
            {% endif %}
            <a href="#contact" class="nav-link flex items-center space-x-3 text-white hover:text-blue-300 transition-colors p-3 rounded-lg hover:bg-white hover:bg-opacity-20">
                <i class="fas fa-envelope"></i>
                <span>Contact</span>
            </a>
        </nav>

        <div class="absolute bottom-6 left-6 right-6">
            <div id="authButtons" class="space-y-3">
                {% if session.get('logged_in') %}
                <p class="text-white text-sm mb-2">Welcome, {{ session.get('user_email', 'User') }}!</p>
                <form method="POST" action="/logout">
                    <button type="submit" class="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                        <i class="fas fa-sign-out-alt mr-2"></i>Sign Out
                    </button>
                </form>
                {% else %}
                <a href="/login" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300 block text-center">
                    <i class="fas fa-sign-in-alt mr-2"></i>Sign In
                </a>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Top Navigation Bar -->
    <nav class="fixed top-0 left-0 right-0 glass-effect z-30 p-4">
        <div class="flex justify-between items-center max-w-7xl mx-auto">
            <button id="menuToggle" class="text-white hover:text-gray-300 transition-colors">
                <i class="fas fa-bars text-xl"></i>
            </button>
            <h1 class="text-xl font-bold text-white">Deepfake Detector Pro</h1>
            <div class="flex items-center space-x-4">
                {% if session.get('logged_in') %}
                <span class="text-white text-sm">Welcome, {{ session.get('user_email', 'User').split('@')[0] }}!</span>
                <i class="fas fa-user-check text-xl text-green-400"></i>
                {% else %}
                <a href="/login" class="text-white hover:text-blue-300 transition-colors">
                    <i class="fas fa-user-circle text-xl"></i>
                </a>
                {% endif %}
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="pt-20 px-4 pb-8">

        <!-- Home Section -->
        <section id="home" class="section-content min-h-screen flex items-center justify-center">
            <div class="glass-effect p-8 rounded-2xl shadow-2xl w-full max-w-4xl text-center">
                <div class="floating-animation mb-8">
                    <i class="fas fa-shield-alt text-6xl text-blue-300 mb-4"></i>
                </div>
                <h1 class="text-5xl font-bold text-white mb-4">Deepfake Detector Pro</h1>
                <p class="text-xl text-gray-200 mb-8 max-w-2xl mx-auto">
                    Advanced AI-powered technology to detect manipulated media and protect digital authenticity
                </p>
                <div class="flex flex-col sm:flex-row gap-4 justify-center">
                    <button onclick="showSection('detector')" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-full transition duration-300 transform hover:scale-105">
                        <i class="fas fa-search mr-2"></i>Start Detection
                    </button>
                    <button onclick="showSection('about')" class="border-2 border-white text-white hover:bg-white hover:text-purple-600 font-bold py-3 px-8 rounded-full transition duration-300">
                        <i class="fas fa-info-circle mr-2"></i>Learn More
                    </button>
                </div>
            </div>
        </section>

        <!-- About Section -->
        <section id="about" class="section-content hidden min-h-screen py-12">
            <div class="max-w-6xl mx-auto space-y-8">
                <div class="glass-effect p-8 rounded-2xl shadow-2xl text-center">
                    <h2 class="text-4xl font-bold text-white mb-6">About Our Technology</h2>
                    <p class="text-lg text-gray-200 max-w-3xl mx-auto">
                        Our deepfake detection system uses cutting-edge machine learning to identify manipulated media with high accuracy.
                    </p>
                </div>

                <div class="grid md:grid-cols-2 gap-8">
                    <div class="glass-effect p-6 rounded-xl">
                        <h3 class="text-2xl font-bold text-white mb-4">
                            <i class="fas fa-brain text-blue-300 mr-3"></i>AI Model Details
                        </h3>
                        <div class="text-gray-200 space-y-3">
                            <p><strong>Architecture:</strong> Convolutional Neural Network (CNN)</p>
                            <p><strong>Input Size:</strong> 128x128 RGB images</p>
                            <p><strong>Training Data:</strong> Specialized deepfake detection dataset</p>
                            <p><strong>Threshold:</strong> 70% confidence for deepfake classification</p>
                            <p><strong>Supported Formats:</strong> JPG, PNG, WEBP, BMP, GIF</p>
                        </div>
                    </div>

                    <div class="glass-effect p-6 rounded-xl">
                        <h3 class="text-2xl font-bold text-white mb-4">
                            <i class="fas fa-cog text-green-300 mr-3"></i>How It Works
                        </h3>
                        <div class="text-gray-200 space-y-3">
                            <div class="flex items-start space-x-3">
                                <span class="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">1</span>
                                <p>Image resizing to 128x128 pixels</p>
                            </div>
                            <div class="flex items-start space-x-3">
                                <span class="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">2</span>
                                <p>RGB conversion and normalization</p>
                            </div>
                            <div class="flex items-start space-x-3">
                                <span class="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">3</span>
                                <p>Deep neural network analysis</p>
                            </div>
                            <div class="flex items-start space-x-3">
                                <span class="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">4</span>
                                <p>Confidence-based classification</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Detector Section -->
        <section id="detector" class="section-content hidden min-h-screen flex items-center justify-center">
            <div class="glass-effect p-8 rounded-2xl shadow-2xl w-full max-w-2xl text-center">
                <h2 class="text-3xl font-bold text-white mb-2">Deepfake Detection</h2>
                <p class="text-gray-200 mb-6">
                    Upload an image to check if it's been manipulated using deepfake technology.
                </p>

                {% if not session.get('logged_in') %}
                <div class="mb-6 p-4 bg-yellow-500 bg-opacity-20 border border-yellow-400 rounded-lg">
                    <i class="fas fa-info-circle text-yellow-400 mr-2"></i>
                    <span class="text-yellow-200">Please <a href="/login" class="text-yellow-400 underline">sign in</a> to use the detector</span>
                </div>
                {% endif %}

                <!-- File Upload Input -->
                <input type="file" id="imageInput" accept="image/*" class="hidden" {% if not session.get('logged_in') %}disabled{% endif %}>
                <label for="imageInput" class="cursor-pointer bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-4 px-8 rounded-full transition duration-300 ease-in-out shadow-lg transform hover:scale-105 inline-block {% if not session.get('logged_in') %}opacity-50 cursor-not-allowed{% endif %}">
                    <i class="fas fa-upload mr-2"></i>Choose an Image
                </label>

                <!-- Image Preview Container -->
                <div id="imagePreviewContainer" class="mt-8 hidden">
                    <h3 class="text-xl font-semibold text-white mb-4">Image Preview:</h3>
                    <div class="relative">
                        <img id="imagePreview" src="" alt="Image Preview" class="w-full h-auto rounded-lg shadow-lg object-contain max-h-96 mx-auto">
                        <div id="loadingOverlay" class="absolute inset-0 bg-black bg-opacity-50 rounded-lg hidden flex items-center justify-center">
                            <div class="text-white text-center">
                                <i class="fas fa-spinner fa-spin text-4xl mb-4"></i>
                                <p class="text-lg">Analyzing image...</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Result Section -->
                <div id="resultContainer" class="mt-8 p-6 rounded-xl hidden">
                    <div id="resultIcon" class="text-4xl mb-4"></div>
                    <p id="resultText" class="text-lg font-bold"></p>
                    <div id="confidenceBar" class="mt-4 bg-gray-300 rounded-full h-4 overflow-hidden">
                        <div id="confidenceLevel" class="h-full transition-all duration-1000 ease-out"></div>
                    </div>
                    <p id="confidenceText" class="text-sm mt-2 text-gray-300"></p>
                    <div class="mt-4">
                        <p class="text-sm text-gray-300">Analysis saved to your history</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- History Section -->
        {% if session.get('logged_in') %}
        <section id="history" class="section-content hidden min-h-screen py-12">
            <div class="max-w-6xl mx-auto">
                <div class="glass-effect p-8 rounded-2xl shadow-2xl">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-3xl font-bold text-white">Analysis History</h2>
                        <button onclick="refreshHistory()" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors">
                            <i class="fas fa-refresh mr-2"></i>Refresh
                        </button>
                    </div>

                    <div id="historyContainer" class="space-y-4">
                        <div class="text-center text-gray-300">
                            <i class="fas fa-spinner fa-spin text-2xl mb-2"></i>
                            <p>Loading your analysis history...</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        {% endif %}

        <!-- Contact Section -->
        <section id="contact" class="section-content hidden min-h-screen flex items-center justify-center">
            <div class="glass-effect p-8 rounded-2xl shadow-2xl w-full max-w-2xl text-center">
                <h2 class="text-3xl font-bold text-white mb-6">Contact Us</h2>
                <div class="text-gray-200 space-y-4">
                    <p><i class="fas fa-envelope text-blue-300 mr-3"></i>support@deepfakedetector.com</p>
                    <p><i class="fas fa-phone text-green-300 mr-3"></i>+917417845421</p>
                    <p><i class="fas fa-map-marker-alt text-red-300 mr-3"></i>123 Tech Street, AI City, TC 12345</p>
                </div>
            </div>
        </section>
    </div>

    <script>
        // Menu functionality
        const menuToggle = document.getElementById('menuToggle');
        const closeMenu = document.getElementById('closeMenu');
        const slideMenu = document.getElementById('slideMenu');
        const overlay = document.getElementById('overlay');
        const navLinks = document.querySelectorAll('.nav-link');

        menuToggle.addEventListener('click', () => {
            slideMenu.classList.add('open');
            overlay.classList.add('active');
        });

        closeMenu.addEventListener('click', () => {
            slideMenu.classList.remove('open');
            overlay.classList.remove('active');
        });

        overlay.addEventListener('click', () => {
            slideMenu.classList.remove('open');
            overlay.classList.remove('active');
        });

        // Navigation functionality
        function showSection(sectionId) {
            const sections = document.querySelectorAll('.section-content');
            sections.forEach(section => {
                section.classList.add('hidden');
            });
            document.getElementById(sectionId).classList.remove('hidden');

            // Load history if showing history section
            if (sectionId === 'history' && {{ 'true' if session.get('logged_in') else 'false' }}) {
                loadHistory();
            }

            // Close menu on mobile
            slideMenu.classList.remove('open');
            overlay.classList.remove('active');
        }

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                showSection(targetId);
            });
        });

        // History functionality
        async function loadHistory() {
            const historyContainer = document.getElementById('historyContainer');

            try {
                const response = await fetch('/api/history');
                if (!response.ok) throw new Error('Failed to load history');

                const data = await response.json();

                if (data.images.length === 0) {
                    historyContainer.innerHTML = `
                        <div class="text-center text-gray-300 py-8">
                            <i class="fas fa-image text-4xl mb-4"></i>
                            <p class="text-lg">No analysis history yet</p>
                            <p class="text-sm">Upload and analyze some images to see them here</p>
                        </div>
                    `;
                } else {
                    historyContainer.innerHTML = data.images.map(image => `
                        <div class="glass-effect p-4 rounded-lg flex items-center justify-between fade-in">
                            <div class="flex items-center space-x-4">
                                <img src="/api/image/${image.id}" alt="${image.filename}" class="w-16 h-16 object-cover rounded-lg">
                                <div>
                                    <h4 class="text-white font-medium">${image.filename}</h4>
                                    <p class="text-sm text-gray-300">${new Date(image.analysis_date).toLocaleDateString()}</p>
                                    <p class="text-sm ${image.is_deepfake ? 'text-red-400' : 'text-green-400'}">
                                        <i class="fas ${image.is_deepfake ? 'fa-exclamation-triangle' : 'fa-check-circle'} mr-1"></i>
                                        ${image.is_deepfake ? 'Deepfake Detected' : 'Authentic'}
                                    </p>
                                </div>
                            </div>
                            <div class="flex items-center space-x-3">
                                <div class="text-right">
                                    <p class="text-sm text-gray-300">Confidence</p>
                                    <p class="text-white font-bold">${Math.round(image.is_deepfake ? image.prediction * 100 : (1 - image.prediction) * 100)}%</p>
                                </div>
                                <button onclick="deleteImage(${image.id})" class="text-red-400 hover:text-red-300 transition-colors">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading history:', error);
                historyContainer.innerHTML = `
                    <div class="text-center text-red-300 py-8">
                        <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                        <p>Error loading history</p>
                    </div>
                `;
            }
        }

        function refreshHistory() {
            loadHistory();
        }

        async function deleteImage(imageId) {
            if (!confirm('Are you sure you want to delete this image?')) return;

            try {
                const response = await fetch(`/api/image/${imageId}`, {
                    method: 'DELETE'
                });

                if (response.ok) {
                    loadHistory(); // Refresh the list
                } else {
                    alert('Failed to delete image');
                }
            } catch (error) {
                console.error('Error deleting image:', error);
                alert('Error deleting image');
            }
        }

        // Deepfake detection functionality
        const imageInput = document.getElementById('imageInput');
        const imagePreviewContainer = document.getElementById('imagePreviewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const resultContainer = document.getElementById('resultContainer');
        const resultText = document.getElementById('resultText');
        const resultIcon = document.getElementById('resultIcon');
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceLevel = document.getElementById('confidenceLevel');
        const confidenceText = document.getElementById('confidenceText');
        const loadingOverlay = document.getElementById('loadingOverlay');

        imageInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                displayImagePreview(file);
                processImageForDeepfake(file);
            }
        });

        function displayImagePreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
                resultContainer.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }

        async function processImageForDeepfake(file) {
            // Show loading
            loadingOverlay.classList.remove('hidden');

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                loadingOverlay.classList.add('hidden');

                // Display result with animation
                setTimeout(() => {
                    resultContainer.classList.remove('hidden');

                    if (data.is_deepfake) {
                        resultContainer.className = 'mt-8 p-6 rounded-xl glass-effect border-2 border-red-400';
                        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle text-red-400"></i>';
                        resultText.innerHTML = `<span class="text-red-400">DEEPFAKE DETECTED</span>`;
                        resultText.className = 'text-lg font-bold';
                        confidenceLevel.className = 'h-full bg-gradient-to-r from-red-500 to-red-600 transition-all duration-1000 ease-out';
                        const confidence = Math.round(data.prediction * 100);
                        confidenceLevel.style.width = `${confidence}%`;
                        confidenceText.textContent = `Confidence: ${confidence}%`;
                    } else {
                        resultContainer.className = 'mt-8 p-6 rounded-xl glass-effect border-2 border-green-400';
                        resultIcon.innerHTML = '<i class="fas fa-check-circle text-green-400"></i>';
                        resultText.innerHTML = `<span class="text-green-400">IMAGE APPEARS AUTHENTIC</span>`;
                        resultText.className = 'text-lg font-bold';
                        confidenceLevel.className = 'h-full bg-gradient-to-r from-green-500 to-green-600 transition-all duration-1000 ease-out';
                        const confidence = Math.round((1 - data.prediction) * 100);
                        confidenceLevel.style.width = `${confidence}%`;
                        confidenceText.textContent = `Confidence: ${confidence}%`;
                    }
                }, 100);

            } catch (error) {
                console.error('Error:', error);
                loadingOverlay.classList.add('hidden');

                setTimeout(() => {
                    resultContainer.classList.remove('hidden');
                    resultContainer.className = 'mt-8 p-6 rounded-xl glass-effect border-2 border-red-400';
                    resultIcon.innerHTML = '<i class="fas fa-exclamation-circle text-red-400"></i>';
                    resultText.innerHTML = `<span class="text-red-400">ERROR OCCURRED</span>`;
                    resultText.className = 'text-lg font-bold';
                    confidenceText.textContent = `Error: ${error.message}`;
                }, 100);
            }
        }

        // Initialize app
        showSection('home');
    </script>
</body>
</html>
"""

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Deepfake Detector Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .floating-animation {
            animation: floating 3s ease-in-out infinite;
        }

        @keyframes floating {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .slide-in {
            animation: slideIn 0.5s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body class="flex items-center justify-center min-h-screen p-4">

    <!-- Back Button -->
    <a href="/" class="absolute top-6 left-6 text-white hover:text-gray-300 transition-colors z-10">
        <i class="fas fa-arrow-left text-2xl"></i>
    </a>

    <!-- Login Container -->
    <div class="glass-effect p-8 rounded-2xl shadow-2xl w-full max-w-md slide-in">

        <!-- Logo/Icon -->
        <div class="text-center mb-8">
            <div class="floating-animation">
                <i class="fas fa-shield-alt text-5xl text-blue-300 mb-4"></i>
            </div>
            <h1 class="text-3xl font-bold text-white mb-2">Welcome Back</h1>
            <p class="text-gray-200">Sign in to your account</p>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="mb-4 p-4 rounded-lg {% if category == 'error' %}bg-red-500 bg-opacity-20 border border-red-400{% else %}bg-green-500 bg-opacity-20 border border-green-400{% endif %}">
                        <p class="{% if category == 'error' %}text-red-400{% else %}text-green-400{% endif %}">{{ message }}</p>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <!-- Login Form -->
        <form method="POST" action="/login" class="space-y-6">

            <!-- Email Field -->
            <div>
                <label for="email" class="block text-white text-sm font-medium mb-2">
                    <i class="fas fa-envelope mr-2"></i>Email Address
                </label>
                <input 
                    type="email" 
                    id="email" 
                    name="email" 
                    required
                    class="w-full px-4 py-3 rounded-lg bg-white bg-opacity-20 border border-white border-opacity-30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all"
                    placeholder="Enter your email"
                >
            </div>

            <!-- Password Field -->
            <div>
                <label for="password" class="block text-white text-sm font-medium mb-2">
                    <i class="fas fa-lock mr-2"></i>Password
                </label>
                <div class="relative">
                    <input 
                        type="password" 
                        id="password" 
                        name="password" 
                        required
                        class="w-full px-4 py-3 rounded-lg bg-white bg-opacity-20 border border-white border-opacity-30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent transition-all pr-12"
                        placeholder="Enter your password"
                    >
                    <button 
                        type="button" 
                        id="togglePassword" 
                        class="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-300 hover:text-white transition-colors"
                    >
                        <i class="fas fa-eye" id="toggleIcon"></i>
                    </button>
                </div>
            </div>

            <!-- Remember Me -->
            <div class="flex items-center justify-between">
                <label class="flex items-center text-white">
                    <input 
                        type="checkbox" 
                        name="remember"
                        class="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    >
                    <span class="text-sm">Remember me</span>
                </label>
            </div>

            <!-- Submit Button -->
            <button 
                type="submit" 
                class="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 transform hover:scale-105 shadow-lg"
            >
                <i class="fas fa-sign-in-alt mr-2"></i>Sign In
            </button>
        </form>

        <!-- Demo Credentials -->
        <div class="mt-6 p-4 bg-blue-500 bg-opacity-20 border border-blue-400 rounded-lg">
            <h3 class="text-blue-400 font-medium mb-2">Demo Credentials:</h3>
            <p class="text-blue-300 text-sm">Email: demo@example.com</p>
            <p class="text-blue-300 text-sm">Password: password123</p>
        </div>
    </div>

    <script>
        // Password toggle functionality
        const togglePassword = document.getElementById('togglePassword');
        const passwordInput = document.getElementById('password');
        const toggleIcon = document.getElementById('toggleIcon');

        togglePassword.addEventListener('click', () => {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            toggleIcon.className = type === 'password' ? 'fas fa-eye' : 'fas fa-eye-slash';
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main page with the deepfake detector interface."""
    return render_template_string(INDEX_HTML)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember')

        # Check credentials against database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['user_email'] = email
            if remember:
                session.permanent = True
            return redirect(url_for('index'))
        else:
            from flask import flash
            flash('Invalid email or password', 'error')

    return render_template_string(LOGIN_HTML)


@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout."""
    session.clear()
    from flask import flash
    flash('Successfully logged out', 'success')
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive an image and return deepfake prediction."""
    # Check if user is logged in
    if not session.get('logged_in'):
        return jsonify({'error': 'Please log in to use the detector'}), 401

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # Read file data
        file_data = file.read()
        file_size = len(file_data)

        # Open image using PIL
        image = Image.open(io.BytesIO(file_data))
        image_format = image.format

        # Preprocess the image
        image_array = preprocess_image(image)

        # Make prediction
        prediction = model.predict(image_array)[0][0]
        is_deepfake = prediction > 0.7

        # Save to database
        user_email = session.get('user_email')
        image_id = save_image_to_db(
            user_email=user_email,
            filename=file.filename,
            image_data=file_data,
            prediction=float(prediction),
            is_deepfake=bool(is_deepfake),
            file_size=file_size,
            image_format=image_format
        )

        # Return JSON response
        return jsonify({
            'prediction': float(prediction),
            'is_deepfake': bool(is_deepfake),
            'threshold': 0.7,
            'image_id': image_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """Get user's image analysis history."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401

    user_email = session.get('user_email')
    images = get_user_images(user_email)

    # Convert to list of dictionaries
    image_list = []
    for image in images:
        image_list.append({
            'id': image['id'],
            'filename': image['filename'],
            'prediction': image['prediction'],
            'is_deepfake': bool(image['is_deepfake']),
            'analysis_date': image['analysis_date'],
            'file_size': image['file_size'],
            'image_format': image['image_format']
        })

    return jsonify({'images': image_list})


@app.route('/api/image/<int:image_id>')
def get_image(image_id):
    """Get image data by ID."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401

    user_email = session.get('user_email')
    image_data = get_image_data(image_id, user_email)

    if not image_data:
        return jsonify({'error': 'Image not found'}), 404

    from flask import Response
    return Response(
        image_data['image_data'],
        mimetype=f'image/{image_data["image_format"].lower()}' if image_data['image_format'] else 'image/jpeg'
    )


@app.route('/api/image/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """Delete image by ID."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Not logged in'}), 401

    user_email = session.get('user_email')
    success = delete_image_from_db(image_id, user_email)

    if success:
        return jsonify({'message': 'Image deleted successfully'})
    else:
        return jsonify({'error': 'Image not found'}), 404


if __name__ == '__main__':
    # Set session permanent lifetime
    from datetime import timedelta

    app.permanent_session_lifetime = timedelta(days=7)

    print("Starting Deepfake Detector Pro with Database...")
    print("Database initialized successfully!")
    print("Demo credentials: demo@example.com / password123")
    print("Admin credentials: admin@deepfakedetector.com / admin123")

    app.run(debug=True, host='0.0.0.0', port=5000)
