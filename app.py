from flask import Flask, render_template, request, Response, jsonify, send_file
import os
import cv2
import numpy as np
import time
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import logging
from threading import Thread

# Import your existing code functions
from og_code import (
    model, correct_sentence_nltk, translate_text, text_to_speech, 
    add_visualizations, create_translation_animation, add_real_time_visualization
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for camera processing
camera = None
output_frame = None
lock = None
stop_camera = False
current_word = ""
sentence = ""
last_detected_letter = None
letter_counter = 0
frame_counter = 0
letter_confidences = {}
LETTER_THRESHOLD = 10
WORD_FRAME_THRESHOLD = 15

# Setup color map for bounding boxes
np.random.seed(42)  # for reproducibility
color_map = {cls: tuple(map(int, np.random.randint(0, 255, 3))) for cls in range(len(model.names))}

def reset_detection_variables():
    global current_word, sentence, last_detected_letter, letter_counter, frame_counter, letter_confidences
    current_word = ""
    sentence = ""
    last_detected_letter = None
    letter_counter = 0
    frame_counter = 0
    letter_confidences = {}

def detect_sign_language():
    global camera, output_frame, stop_camera, current_word, sentence, last_detected_letter, letter_counter, frame_counter, letter_confidences
    
    logger.info("Starting sign language detection thread")
    
    while True:
        if stop_camera:
            logger.info("Stopping camera detection thread")
            break
            
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            continue
            
        # Process frame with YOLO model
        results = model(frame)
        detected_letter = None
        conf = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                if class_name.isalpha() and len(class_name) == 1 and conf > 0.7:
                    detected_letter = class_name.upper()
                    # Store confidence score
                    letter_confidences[detected_letter] = conf
                    
                    # Draw bounding box
                    color = color_map[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Add real-time visualization
        frame = add_real_time_visualization(frame, detected_letter, conf if detected_letter else 0, 
                                        current_word, sentence)
        
        # Process letters and words
        if detected_letter == last_detected_letter and detected_letter is not None:
            letter_counter += 1
            if letter_counter == LETTER_THRESHOLD:
                current_word += detected_letter
                letter_counter = 0
                frame_counter = 0
        else:
            letter_counter = 0
            
        frame_counter += 1
        
        if current_word and frame_counter > WORD_FRAME_THRESHOLD:
            sentence += current_word + " "
            current_word = ""
            frame_counter = 0
            
        last_detected_letter = detected_letter
        
        # Update the output frame
        output_frame = frame.copy()

def generate_frames():
    global output_frame
    while True:
        if output_frame is None:
            continue
            
        # Encode the frame in JPEG format
        (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue
            
        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for streaming video from the camera."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    """Start the camera and detection process."""
    global camera, stop_camera
    
    if camera is not None:
        return jsonify({"status": "Camera already running"})
    
    try:
        camera = cv2.VideoCapture(0)  # Use default camera
        stop_camera = False
        reset_detection_variables()
        
        # Start detection in a separate thread
        detection_thread = Thread(target=detect_sign_language)
        detection_thread.daemon = True
        detection_thread.start()
        
        return jsonify({"status": "Camera started successfully"})
    except Exception as e:
        logger.error(f"Error starting camera: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/stop_camera')
def stop_camera_route():
    """Stop the camera and detection process."""
    global camera, stop_camera
    
    if camera is None:
        return jsonify({"status": "Camera not running"})
    
    try:
        stop_camera = True
        time.sleep(1)  # Give time for the thread to stop
        camera.release()
        camera = None
        return jsonify({"status": "Camera stopped successfully"})
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/get_detected_text')
def get_detected_text():
    """Return the current detected text."""
    global sentence, current_word
    
    full_text = (sentence + current_word).strip()
    return jsonify({
        "raw_text": full_text,
        "processed_text": correct_sentence_nltk(full_text) if full_text else ""
    })

@app.route('/translate_text', methods=['POST'])
def translate_text_route():
    """Translate the detected text."""
    data = request.json
    text = data.get('text', '')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'mr')
    
    try:
        translated = translate_text(text, source_lang, target_lang)
        return jsonify({"translated_text": translated})
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech_route():
    """Convert text to speech and return audio file."""
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'en')
    
    try:
        # Modified to return audio file instead of playing it
        from gtts import gTTS
        tts = gTTS(text=text, lang=language)
        
        # Save to memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_mp3(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to base64 for sending to frontend
        audio_b64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return jsonify({"audio_data": audio_b64})
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/get_visualizations', methods=['POST'])
def get_visualizations():
    """Generate visualizations and return as an image."""
    data = request.json
    raw_sentence = data.get('raw_sentence', '')
    processed_sentence = data.get('processed_sentence', '')
    marathi_text = data.get('marathi_text', None)
    confidence_scores = data.get('confidence_scores', None)
    
    try:
        # Use global letter_confidences if none provided
        if confidence_scores is None and letter_confidences:
            confidence_scores = letter_confidences
            
        # Generate processing times for visualization
        processing_times = {
            "Detection": data.get('detection_time', 0.5),
            "Text Processing": data.get('processing_time', 0.2),
            "Translation": data.get('translation_time', 0.3),
            "TTS": data.get('tts_time', 0.1)
        }
        
        # Generate visualizations
        img = add_visualizations(raw_sentence, processed_sentence, marathi_text, confidence_scores, processing_times)
        
        # Convert PIL image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({"visualization_image": img_str})
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/create_animation', methods=['POST'])
def create_animation_route():
    """Create an animation of the translation process."""
    data = request.json
    text = data.get('text', '')
    confidence_scores = data.get('confidence_scores', None)
    
    try:
        # Create temporary file for animation
        output_path = "static/temp_animation.mp4"
        create_translation_animation(text, confidence_scores, output_path)
        
        return jsonify({"animation_path": output_path})
    except Exception as e:
        logger.error(f"Animation error: {str(e)}")
        return jsonify({"status": "Error", "message": str(e)})

@app.route('/reset_detection')
def reset_detection():
    """Reset all detection variables."""
    reset_detection_variables()
    return jsonify({"status": "Detection variables reset successfully"})

if __name__ == '__main__':
    # Create required directories
    os.makedirs('static', exist_ok=True)
    
    # Start the Flask application
    app.run(debug=False, host="0.0.0.0", port=5000)