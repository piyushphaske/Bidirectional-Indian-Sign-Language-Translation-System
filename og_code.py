import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
import cv2
from collections import Counter


def add_visualizations(raw_sentence, processed_sentence, marathi_text=None, confidence_scores=None, processing_times=None):
    """
    Generate visualizations after translation task execution
    
    Parameters:
    - raw_sentence: The raw detected sentence from sign language
    - processed_sentence: The corrected/processed sentence
    - marathi_text: Translated Marathi text (optional)
    - confidence_scores: Dictionary of {letter: confidence_score} (optional)
    - processing_times: Dictionary with processing time metrics (optional)
    
    Returns:
    - PIL Image object with the visualization
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    from PIL import Image
    from wordcloud import WordCloud
    
    # Helper functions for individual visualizations
    def text_processing_visualization(raw, processed, translated=None):
        plt.title('Text Processing Comparison')
        plt.axis('off')
        
        text_content = f"Raw Text:\n{raw}\n\n"
        text_content += f"Processed Text:\n{processed}\n\n"
        
        if translated:
            text_content += f"Translated Text:\n{translated}"
            
        plt.text(0.1, 0.5, text_content, wrap=True, fontsize=10, 
                 verticalalignment='center')
    
    def letter_confidence_visualization(confidence_data):
        plt.title('Letter Detection Confidence')
        letters = list(confidence_data.keys())
        scores = list(confidence_data.values())
        
        plt.bar(letters, scores, color='skyblue')
        plt.ylabel('Confidence Score')
        plt.ylim(0, 1.1)  # Confidence scores are typically 0-1
        plt.xticks(rotation=45)
    
    def generate_word_cloud(text):
        if not text or len(text.strip()) == 0:
            plt.title('Word Cloud (No text available)')
            return
            
        plt.title('Word Cloud')
        # Simple word cloud 
        wordcloud = WordCloud(width=400, height=300, background_color='white', 
                              min_font_size=10).generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
    
    def processing_time_visualization(times):
        plt.title('Processing Time Breakdown')
        tasks = list(times.keys())
        durations = list(times.values())
        
        plt.bar(tasks, durations, color='lightgreen')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
    
    def character_frequency_visualization(text):
        plt.title('Character Frequency')
        if not text or len(text.strip()) == 0:
            plt.text(0.5, 0.5, 'No text available for analysis', 
                    horizontalalignment='center')
            return
            
        # Count character frequencies
        char_counts = {}
        for char in text.upper():
            if char.isalpha():  # Only count letters
                if char in char_counts:
                    char_counts[char] += 1
                else:
                    char_counts[char] = 1
        
        if not char_counts:
            plt.text(0.5, 0.5, 'No letters found in text', 
                    horizontalalignment='center')
            return
            
        chars = list(char_counts.keys())
        counts = list(char_counts.values())
        
        plt.bar(chars, counts, color='salmon')
        plt.ylabel('Frequency')
    
    # Create a figure with subplots for all visualizations
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Text Processing Comparison Visualization
    plt.subplot(2, 3, 1)
    text_processing_visualization(raw_sentence, processed_sentence, marathi_text)
    
    # 2. Translation Confidence Visualization
    plt.subplot(2, 3, 2)
    if confidence_scores and isinstance(confidence_scores, dict) and len(confidence_scores) > 0:
        letter_confidence_visualization(confidence_scores)
    else:
        # Generate dummy data if not provided
        dummy_confidence = {letter: np.random.uniform(0.7, 1.0) for letter in set(raw_sentence.upper()) if letter.isalpha()}
        if dummy_confidence:
            letter_confidence_visualization(dummy_confidence)
        else:
            plt.title('Letter Detection Confidence')
            plt.text(0.5, 0.5, 'No confidence data available', 
                    horizontalalignment='center', verticalalignment='center')
    
    # 3. Word Cloud Visualization
    plt.subplot(2, 3, 3)
    try:
        generate_word_cloud(processed_sentence)
    except ImportError:
        plt.title('Word Cloud (WordCloud library not available)')
        plt.axis('off')
    
    # 4. Processing Time Breakdown
    plt.subplot(2, 3, 4)
    if processing_times and isinstance(processing_times, dict) and len(processing_times) > 0:
        processing_time_visualization(processing_times)
    else:
        # Generate dummy data if not provided
        dummy_times = {
            "Detection": np.random.uniform(0.1, 0.5),
            "Text Processing": np.random.uniform(0.05, 0.2),
            "Translation": np.random.uniform(0.2, 0.8),
            "TTS": np.random.uniform(0.1, 0.3)
        }
        processing_time_visualization(dummy_times)
    
    # 5. Character Frequency Visualization
    plt.subplot(2, 3, 5)
    character_frequency_visualization(raw_sentence)
    
    plt.tight_layout()
    
    # Save to BytesIO buffer and convert to PIL Image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)  # This is important to avoid memory leaks
    
    img_buf.seek(0)
    return Image.open(img_buf)


# Bonus animation visualization - can be used as a standalone visualization
def create_translation_animation(text, confidence_scores=None, output_path="translation_animation.mp4"):
    """
    Create an animation showing the letters being detected in sequence
    
    Parameters:
    - text: The text to animate
    - confidence_scores: Dictionary of {letter: confidence_score}
    - output_path: Path to save the animation video
    """
    # Generate random confidence scores if not provided
    if not confidence_scores:
        confidence_scores = {letter: np.random.uniform(0.7, 1.0) for letter in text}
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        
        # Show detected text so far
        detected_text = text[:frame+1]
        remaining_text = text[frame+1:] if frame+1 < len(text) else ""
        
        # Set up the plot
        ax.text(0.1, 0.5, detected_text, fontsize=24, color='blue')
        ax.text(0.1 + len(detected_text) * 0.025, 0.5, remaining_text, fontsize=24, color='gray', alpha=0.5)
        
        # Show confidence for current letter
        if frame < len(text):
            current_letter = text[frame]
            confidence = confidence_scores.get(current_letter, 0.8)
            ax.text(0.1, 0.3, f"Detecting: {current_letter}", fontsize=16)
            ax.text(0.1, 0.2, f"Confidence: {confidence:.2f}", fontsize=16)
            
            # Add a confidence bar
            ax.barh(0.1, confidence, height=0.05, color='green')
            ax.barh(0.1, 1.0, height=0.05, color='lightgray', alpha=0.3)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Sign Language Detection Animation')
        
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(text) + 10, interval=500)
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', fps=2, dpi=100)
    
    plt.close()
    return output_path


# Real-time visualization for webcam input
def add_real_time_visualization(frame, detected_letter, confidence, current_word, sentence):
    """
    Add real-time visualization elements to the camera frame
    
    Parameters:
    - frame: The current camera frame
    - detected_letter: Currently detected letter
    - confidence: Confidence score for the detection
    - current_word: Current word being formed
    - sentence: Complete sentence formed so far
    
    Returns:
    - Modified frame with visualizations
    """
    h, w = frame.shape[:2]
    
    # Create a semi-transparent overlay for visualizations
    overlay = frame.copy()
    
    # Draw detection confidence bar
    if detected_letter is not None:
        # Background bar
        cv2.rectangle(overlay, (w-220, 20), (w-20, 50), (200, 200, 200), -1)
        # Confidence level
        conf_width = int(200 * confidence)
        cv2.rectangle(overlay, (w-220, 20), (w-220+conf_width, 50), (0, 255, 0), -1)
        # Text
        cv2.putText(overlay, f"{detected_letter}: {confidence:.2f}", 
                   (w-210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Create bottom info panel
    cv2.rectangle(overlay, (0, h-150), (w, h), (240, 240, 240), -1)
    
    # Current word with letter-by-letter color coding
    word_x = 20
    word_y = h-110
    cv2.putText(overlay, "Current Word:", (word_x, word_y-30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    for i, letter in enumerate(current_word):
        color = (0, 100, 200)  # Blue for letters in the word
        cv2.putText(overlay, letter, (word_x + i*30, word_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    # Sentence so far
    cv2.putText(overlay, "Sentence:", (20, h-60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display sentence with word wrapping
    max_width = w - 40
    words = sentence.split()
    line = ""
    line_y = h-30
    
    for word in words:
        test_line = line + word + " "
        # Check if adding word would exceed frame width
        text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        if text_size[0] > max_width:
            cv2.putText(overlay, line, (20, line_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            line = word + " "
            line_y += 30
        else:
            line = test_line
    
    # Print the last line
    cv2.putText(overlay, line, (20, line_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Combine the frame and overlay
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    
    return frame

def calculate_system_metrics(test_data=None):
    """
    Calculate and display various evaluation metrics for the sign language detection system
    
    Parameters:
    - test_data: Optional test data to evaluate against. If None, will use sample data or run a test.
    
    Returns:
    - Dictionary of metrics
    """
    # If no test data is provided, use sample data for demonstration
    if test_data is None:
        print("No test data provided. Using sample evaluation data...")
        # Sample data - in a real system, this would come from evaluation runs
        y_true = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Simulate predictions with some errors
        y_pred = ['A', 'B', 'C', 'D', 'F', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 
                  'N', 'O', 'R', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Detection times (seconds)
        detection_times = np.random.uniform(0.05, 0.2, len(y_true))
        
        # Confidence scores (0-1)
        confidence_scores = np.random.uniform(0.7, 0.99, len(y_true))
    else:
        # Use provided test data
        y_true = test_data['true_labels']
        y_pred = test_data['predicted_labels']
        detection_times = test_data['detection_times']
        confidence_scores = test_data['confidence_scores']
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    
    # Calculate additional metrics
    avg_detection_time = np.mean(detection_times)
    avg_confidence = np.mean(confidence_scores)
    
    # Error rate
    error_rate = 1 - accuracy
    
    # Calculate per-letter accuracy
    letter_metrics = {}
    unique_letters = sorted(set(y_true))
    
    for letter in unique_letters:
        indices = [i for i, l in enumerate(y_true) if l == letter]
        correct = sum(1 for i in indices if y_pred[i] == y_true[i])
        total = len(indices)
        letter_accuracy = correct / total if total > 0 else 0
        avg_letter_conf = np.mean([confidence_scores[i] for i in indices]) if indices else 0
        letter_metrics[letter] = {
            'accuracy': letter_accuracy,
            'avg_confidence': avg_letter_conf,
            'sample_count': total
        }
    
    # Compile all metrics
    all_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'error_rate': error_rate,
        'avg_detection_time': avg_detection_time,
        'avg_confidence': avg_confidence,
        'confusion_matrix': cm,
        'letter_metrics': letter_metrics
    }
    
    return all_metrics

def display_evaluation_metrics(metrics=None):
    """
    Display system evaluation metrics with visualizations
    
    Parameters:
    - metrics: Dictionary of metrics. If None, will calculate metrics.
    """
    if metrics is None:
        metrics = calculate_system_metrics()
    
    # Create figure with subplots
    plt.figure(figsize=(18, 12))
    
    # 1. Overall metrics bar chart
    plt.subplot(2, 3, 1)
    overall_metrics = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    plt.bar(metric_labels, overall_metrics, color=['steelblue', 'forestgreen', 'darkorange', 'purple'])
    plt.ylim(0, 1)
    plt.title('Overall System Performance')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for i, v in enumerate(overall_metrics):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Confusion Matrix Heatmap
    plt.subplot(2, 3, 2)
    cm = metrics['confusion_matrix']
    unique_letters = sorted(metrics['letter_metrics'].keys())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_letters, yticklabels=unique_letters)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 3. Per-letter Accuracy
    plt.subplot(2, 3, 3)
    letters = list(metrics['letter_metrics'].keys())
    letter_accuracies = [metrics['letter_metrics'][l]['accuracy'] for l in letters]
    
    # Create gradient colors based on accuracy
    colors = plt.cm.RdYlGn(np.array(letter_accuracies))
    
    plt.bar(letters, letter_accuracies, color=colors)
    plt.axhline(y=metrics['accuracy'], color='r', linestyle='--', label=f'Overall Accuracy: {metrics["accuracy"]:.3f}')
    plt.ylim(0, 1.1)
    plt.title('Per-Letter Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 4. Detection Time vs Confidence Analysis
    plt.subplot(2, 3, 4)
    letter_times = []
    letter_confs = []
    sizes = []
    
    for letter in letters:
        metrics_data = metrics['letter_metrics'][letter]
        letter_times.append(metrics['avg_detection_time'])  # Using overall average as example
        letter_confs.append(metrics_data['avg_confidence'])
        sizes.append(metrics_data['sample_count'] * 20)  # Scale by sample count
    
    plt.scatter(letter_times, letter_confs, s=sizes, alpha=0.6, c=letter_accuracies, cmap='viridis')
    
    for i, letter in enumerate(letters):
        plt.annotate(letter, (letter_times[i], letter_confs[i]), fontsize=9)
    
    plt.title('Detection Time vs. Confidence')
    plt.xlabel('Detection Time (s)')
    plt.ylabel('Confidence Score')
    plt.colorbar(label='Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 5. System Performance Summary
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    summary_text = f"""
    System Performance Summary:
    
    • Overall Accuracy: {metrics['accuracy']:.3f}
    • Precision: {metrics['precision']:.3f}
    • Recall: {metrics['recall']:.3f}
    • F1 Score: {metrics['f1_score']:.3f}
    • Error Rate: {metrics['error_rate']:.3f}
    
    Speed Metrics:
    • Avg. Detection Time: {metrics['avg_detection_time']:.3f}s
    
    Confidence Metrics:
    • Avg. Confidence Score: {metrics['avg_confidence']:.3f}
    
    Most Accurate Letters:
    {get_top_n_letters(metrics, 'accuracy', 3)}
    
    Least Accurate Letters:
    {get_top_n_letters(metrics, 'accuracy', 3, bottom=True)}
    """
    
    plt.text(0, 1, summary_text, fontsize=12, va='top')
    
    # 6. Word-level Performance Estimation (Simulated)
    plt.subplot(2, 3, 6)
    
    # Simulate word accuracy based on letter accuracy
    # Word accuracy typically decreases with word length due to compounding errors
    word_lengths = list(range(1, 11))
    word_accuracies = [metrics['accuracy'] ** length for length in word_lengths]
    
    plt.plot(word_lengths, word_accuracies, 'o-', color='teal')
    plt.title('Estimated Word Recognition Accuracy')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Estimated Accuracy')
    plt.grid(True)
    plt.xticks(word_lengths)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("system_evaluation_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print out key metrics to console
    print("\nSYSTEM EVALUATION METRICS SUMMARY:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Average Detection Time: {metrics['avg_detection_time']:.3f} seconds")
    print(f"Average Confidence Score: {metrics['avg_confidence']:.3f}")

def get_top_n_letters(metrics, metric_name, n=3, bottom=False):
    """Helper function to get top or bottom N letters based on a metric"""
    letters = list(metrics['letter_metrics'].keys())
    metric_values = [metrics['letter_metrics'][l][metric_name] for l in letters]
    
    # Sort letters based on metric values
    sorted_indices = np.argsort(metric_values)
    if not bottom:
        sorted_indices = sorted_indices[::-1]  # Reverse for top N
    
    result = ""
    for i in range(min(n, len(letters))):
        idx = sorted_indices[i]
        result += f"• {letters[idx]}: {metric_values[idx]:.3f}\n"
    
    return result

def collect_test_data(test_video_path=None, num_samples=None):
    """
    Collect real test data by processing a known test video or sample data
    
    Parameters:
    - test_video_path: Path to a test video with known ground truth
    - num_samples: Number of samples to process (if None, process all)
    
    Returns:
    - Dictionary with test data metrics
    """
    print("This function would normally gather real test data from a video with known ground truth.")
    print("For demonstration purposes, we're using simulated test data.")
    
    # In a real implementation, this would process actual test videos and compare with ground truth
    # Simulated test data
    test_data = {
        'true_labels': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                        'U', 'V', 'W', 'X', 'Y', 'Z'],
        'predicted_labels': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                            'U', 'V', 'W', 'X', 'Z', 'Y'],  # Added some errors
        'detection_times': np.random.uniform(0.05, 0.2, 26),
        'confidence_scores': np.random.uniform(0.7, 0.95, 26)
    }
    
    return test_data

import os
import cv2
import numpy as np
import torch
import warnings
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, concatenate_videoclips
from IPython.display import Video, display
import moviepy.editor as mpy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.metrics import edit_distance
from googletrans import Translator
from gtts import gTTS
import pygame
from io import BytesIO
import speech_recognition as sr
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from matplotlib.animation import FuncAnimation
from PIL import Image
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the trained YOLO model
model = YOLO(r'train5/weights/last.pt')
model.to('cpu')

# Generate a color map for classes
np.random.seed(42)  # for reproducibility
color_map = {cls: tuple(map(int, np.random.randint(0, 255, 3))) for cls in range(len(model.names))}

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def correct_sentence_nltk(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence.lower())
    pos_tags = pos_tag(tokens)
    
    corrected_tokens = []
    for token, pos in pos_tags:
        wordnet_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(token, wordnet_pos)
        corrected_tokens.append(lemma)
    
    # Modified: Don't remove single-letter words like "a" and "i"
    # Only remove stopwords that are not single letters
    stop_words = set(stopwords.words('english'))
    corrected_tokens = [word for word in corrected_tokens if word not in stop_words or (len(word) == 1 and word in ['a', 'i'])]
    
    corrected_sentence = ' '.join(corrected_tokens)
    return corrected_sentence.capitalize()

def process_video_from_camera():
    current_word = ""
    sentence = ""
    last_detected_letter = None
    letter_counter = 0
    frame_counter = 0
    no_detection_counter = 0
    LETTER_THRESHOLD = 10
    WORD_FRAME_THRESHOLD = 30  # Increased to give more time between words
    NO_DETECTION_THRESHOLD = 45  # Frames with no detection to finalize a word
    MAX_RECORDING_TIME = 60  # Maximum recording time in seconds
    
    # Initialize dictionary to store letter confidences
    letter_confidences = {}
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return "", "", letter_confidences
    
    print("Camera initialized successfully. Recording started...")
    print("Press 'q' to stop recording.")
    print("Make hand signs and pause briefly between different letters and words.")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Check if maximum recording time has elapsed
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_RECORDING_TIME:
            print(f"Maximum recording time of {MAX_RECORDING_TIME} seconds reached.")
            break
        
        # Display elapsed time on frame
        cv2.putText(frame, f"Time: {int(elapsed_time)}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        results = model(frame)
        detected_letter = None
        conf = 0  # Initialize conf to avoid reference errors

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                if class_name.isalpha() and len(class_name) == 1 and conf > 0.7:
                    detected_letter = class_name.upper()
                    # Store confidence score for visualization
                    letter_confidences[detected_letter] = conf
                    
                    # Draw bounding box with color based on class
                    color = color_map[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thickness of 3
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Add real-time visualization to the frame
        frame = add_real_time_visualization(frame, detected_letter, conf if detected_letter else 0, 
                                           current_word, sentence)

        # Display current word and sentence
        cv2.putText(frame, f"Current Word: {current_word}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {sentence}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Camera Input - Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped manually.")
            break

        if detected_letter is None:
            no_detection_counter += 1
            if no_detection_counter >= NO_DETECTION_THRESHOLD and current_word:
                # No detection for a while and we have a current word
                sentence += current_word + " "
                current_word = ""
                no_detection_counter = 0
                frame_counter = 0
        else:
            no_detection_counter = 0
            
            if detected_letter == last_detected_letter:
                letter_counter += 1
                if letter_counter == LETTER_THRESHOLD:
                    # Only add the letter if it's not already the last letter in the current word
                    # This prevents stuttering (repeating the same letter)
                    if not current_word or current_word[-1] != detected_letter:
                        current_word += detected_letter
                    letter_counter = 0
                    frame_counter = 0
            else:
                letter_counter = 0

        frame_counter += 1

        if current_word and frame_counter > WORD_FRAME_THRESHOLD:
            # If the current word looks like it might be a letter-by-letter spelling of a commonly used word
            # we can try to match it against common words
            common_words = {
                "CAP": "CAP",
                "HAT": "HAT",
                "HELLO": "HELLO",
                "HI": "HI",
                "THANK": "THANK",
                "YOU": "YOU",
                "GOOD": "GOOD",
                "BYE": "BYE",
                # Add more common words as needed
            }
            
            if current_word.upper() in common_words:
                current_word = common_words[current_word.upper()]
                
            sentence += current_word + " "
            current_word = ""
            frame_counter = 0

        last_detected_letter = detected_letter

    cap.release()
    cv2.destroyAllWindows()

    raw_sentence = sentence.strip()
    print(f"Raw detected sentence: {raw_sentence}")
    
    corrected_sentence = correct_sentence_nltk(raw_sentence)
    print(f"Corrected sentence: {corrected_sentence}")

    return raw_sentence, corrected_sentence, letter_confidences
    
def process_video_from_file(video_path):
    current_word = ""
    sentence = ""
    last_detected_letter = None
    letter_counter = 0
    frame_counter = 0
    LETTER_THRESHOLD = 10
    WORD_FRAME_THRESHOLD = 15

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detected_letter = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]

                if class_name.isalpha() and len(class_name) == 1 and conf > 0.7:
                    detected_letter = class_name.upper()
                    
                    # Draw bounding box with color based on class
                    color = color_map[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thickness of 3
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

    cap.release()
    cv2.destroyAllWindows()

    raw_sentence = sentence.strip()
    corrected_sentence = correct_sentence_nltk(raw_sentence)

    return raw_sentence, corrected_sentence

def translate_text(text, src_lang, dest_lang):
    # Dictionary of common English words and their Marathi translations
    normalized_text = text.lower().replace(" ", "")
    #normalized_text = ''.join(text.lower().split())
    common_translations = {
        'cap': 'टोपी',
        'hat': 'टोपी',
        'hello': 'नमस्कार',
        'hi': 'नमस्ते',
        'good': 'चांगले',
        'morning': 'सकाळ',
        'evening': 'संध्याकाळ',
        'thank': 'धन्यवाद',
        'thanks': 'धन्यवाद',
        'you': 'तुम्ही',
        'how': 'कसे',
        'are': 'आहात',
        'fine': 'छान',
        'water': 'पाणी',
        'food': 'अन्न',
        'yes': 'होय',
        'no': 'नाही',
        'please': 'कृपया',
        'sorry': 'माफ करा',
        'help': 'मदत',
        'book': 'पुस्तक',
        'pen': 'पेन',
        'car': 'कार',
        'house': 'घर',
        'school': 'शाळा'
    }

    if normalized_text in common_translations:
        return common_translations[normalized_text]
    
    # First check if the word is in our dictionary (case insensitive)
    if text.lower() in common_translations:
        return common_translations[text.lower()]
    
    # If the text contains multiple words, try to translate each word separately
    words = text.split()
    if len(words) > 1:
        translated_words = []
        for word in words:
            if word.lower() in common_translations:
                translated_words.append(common_translations[word.lower()])
            else:
                # Use Google Translate API for words not in our dictionary
                translator = Translator()
                try:
                    translated = translator.translate(word, src=src_lang, dest=dest_lang)
                    translated_words.append(translated.text)
                except Exception as e:
                    print(f"Translation error for '{word}': {e}")
                    translated_words.append(word)  # Keep original if translation fails
        return " ".join(translated_words)
    
    # For single words not in our dictionary or multi-word phrases, use Google Translate
    translator = Translator()
    try:
        translated = translator.translate(text, src=src_lang, dest=dest_lang)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def text_to_speech(text, language):
    pygame.init()
    pygame.mixer.init()
    
    tts = gTTS(text=text, lang=language)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    pygame.mixer.quit()
    pygame.quit()

def speech_to_text(language_code):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print(f"Say something in {language_code.split('-')[0]}...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    try:
        text = recognizer.recognize_google(audio, language=language_code)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError:
        print("Sorry, there was an error with the request.")
    return None
def find_word_folder_case_insensitive(dataset_folder, word):
    """Find a folder matching the word, ignoring case."""
    word = word.lower()
    # Check if the exact folder exists
    if os.path.exists(os.path.join(dataset_folder, word)):
        return os.path.join(dataset_folder, word)
    
    # Check case-insensitive
    for folder in os.listdir(dataset_folder):
        if folder.lower() == word:
            return os.path.join(dataset_folder, folder)
    
    print(f"No folder found for word: {word}")
    return None

def find_and_concatenate_videos(sentence, dataset_folder, output_folder):
    video_clips = []
    words = word_tokenize(sentence.lower())
    
    print(f"Words to find videos for: {words}")
    
    for word in words:
        word_folder = find_word_folder_case_insensitive(dataset_folder, word)
        if word_folder:
            print(f"Found folder for word '{word}': {word_folder}")
            video_files = [f for f in os.listdir(word_folder) if f.lower().endswith('.mp4')]
            if video_files:
                print(f"Found video files: {video_files}")
                video_path = os.path.join(word_folder, video_files[0])
                try:
                    video_clips.append(VideoFileClip(video_path))
                    print(f"Added video: {video_path}")
                except Exception as e:
                    print(f"Error loading video {video_path}: {e}")
            else:
                print(f"No .mp4 files found in folder: {word_folder}")
        else:
            print(f"No folder found for word: {word}")
    
    if video_clips:
        print(f"Total video clips found: {len(video_clips)}")
        try:
            final_clip = concatenate_videoclips(video_clips, method="compose")
            output_video_path = os.path.join(output_folder, "output_video.mp4")
            final_clip.write_videofile(output_video_path, codec="libx264", fps=24)
            print(f"Successfully created video at: {output_video_path}")
            return output_video_path
        except Exception as e:
            print(f"Error concatenating videos: {e}")
            return None
    else:
        print("No videos found for the sentence.")
        return None

def find_char_folder_case_insensitive(dataset_path, char):
    """Find a folder matching the character, ignoring case."""
    char = char.upper()
    # Check if the exact folder exists
    if os.path.exists(os.path.join(dataset_path, char)):
        return os.path.join(dataset_path, char)
    
    # Check case-insensitive
    for folder in os.listdir(dataset_path):
        if folder.upper() == char:
            return os.path.join(dataset_path, folder)
    
    return None

def get_image_path(char, dataset_path):
    if not char.isalnum() and not char.isspace():
        return None  # Skip punctuation and special characters
        
    if char.isspace():
        # Handle spaces - you might want to add a blank image for spaces
        return None
        
    char_folder = find_char_folder_case_insensitive(dataset_path, char)
    if char_folder and os.path.exists(char_folder) and os.listdir(char_folder):
        image_files = [f for f in os.listdir(char_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            image_path = os.path.join(char_folder, image_files[0])
            return image_path
    
    print(f"No image found for character: {char}")
    return None

def create_image_sequence(text, dataset_path):
    image_paths = []
    print(f"Creating image sequence for text: {text}")
    
    for char in text:
        if char.isspace():
            continue  # Skip spaces
            
        image_path = get_image_path(char, dataset_path)
        if image_path:
            print(f"Found image for character '{char}': {image_path}")
            image_paths.append(image_path)
        else:
            print(f"No image found for character: '{char}'")
    
    print(f"Total images found: {len(image_paths)}")
    return image_paths

def create_video_from_images(image_paths, output_video_path, frame_duration=1):
    if not image_paths:
        print("No images provided to create video")
        return False
        
    try:
        clips = [mpy.ImageClip(img).set_duration(frame_duration) for img in image_paths]
        video = mpy.concatenate_videoclips(clips, method="compose")
        video.write_videofile(output_video_path, fps=24)
        print(f"Successfully created video at: {output_video_path}")
        return True
    except Exception as e:
        print(f"Error creating video from images: {e}")
        return False

def video_from_image(text):
    dataset_path = "single_photo_dataset"
    sanitized_text = ''.join(e for e in text if e.isalnum() or e.isspace())
    output_video_path = f"output_video/{sanitized_text}.mp4"
    
    if not os.path.exists("output_video"):
        os.makedirs("output_video")
    
    image_paths = create_image_sequence(text, dataset_path)
    if image_paths:
        success = create_video_from_images(image_paths, output_video_path)
        if success:
            return output_video_path
    
    print("Failed to create video from images.")
    return None

def main():
    # Ensure output directory exists
    if not os.path.exists("output_video"):
        os.makedirs("output_video")
    
    # Hardcoded video file path for option 1
    video_file_path = r"C:\Users\Piyush\NLP_mini_Project\NLP_mini_Project\example_video.mp4"

    while True:
        print("\n1. Translate Hand Signs to Marathi Speech")
        print("2. Translate Text/Speech to Hand Signs")
        print("3. View Translation Visualizations")
        print("4. Create Translation Animation")
        print("5. Show System Evaluation Metrics")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            # Ask user to select input source
            print("\nSelect Input Source:")
            print("1. Camera Input")
            print("2. Video File Input")
            input_source = input("Enter your choice (1-2): ")
            
            # Record starting time
            start_time = time.time()
            detection_start = time.time()
            
            if input_source == '1':
                # Use camera input
                print("Using camera input for sign language detection...")
                raw_sentence, processed_sentence, letter_confidences = process_video_from_camera()
            else:
                # Use video file input
                print(f"Using video file: {video_file_path}")
                raw_sentence, processed_sentence = process_video_from_file(video_file_path)
                letter_confidences = {}  # Empty dict as process_video_from_file doesn't return confidences
            
            # Record detection time
            detection_time = time.time() - detection_start
            processing_start = time.time()
            
            if processed_sentence:
                print("Detected sentence:", processed_sentence)
                
                # Record translation start time
                translation_start = time.time()
                marathi_text = translate_text(processed_sentence, 'en', 'mr')
                translation_time = time.time() - translation_start
                
                print("Marathi translation:", marathi_text)
                
                # Record TTS start time
                tts_start = time.time()
                print("Converting Marathi text to speech...")
                text_to_speech(marathi_text, 'mr')
                tts_time = time.time() - tts_start
                
                # Calculate processing time
                processing_time = time.time() - processing_start
                
                # Collect timing metrics
                processing_times = {
                    "Detection": detection_time,
                    "Text Processing": processing_time - translation_time - tts_time,
                    "Translation": translation_time,
                    "TTS": tts_time
                }
                
                # Generate and show visualizations
                add_visualizations(raw_sentence, processed_sentence, marathi_text, letter_confidences, processing_times)
                
            else:
                print("No sign language detected. Please try again.")

        elif choice == '2':
            # Modified option 2 with input type and language selection
            print("\nSelect Input Type:")
            print("1. Text Input")
            print("2. Speech Input")
            input_choice = input("Enter your choice (1-2): ")
            
            print("\nSelect Language:")
            print("1. English")
            print("2. Marathi")
            lang_choice = input("Enter your choice (1-2): ")
            
            # Set language based on user selection
            input_language = 'en' if lang_choice == '1' else 'mr'
            language_code = 'en-US' if lang_choice == '1' else 'mr-IN'
            language_name = 'English' if lang_choice == '1' else 'Marathi'
            
            # Get input based on user selection
            if input_choice == '1':
                # Text input
                text_input = input(f"Enter your {language_name} text: ")
            else:
                # Speech input
                text_input = speech_to_text(language_code)
                if text_input:
                    print(f"Recognized {language_name} text:", text_input)
                else:
                    print(f"Failed to recognize {language_name} speech. Please try again.")
                    continue
            
            # Translate to English if input was in Marathi
            if input_language == 'mr':
                english_text = translate_text(text_input, 'mr', 'en')
                print("English translation:", english_text)
            else:
                english_text = text_input
            
            # Preserve important words and don't treat them as stopwords
            greeting_words = {"good", "morning", "hello", "hi", "afternoon", "evening", "night"}
            
            # Use NLTK for tokenization, normalization, and lemmatization
            tokens = word_tokenize(english_text.lower())
            lemmatizer = WordNetLemmatizer()
            pos_tags = pos_tag(tokens)
            normalized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
            
            # Remove stopwords but preserve greeting words
            stop_words = set(stopwords.words('english')) - greeting_words
            filtered_tokens = []
            
            for word in normalized_tokens:
                # Keep the word if it's not a stopword or is a greeting word
                if word not in stop_words or word.lower() in greeting_words:
                    filtered_tokens.append(word)
            
            normalized_text = ' '.join(filtered_tokens)
            
            print(f"Normalized tokens: {filtered_tokens}")
            print(f"Normalized text for video search: {normalized_text}")
            
            dataset_folder = 'youtube_dataset'
            output_folder = 'output_video'
            
            # Try to find and concatenate existing word videos
            output_video_path = find_and_concatenate_videos(normalized_text, dataset_folder, output_folder)
            if output_video_path:
                print("Hand sign video generated from existing clips. Displaying video...")
                display(Video(output_video_path, embed=True))
            else:
                print("No existing video clips found. Trying to generate video from raw English text...")
                # Try using the original text directly
                output_video_path = find_and_concatenate_videos(english_text, dataset_folder, output_folder)
                if output_video_path:
                    print("Hand sign video generated from raw text. Displaying video...")
                    display(Video(output_video_path, embed=True))
                else:
                    print("Generating video from images...")
                    output_video_path = video_from_image(normalized_text)
                    if output_video_path:
                        print("Hand sign video generated from images. Displaying video...")
                        display(Video(output_video_path, embed=True))
                    else:
                        print("Failed to generate hand sign video from normalized text.")
                        # As a last resort, try to create video from original English text
                        output_video_path = video_from_image(english_text)
                        if output_video_path:
                            print("Hand sign video generated from original English text. Displaying video...")
                            display(Video(output_video_path, embed=True))
                        else:
                            print("All methods failed to generate hand sign video.")

        elif choice == '3':
            # Modified option 3 to show fixed images instead of generated ones
            print("\nSelect visualization to view:")
            print("1.F1 curve")
            print("2.Results")
            print("3.R curve")
            print("4.Confusion Matrix")
            viz_choice = input("Enter your choice (1-4): ")
            
            # Define paths to the fixed image files
            viz_paths = {
                '1': r"C:\Users\Piyush\NLP_mini_Project\NLP_mini_Project\train5\F1_curve.png",
                '2': r"C:\Users\Piyush\NLP_mini_Project\NLP_mini_Project\train5\results.png",
                '3': r"C:\Users\Piyush\NLP_mini_Project\NLP_mini_Project\train5\R_curve.png",
                '4': r"C:\Users\Piyush\NLP_mini_Project\NLP_mini_Project\train5\confusion_matrix_normalized.png"
            }
            
            if viz_choice in viz_paths:
                image_path = viz_paths[viz_choice]
                
                if os.path.exists(image_path):
                    # Read the image
                    img = cv2.imread(image_path)
                    
                    # Get screen dimensions - using a reasonable default for a 1920x1080 screen
                    screen_width, screen_height = 1800, 950  # Slightly smaller than full screen to account for window borders
                    
                    # Get image dimensions
                    img_height, img_width = img.shape[:2]
                    
                    # Calculate scaling factor to fit on screen while maintaining aspect ratio
                    scale_factor = min(screen_width / img_width, screen_height / img_height)
                    
                    # Resize the image
                    new_width = int(img_width * scale_factor)
                    new_height = int(img_height * scale_factor)
                    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    # Create a named window that can be resized by the user if needed
                    title = "Sign Language Detection" if viz_choice == '1' else "Translation Workflow"
                    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                    
                    # Set the initial window size
                    cv2.resizeWindow(title, new_width, new_height)
                    
                    # Show the resized image
                    cv2.imshow(title, resized_img)
                    
                    # Wait for user to press any key
                    print("Press any key to close the visualization window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"Visualization file not found: {image_path}")
                    print("Please ensure the 'visualization_files' directory exists with the required images.")
            else:
                print("Invalid choice. Please try again.")
        
        elif choice == '4':
            # Create animation of the translation process
            text = input("Enter text to animate: ")
            create_translation_animation(text, output_path="translation_animation.mp4")
            print("Animation created at translation_animation.mp4")

        elif choice == '5':
            print("\nEvaluating system performance metrics...")
            # Normally we would collect real test data, but for demonstration we'll use simulated data
            test_data = collect_test_data()
            metrics = calculate_system_metrics(test_data)
            display_evaluation_metrics(metrics)

        elif choice == '6':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()