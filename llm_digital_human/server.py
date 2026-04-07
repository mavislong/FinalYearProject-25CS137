

import os
import json
import wave
import base64
import numpy as np
import torch
import librosa
import piper
import whisper
import tempfile
import requests
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
#import threading
import time

#local lip simple, no http
from lipsync_simple import (
    EmotionModel, LipSyncModel, normalize_values, create_keyframes,
    DEVICE, SAMPLE_RATE, Num_MFCC, HOP_LENGTH, ADD_ENERGY, FPS, ModelPath, Voice, Output_dir
)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})
port=5000
host = "0.0.0.0"
LOCAL_LLM = "mistral"

mcache = {
    "whisper": None,    
    "lipsync": None,
    "emotion": None,
}

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

print("Start Digital Human Server...")
print(f" Device: {DEVICE}")



def load_whisper_model():
    
    if mcache["whisper"] is None:
        print("Load Whisper model...")
        mcache["whisper"] = whisper.load_model("tiny")  
        print(" Whisper model loaded")
    return mcache["whisper"]

def stt_process(audio_data):

    try:
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmpPath = tmp.name
        
        print(f"Processing audio ({len(audio_data)} bytes)...")
        model = load_whisper_model()
        
        
        result = model.transcribe(tmpPath, language="en", fp16=False)
        text = result["text"].strip()
        
        print(f"  Transcribed: '{text}'")
        
        
        os.remove(tmpPath)
        
        return text
    
    except Exception as e:
        print(f"[ Error: {e}")
        return None

def llm_process(input_text):
    
    try:
        print(f"LLM: '{input_text}'")
        
        
        try:
            print(f"Local Ollama running,,,")
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": LOCAL_LLM,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Answer in English only, keep the length of answer in 30 words."
                        },
                        {
                            "role": "user",
                            "content": input_text
                        }
                    ],
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            response_text = response.json()["message"]["content"]
            print(f"Local Ollama response: '{response_text}'")
            return response_text
        
        except requests.exceptions.ConnectionError:
            print(f"Local Ollama not available, use OpenAI API as fallback...")
        

        try:
            import openai
            openai.api_key = "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"
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openai.api_key
            )
            
            completion = client.chat.completions.create(
                model="tngtech/deepseek-r1t2-chimera:free",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer in English only, keep the length of answer in 30 words."
                    },
                    {
                        "role": "user",
                        "content": input_text
                    }
                ],
                max_tokens=2048
            )
            
            response_text = completion.choices[0].message.content
            print(f"OpenAI response: '{response_text}'")
            return response_text
        
        except ImportError:
            print(f"Error:OpenAI not installed")
            return "Please ensure Ollama is running or install OpenAI SDK."
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Please try again."
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Please try again."



def tts_process(text, output_path=None):
    
    try:
        print(f"Generating audio for: '{text}'")
        
        ##new path
        if output_path is None:
            output_path = os.path.join(Output_dir, "response_audio.wav")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        voice = piper.PiperVoice.load(Voice)
        
        #new audio
        with wave.open(output_path, "wb") as wf:
            voice.synthesize_wav(text, wf)
        
        with wave.open(output_path, "rb") as wf:
            audio_data = wf.readframes(wf.getnframes())
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
        
        print(f"Generated audio ({duration:.2f}s, {len(audio_data)} bytes)")
        
        return audio_data, duration, output_path
    
    except Exception as e:
        print(f"Error: {e}")
        return None, 0, None

def load_lipsync_model():
    
    if mcache["lipsync"] is None:
        print("Loading model...")
        mcache["lipsync"] = LipSyncModel(ModelPath)
        print(" Lip Model loaded")
    return mcache["lipsync"]

def load_emotion_model():
    if mcache["emotion"] is None:
        print("Loading model...")
        mcache["emotion"] = EmotionModel()
        print("Emotion Model loaded")
    return mcache["emotion"]

def extract_audio_features(audio_path):
    print(f"Extract from {audio_path}...")
    
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=Num_MFCC, hop_length=HOP_LENGTH)
    
    if ADD_ENERGY:
        S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        energy= librosa.power_to_db(np.maximum(1e-5, S), ref=np.max).mean(axis=0, keepdims=True)
        energy= (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        mfcc= np.vstack([mfcc, energy])
    
    print(f"Shape: {mfcc.shape}")
    return mfcc.astype(np.float32)

def generate_animation_json(audio_path, response_text):
    print()
    print(f"Generating animation data...")
    
   
    lipsync_model = load_lipsync_model()
    emotion_model = load_emotion_model()
    
    features = extract_audio_features(audio_path)
   
    print(f"Running lip sync model...")
    predictions = lipsync_model.predict(features)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions min/max: {predictions.min():.4f} / {predictions.max():.4f}")
    
    
    blendshapes = normalize_values(predictions)
    print(f"Blendshapes shape: {blendshapes.shape}")
    #print(f"Blendshapes min/max: {blendshapes.min():.4f} / {blendshapes.max():.4f}")
    #(f"Sample blendshape values (frame 0): {blendshapes[:, 0]}")
    
    
    emotion_data = emotion_model.predict(response_text)
    if emotion_data:
        print(f"Emotion: {emotion_data.get('emotion', 'neutral')}")
    
    keyframes = create_keyframes(blendshapes, emotion_data=emotion_data)
    
    with wave.open(audio_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / rate
    
    expected_frames = int(duration * FPS)
    current_frames = len(keyframes) - 1
    
    if current_frames > expected_frames:
        keyframes = keyframes[:expected_frames + 1]
    elif current_frames < expected_frames:
        neutral_frame = {"frame": int(expected_frames)}
        neutral_frame.update({k: 0.0 for k in keyframes[0].keys() if k != "frame"})
        keyframes.append(neutral_frame)
    
    animation_data = {
        "frames": keyframes,
        "fps": int(FPS),
        "total_duration": float(len(keyframes) / FPS),
        "emotion_metadata": {
            "emotion": emotion_data.get("emotion", "neutral") if emotion_data else "neutral",
            "confidence": emotion_data.get("confidence", 0) if emotion_data else 0,
        }
    }
    

    print(f"Generated {len(keyframes)} frames at {int(FPS)} FPS (Audio: {duration:.2f}s)")
    #if keyframes:
    #frame_0 = keyframes[0]
    #print(f"[Animation] Frame 0 ARKit blendshapes:")
    #print(f"  jawOpen={frame_0.get('jawOpen', 0):.3f}")
    #print(f"  mouthSmile: Left={frame_0.get('mouthSmileLeft', 0):.3f}, Right={frame_0.get('mouthSmileRight', 0):.3f}")
    #print(f"  browInnerUp={frame_0.get('browInnerUp', 0):.3f}")
    #print(f"  Total keys in frame: {len(frame_0)}")
    
    return animation_data



@app.route('/process', methods=['POST', 'OPTIONS'])
def process_audio():
    
    
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        start_time = time.time()
        
        print("\n" + "="*70)
        print("REQUEST RECEIVED")
        print("="*70)
        print(f"Content-Type: {request.content_type}")
        print(f"Method: {request.method}")
        
        
        data = None
        try:
            data = request.get_json(force=True)  
        except Exception as e:
            print(f"JSON parse warning: {e}")
            
            try:
                raw_data = request.get_data(as_text=True)
                data = json.loads(raw_data)
            except:
                pass
        
        
        print(f"Parsed data type: {type(data)}")
        if data:
            print(f"JSON keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            print(f"Full JSON: {str(data)[:200]}...")
        else:
            print(f"Raw request body: {request.get_data(as_text=True)[:200]}...")
        
        if not data or "audio" not in data:
            error_msg = "No audio data provided"
            print(f"ERROR: {error_msg}")
            print(f"Available keys: {list(data.keys()) if data else 'None'}")
            return jsonify({"status": "error", "message": error_msg}), 400
        
        #Decoding
        audio_64 = data["audio"]
        audio_bytes = base64.b64decode(audio_64)
        print(f"Server got audio: {len(audio_bytes)} bytes")
        
        #try 
        user_text = stt_process(audio_bytes)
        if not user_text:
            return jsonify({"status": "error", "message": "STT failed"}), 500
        
        
        response_text = llm_process(user_text)
        if not response_text:
            return jsonify({"status": "error", "message": "LLM failed"}), 500
        
        response_audio_bytes, audio_duration, audio_path = tts_process(response_text)
        if response_audio_bytes is None:
            return jsonify({"status": "error", "message": "TTS failed"}), 500
        
        
        animation_data = generate_animation_json(audio_path, response_text)
        
        
        animation_file = os.path.join(Output_dir, "animation.json")
        with open(animation_file, "w") as f:
            json.dump(animation_data, f, indent=2)
        
        elapsed_time = time.time() - start_time
        
        
        response = {
            "status": "success",
            "user_text": user_text,
            "response_text": response_text,
            "emotion": animation_data["emotion_metadata"]["emotion"],
            "confidence": animation_data["emotion_metadata"]["confidence"],
            "processing_time": round(elapsed_time, 2),
            "audio_duration": round(audio_duration, 2),
            "audio_url": f"http://localhost:{port}/download/response_audio.wav",
            "animation_url": f"http://localhost:{port}/download/animation.json"
        }
        
        print("\n" + "="*70)
        print("response:")
        print("="*70)
        print(f"User: {user_text}")
        print(f"Response: {response_text}")
        print(f"Emotion: {response['emotion']}")
        print(f"Processing Time: {elapsed_time:.2f}s")
        print(f"Audio Duration: {audio_duration:.2f}s")
        print(f"Audio URL: {response['audio_url']}")
        print(f"Animation URL: {response['animation_url']}")
        print("="*70 + "\n")
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    
    #print(f"download: {filename}")
    
    if filename not in ["response_audio.wav", "animation.json"]:
        print(f"ERROR: Invalid file: {filename}")
        return jsonify({"status": "error", "message": "Invalid file"}), 400
    
    file_path = os.path.join(Output_dir, filename)
    print(f"Finding: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return jsonify({"status": "error", "message": f"File not found: {filename}"}), 404
    
    try:
        file_size = os.path.getsize(file_path)
        
        if filename.endswith('.wav'):
            print(f"Serving {filename} ({file_size} bytes) as audio/wav")
            return send_file(file_path, mimetype='audio/wav', as_attachment=False)
        else:  
            print(f"Serving {filename} ({file_size} bytes) as application/json")
            return send_file(file_path, mimetype='application/json', as_attachment=False)
    
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500





if __name__ == '__main__':
    
    os.makedirs(Output_dir, exist_ok=True)
    
    
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )
