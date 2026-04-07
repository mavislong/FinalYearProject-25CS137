## no flask, CLI input text to lip
import os
import wave
import numpy as np
import torch
import librosa
import piper
import json
from tqdm import tqdm
import sys
import time
from transformers import BertTokenizerFast, BertForSequenceClassification
from label_mapping import GoEmotion_Labels, Label_Group


timing_data = {
    "program_start": 0,       
    "stages": {}
}


Script_dir =os.path.dirname(os.path.abspath(__file__))

ModelPath =os.path.join(Script_dir, "best_1dcnn.pt")
Voice = os.path.join(Script_dir, "en_GB-southern_english_female-low.onnx")
Output_dir = os.path.join(Script_dir, "lipsync_output")

#const
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
Num_MFCC =13
HOP_LENGTH =160
ADD_ENERGY = True  #flag


FPS = SAMPLE_RATE /HOP_LENGTH  

print(f"Target FPS: {FPS}")

os.makedirs(Output_dir, exist_ok=True)

print(f"Device: {DEVICE}")
#print(f"Output dir: {Output_dir}")


class EmotionModel:
    def __init__(self):
        
        print("Get emotion model...")
        try:
            self.model_dir = "goemotions_single_plain_best"
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.available = True
            print("Emotion model loaded")
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            self.available = False
    
    def predict(self, text):
        
        if not self.available:
            return {}
        
        enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64,
            )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs=torch.softmax(logits, dim=-1)[0].cpu().numpy()
            
            
        predID = int(probs.argmax())
        confidence = float(probs[predID])
        
        raw_emo = GoEmotion_Labels[predID]
        print(raw_emo, confidence)
        emotion_group = Label_Group[raw_emo]
            
        #print(emotion_group)
        return {
            "emotion": emotion_group,
            "raw_emotion": raw_emo,
            "confidence": confidence,
        }
            
        

class LipSyncModel:
    def __init__(self, modelpath):
        print(f"Loading from {modelpath}...")
        
        checkpoint = torch.load(modelpath, map_location=DEVICE, weights_only=False)
        self.in_ch = checkpoint.get("in_ch", None)
        self.out_ch = checkpoint.get("out_ch", 6)
        if self.in_ch is None:
            for key in checkpoint["model_state"]:
                if "weight" in key and "0" in key: 
                    self.in_ch = checkpoint["model_state"][key].shape[1]
                    break
        ##mfcc 13+1
        if self.in_ch is None:
            self.in_ch = 14  
        
        
        from train_1dcnn2 import CNN1D
        self.model = CNN1D(self.in_ch, self.out_ch).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
        print(f"Loaded 1dCNN: {self.in_ch} inputs --> {self.out_ch} outputs")
    
    def predict(self, audio_features):
        
        x = torch.from_numpy(audio_features).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            y = self.model(x)
        
        return y.squeeze(0).cpu().numpy()


def generate_audio(text, output_path):
    print()
    print(f"Generatig audio...")
    print(f"TTS Text: '{text}'")
    
    voice = piper.PiperVoice.load(Voice)
    
    with wave.open(output_path, "wb") as wf:
        voice.synthesize_wav(text, wf)
    

    with wave.open(output_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / rate
    
    #rint(f"Saved to {output_path}")
    #print(duration)
    
    return duration

def extract_features(audio_path):
    print()
    print(f"Extracting from {audio_path}...")
    
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=Num_MFCC, hop_length=HOP_LENGTH)
    
    if ADD_ENERGY:
        S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        energy = librosa.power_to_db(np.maximum(1e-5, S), ref=np.max).mean(axis=0, keepdims=True)
       
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        mfcc = np.vstack([mfcc, energy])
    
    #print(f"Shape: {mfcc.shape} ({mfcc.shape[1]} frames)")
    
    return mfcc.astype(np.float32)

ARKIT_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]


BLENDSHAPE_MAPPING = {
    "jawOpen": 0,             # Channel 0
    "mouthSmileLeft": 1,      # 1
    "mouthSmileRight": 1,     # 1 
    "mouthFrownLeft": 2,      # 2
    "mouthFrownRight": 2,     # 2 
    "mouthPucker": 3,         # 3
    "mouthUpperUpLeft": 4,    # 4
    "mouthUpperUpRight": 4,   # 4
    "mouthLowerDownLeft": 5,  # 5
    "mouthLowerDownRight": 5, # 5
}

'''# Old mapping for reference
BLENDSHAPES = {
    0: "jawOpen",
    1: "mouthSmile",
    2: "mouthFrown",
    3: "mouthPucker",
    4: "mouthUpperUp",
    5: "mouthLowerDown",
}
'''


EMOTION_BLENDSHAPES = {
    "neutral": {
        "browInnerUp": 0.0, "eyeWideLeft": 0.0, "eyeWideRight": 0.0,
        "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0,
    },
    "positive": {  # joy, amusement, excitement, gratitude, love, optimism, relief, pride, admiration, approval
        "browInnerUp": 0.6, "browOuterUpLeft": 0.4, "browOuterUpRight": 0.4,
        "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5, "eyeWideLeft": 0.3, "eyeWideRight": 0.3,
        "mouthSmileLeft": 0.7, "mouthSmileRight": 0.7, "cheekPuff": 0.4,
    },
    "warm": {  # caring
        "browInnerUp": 0.3, "eyeWideLeft": 0.2, "eyeWideRight": 0.2,
        "mouthSmileLeft": 0.4, "mouthSmileRight": 0.4,
    },
    "curious": {  # curiosity, realization
        "browOuterUpLeft": 0.4, "browOuterUpRight": 0.4, "eyeWideLeft": 0.3, "eyeWideRight": 0.3,
    },
    "desire": {  # desire
        "eyeWideLeft": 0.5, "eyeWideRight": 0.5, "mouthSmileLeft": 0.3, "mouthSmileRight": 0.3,
    },
    "angry": {  # anger, annoyance, disapproval
        "browDownLeft": 0.7, "browDownRight": 0.7, "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
        "mouthFrownLeft": 0.5, "mouthFrownRight": 0.5, "noseSneerLeft": 0.3, "noseSneerRight": 0.3,
    },
    "disgust": {  # disgust
        "noseSneerLeft": 0.7, "noseSneerRight": 0.7, "browDownLeft": 0.5, "browDownRight": 0.5,
        "mouthUpperUpLeft": 0.6, "mouthUpperUpRight": 0.6, "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
    },
    "sad": {  # sadness, disappointment, embarrassment, grief, remorse
        "browInnerUp": 0.6, "browDownLeft": 0.5, "browDownRight": 0.5,
        "mouthFrownLeft": 0.6, "mouthFrownRight": 0.6,
        "eyeLookDownLeft": 0.3, "eyeLookDownRight": 0.3,
    },
    "anxious": {  # fear, nervousness
        "browInnerUp": 0.7, "browOuterUpLeft": 0.6, "browOuterUpRight": 0.6,
        "eyeWideLeft": 0.8, "eyeWideRight": 0.8, "jawOpen": 0.3, "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
    },
    "surprised": {  # surprise, confusion
        "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8, "browInnerUp": 0.5,
        "eyeWideLeft": 0.7, "eyeWideRight": 0.7, "jawOpen": 0.4,
    },
}

def normalize_values(predictions):
    out_ch, T = predictions.shape
    normalized = np.zeros_like(predictions)
    
    for ch in range(out_ch):
        seq = predictions[ch, :]
        min_v, max_v = seq.min(), seq.max()
        
        if max_v > min_v:
            normalized[ch, :] = (seq - min_v) / (max_v - min_v)
        else:
            normalized[ch, :] = 0.5
    
    return np.clip(normalized, 0, 1)

def create_keyframes(blendshapes, frame_duration_ms=None, emotion_data=None):
    
    out_ch, T = blendshapes.shape
    keyframes = []
    
    if frame_duration_ms is None:
        frame_duration_ms = 1000.0/FPS
    
    emotion_label = emotion_data.get("emotion", "neutral") if emotion_data else "neutral"
    emotion_weights = EMOTION_BLENDSHAPES.get(emotion_label, EMOTION_BLENDSHAPES["neutral"])

    all_arkit_blendshapes = {
        "browDownLeft": 0.0, "browDownRight": 0.0, "browInnerUp": 0.0, "browOuterUpLeft": 0.0, "browOuterUpRight": 0.0,
        "cheekPuff": 0.0, "cheekSquintLeft": 0.0, "cheekSquintRight": 0.0,
        "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0, "eyeLookDownLeft": 0.0, "eyeLookDownRight": 0.0,
        "eyeLookInLeft": 0.0, "eyeLookInRight": 0.0, "eyeLookOutLeft": 0.0, "eyeLookOutRight": 0.0,
        "eyeLookUpLeft": 0.0, "eyeLookUpRight": 0.0, "eyeSquintLeft": 0.0, "eyeSquintRight": 0.0,
        "eyeWideLeft": 0.0, "eyeWideRight": 0.0,
        "jawForward": 0.0, "jawLeft": 0.0, "jawRight": 0.0, "jawOpen": 0.0,
        "mouthClose": 0.0, "mouthDimpleLeft": 0.0, "mouthDimpleRight": 0.0, "mouthFrownLeft": 0.0,
        "mouthFrownRight": 0.0, "mouthFunnel": 0.0, "mouthLeft": 0.0, "mouthLowerDownLeft": 0.0,
        "mouthLowerDownRight": 0.0, "mouthPressLeft": 0.0, "mouthPressRight": 0.0, "mouthPucker": 0.0,
        "mouthRight": 0.0, "mouthRollLower": 0.0, "mouthRollUpper": 0.0, "mouthShrugLower": 0.0,
        "mouthShrugUpper": 0.0, "mouthSmileLeft": 0.0, "mouthSmileRight": 0.0, "mouthStretchLeft": 0.0,
        "mouthStretchRight": 0.0, "mouthUpperUpLeft": 0.0, "mouthUpperUpRight": 0.0,
        "noseSneerLeft": 0.0, "noseSneerRight": 0.0,
        "tongueOut": 0.0
    }
    
    for frame_i in range(T):
        frame_data = {"frame": int(frame_i)}
        frame_data.update(all_arkit_blendshapes)
        frame_data["jawOpen"] = float(blendshapes[0, frame_i])
        
        
        smile_val = float(blendshapes[1, frame_i])
        frame_data["mouthSmileLeft"] = smile_val
        frame_data["mouthSmileRight"] = smile_val
        frame_data["cheekPuff"] = smile_val * 0.5
        
       
        frown_val = float(blendshapes[2, frame_i])
        frame_data["mouthFrownLeft"] = frown_val
        frame_data["mouthFrownRight"] = frown_val
        
       
        frame_data["mouthPucker"] = float(blendshapes[3, frame_i])
        
        upper_val = float(blendshapes[4, frame_i])
        frame_data["mouthUpperUpLeft"] = upper_val
        frame_data["mouthUpperUpRight"] = upper_val
        
        
        lower_val = float(blendshapes[5, frame_i])
        frame_data["mouthLowerDownLeft"] = lower_val
        frame_data["mouthLowerDownRight"] = lower_val
        
        
        frame_data["mouthLeft"] = smile_val * 0.3
        frame_data["mouthRight"] = smile_val * 0.3
        
        ##lip+emo
        for blend_name, blend_value in emotion_weights.items():
            if blend_name in frame_data:
                
                current_val = frame_data[blend_name]
                frame_data[blend_name] = float(np.clip(current_val + blend_value * 0.5, 0, 1))
            else:
                
                frame_data[blend_name] = float(blend_value)
        
        keyframes.append(frame_data)
    
    neutral_frame = {"frame": int(T)}
    neutral_frame.update(all_arkit_blendshapes)
    keyframes.append(neutral_frame)
    
    return keyframes

def save_animation_data(keyframes, audio_path, model_info, emotion_data=None):
    
    
    output_file = os.path.join(Output_dir, "lipsync_animation.json")
    
    ##check
    duration = len(keyframes) / FPS if keyframes else 0
    
    data = {
        "frames": keyframes,
        "audio_path": os.path.abspath(audio_path),
        "fps": int(FPS),
        "total_frames": len(keyframes),
        "duration": float(duration)
    }
    
    
    if emotion_data:
        data["emotion_metadata"] = {
            "emotion": emotion_data.get("emotion", "neutral"),
            "confidence": emotion_data.get("confidence", 0),
        }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print()
    print(f"Animation data: {output_file}")
    print(f"Total no. of frames: {len(keyframes)}")
    print(f"FPS: {int(FPS)}")
    print(f"Dur: {duration:.2f}s")
    if emotion_data:
        blendshape_count = len(keyframes[0]) - 1 if keyframes else 0
        print(f"lip sync + emotion applied: {blendshape_count} blendshapes per frame")
    
    return output_file


def lipsync_from_text(text):
    
    global timing_data
    
    
    print()
    print(f"Predicting emotions from text...")
    emotion_start = time.perf_counter()
    
    emotion_model = EmotionModel()
    emotion_data = None
    
    if emotion_model.available:
        emotion_data = emotion_model.predict(text)
        if emotion_data:
            print(f"{emotion_data['emotion']} (confidence: {emotion_data['confidence']:.2f})")
    
    emotion_elapsed = time.perf_counter() - emotion_start
    timing_data["stages"]["Emotion Detection"] = emotion_elapsed
    print(f"Time: {emotion_elapsed*1000:.2f} ms")
    print()
    print(f"TTS  is generating audio from text...")
    tts_start = time.perf_counter()
    
    audio_path = os.path.join(Output_dir, "audio.wav")
    duration = generate_audio(text, audio_path)
    
    tts_elapsed = time.perf_counter() - tts_start
    timing_data["stages"]["TTS (Audio Generation)"] = tts_elapsed
    print(f"TTS Time: {tts_elapsed*1000:.2f} ms")
    
    print()
    print("Extracting MFCC features...")
    features_start = time.perf_counter()
    
    features = extract_features(audio_path)
    
    features_elapsed = time.perf_counter() - features_start
    timing_data["stages"]["Feature Extraction"] = features_elapsed
    print(f"Time: {features_elapsed*1000:.2f} ms")
    
    print()
    print("Running lip sync model...")
    inference_start = time.perf_counter()
    
    model = LipSyncModel(ModelPath)
    predictions = model.predict(features)
    print(f"Output shape: {predictions.shape}")
    
    inference_elapsed  = time.perf_counter() - inference_start
    timing_data["stages"]["Model Inference"] = inference_elapsed
    print(f"Inf Time: {inference_elapsed*1000:.2f} ms")
    
    print()
    print("Converting to blendshapes...")
    normalize_start = time.perf_counter() # <1
    
    blendshapes = normalize_values(predictions)
    print(f" [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
    
    normalize_elapsed = time.perf_counter() - normalize_start
    timing_data["stages"]["Normalization"] = normalize_elapsed
    print(f"Time: {normalize_elapsed*1000:.2f} ms")
    
    print()
    print(f"Creating animation frames...")
    keyframes_start = time.perf_counter()
    
    keyframes = create_keyframes(blendshapes, emotion_data=emotion_data)
    print (f"{len(keyframes)} frames at {int(FPS)} FPS")
    if emotion_data:
        emotion_label = emotion_data.get('emotion', 'neutral')
        num_blendshapes = len(keyframes[0]) - 1  # -1 for frame number
        print(f" Emotion ({emotion_label}) blendshapes integrated: {num_blendshapes} total")
    
    keyframes_elapsed = time.perf_counter() - keyframes_start
    timing_data["stages"]["Keyframe Creation"] = keyframes_elapsed
    #print(f" Time: {keyframes_elapsed*1000:.2f} ms")
    
    print()
    print(f"[Save] Saving animation data...")
    save_start = time.perf_counter()
    
    model_info = {
        "name": "best_1dcnn.pt",
        "inputChannels": model.in_ch,
        "outputChannels": model.out_ch,
        "device": DEVICE,
    }
    animation_file = save_animation_data(keyframes, audio_path, model_info, emotion_data)
    
    save_elapsed = time.perf_counter() - save_start
    timing_data["stages"]["Save to JSON"] = save_elapsed
    #print(f"Time: {save_elapsed*1000:.2f} ms")
    
    #print("="*60)
    print("Complete simple lip sync")
    print("="*60)
    print(f"\n Files generated:")
    print(f" {audio_path}")
    print(f" {animation_file}")
    if emotion_data:
        print()
        print(f"Emotion Integration:")
        print(f"Emotion: {emotion_data.get('emotion', 'unknown')}")
        print(f"Confidence: {emotion_data.get('confidence', 0):.2f}")
        print(f"Blendshapes: LIP SYNC + EMOTION combined in each frame")
    
        #direct copy to Unity Script path, with wav + json
    
    return animation_file, audio_path

if __name__ == "__main__":
    import sys
    
    timing_data["program_start"] = time.perf_counter()
    
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("\n" + "="*60)
        print("LIP SYNC with run time input" )
        print("="*60)
        print("Enter the text you want the avatar to say:")
        print("(Press Enter twice when done)")
        
        lines = []
        while True:
            line = input()
            if line == "":
                if lines:
                    break
                else:
                    print("Text Input:")
                    continue
            lines.append(line)
        
        text = " ".join(lines)
    
    print(f"Text: '{text}'\n")
    '''
    try:
        animation_file, audio_path = lipsync_from_text(text)
        #t=start-end
        #print(t)
        total_time = time.perf_counter() - timing_data["program_start"]
        
        
        print("\n" + "="* 70)
        print("TIME SPENT DISTRIBUTION")
        print("="* 70)
        
        print()
        print("Stage Breakdown:")
        print("-" *  70)
        for stage_name, elapsed in timing_data["stages"].items():
            percentage = (elapsed / total_time) * 100
            print(f"{stage_name:30} {elapsed*1000:10.2f} ms  ({percentage:5.1f}%)")
        
        print("-"  * 70)
        print(f"{'TOTAL EXECUTION TIME':30} {total_time*1000:10.2f} ms (100.0%)")
        print("="  * 70)
        print()
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

    '''