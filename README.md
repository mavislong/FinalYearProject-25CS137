# AI-driven Digital Human Representative

This project aims to create a digital human model powered by artificial intelligence that can react in real time using natural language through an expressive and realistic audiovisual interface. In a general context, such as question-answer engagement, the digital avatar will offer dependable, human-like services through the use of Large Language Models (LLMs), facilitating efficient communication through lifelike conversations.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Software Requirements](#software-requirements)
- [Installation](#installation)
  - [Backend Setup (Python)](#backend-setup-python)
  - [Frontend Setup (Unity)](#frontend-setup-unity)
- [Project Execution](#project-execution)
  - [Terminal 1: Start Python Backend](#terminal-1-start-python-backend)
  - [Terminal 2: Run Unity Application](#terminal-2-run-unity-application)
  - [Interact with Avatar](#interact-with-avatar)
  - [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
- [License](#license)

---

## Project Overview

This project implements a complete digital human avatar system that:
- **Captures** user voice input
- **Transcribes** speech to text using Whisper 
- **Processes** text through a Large Language Model (LLM)
- **Synthesizes** response speech using Piper TTS
- **Generates** facial blendshape animations synchronized to synthesized speech
- **Renders** real-time lip-synced animation in a 3D avatar
- **Detects** emotional content and applies emotion-based facial expressions

**Key Features:**
- 100 FPS lip-sync animation precision
- 52 ARKit compatible blendshapes for facial animation
- Emotion detection using BERT-based classifier
- Local LLM support with OpenRouter fallback
- CORS-enabled REST API for cross-origin communication

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNITY FRONTEND (C#)                      │
│  [Audio Capture] → [Base64 Encode] → [REST POST Request]   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST /process
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FLASK BACKEND (Python) :5000                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. TRANSCRIPTION: Whisper (Speech-to-Text STT)        │ │
│  │    Audio (WAV) → OpenAI Whisper → English Text        │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 2. LLM PROCESSING: Large Language Model               │ │
│  │    Text → Ollama/OpenRouter → Response (around 30 words) │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 3. TEXT-TO-SPEECH: Piper TTS                          │ │
│  │    Response Text → ONNX Synthesis → WAV Audio         │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 4. EMOTION DETECTION: BERT Classifier                 │ │
│  │    Response Text → GoEmotions → Emotion Tag           │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 5. LIP-SYNC GENERATION: 1D CNN Model                  │ │
│  │    Audio → MFCC Features → PyTorch 1D CNN →           │ │
│  │    8 Blendshape Predictions (100 FPS)                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 6. ANIMATION KEYFRAME GENERATION                      │ │
│  │    52 ARKit Blendshapes + Emotion Expression +        │ │
│  │    Neutral Frame + JSON Serialization                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│         JSON Response + Audio File + Animation Data         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP Response
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              UNITY FRONTEND (C#) - PLAYBACK                 │
│  [Parse JSON] → [Play Audio] → [Apply Blendshapes] →       │
│  [100 FPS Animation] → [Return to Neutral]                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
project_root/
│
├─── llm_digital_human/                    # Python Backend
│    ├── server.py                         # Flask REST API endpoint
│    ├── lipsync_simple.py                 # models inferencing & keyframe generation
│    ├── label_mapping.py                  # Emotion label mappings
│    ├── train_1dcnn2.py                   # 1D CNN model training script
│    ├── UnityClient.cs                    # Unity C# script
│    ├── best_1dcnn.pt                     # Pre-trained 1D CNN model
│    ├── en_GB-southern_english_female-low.onnx  # Piper TTS model
│    ├── en_GB-southern_english_female-low.onnx.json # TTS config
│    ├── goemotions_single_plain_best/     # Emotion model & tokenizer
│    │   ├── config.json
│    │   ├── model.safetensors
│    │   ├── special_tokens_map.json
│    │   ├── tokenizer_config.json
│    │   ├── tokenizer.json
│    │   └── vocab.txt
│    ├── 3d/                               # 3D model assets
│    ├── MODNet/                           # MODNet background removal
│    ├── lipsync_output/                   # Generated animation output
│    └── __pycache__/                      # Python cache
│
└─── unityavatar/                          # Unity Frontend (v2022+)
     ├── Assets/
     │   ├── Scripts/
     │   │   └── UnityClient.cs            # Main interaction script
     │   ├── Prefabs/                      # Avatar 3D models
     │   ├── Scenes/
     │   │   └── Main.unity                # Main scene
     │   └── Materials/                    # Shaders & materials
     ├── ProjectSettings/
     ├── Packages/
     └── [other Unity project files]
```

---

## Prerequisites

### System Requirements
- **OS:** Windows 10/11
- **GPU:** Optional for better inferencing capability
- **RAM:** Minimum 8GB 
- **Disk Space:** 5GB 

### Software Requirements
- **Python:** 3.9 or higher 
- **CUDA:** Optional 
- **Unity:** 2022 LTS or newer
- **C#:** Compatible with .NET Framework 4.7.1+

---

## Installation

### Backend Setup (Python)

#### Step 1: Create Python Virtual Environment
```bash
cd llm_digital_human
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
```

#### Step 2: Install Required Libraries

```
Flask==3.1.2
Flask-CORS==4.0.0
numpy==1.24.3
torch==1.13.1+cpu
librosa==0.11.0
piper-tts
openai-whisper==20250625
transformers==4.35.0
requests==2.32.3
tqdm==4.67.1
scipy==1.13.1
soundfile==0.13.1
```





#### Step 3: Download Pre-trained Models

The following models should be placed in the `llm_digital_human/` directory:

1. **1D CNN Lip-Sync Model** (already included)
   - File: `best_1dcnn.pt`
   - Size: ~2MB
   - Converts MFCC audio features to 8 blendshape predictions

2. **Piper TTS Model** 
   - File: `en_GB-southern_english_female-low.onnx`
   - Size: ~60MB
   - Synthesizes text to speech

3. **Whisper STT Model** 
   - Model: `tiny` variant
   - Size: ~140MB
   - Transcribes speech to text

4. **Emotion Detection Model** (already included)
   - Directory: `goemotions_single_plain_best/`
   - Size: ~440MB
   - BERT-based emotion classifier

#### Step 4: Install Ollama (Optional - for Local LLM)

For local LLM inference without internet dependency:

1. Download from https://ollama.ai
2. Run: `ollama pull mistral`
3. Start Ollama service: `ollama serve`

The server will automatically fall back to OpenRouter API if Ollama is unavailable.

### Frontend Setup (Unity)

#### Step 1: Open Unity Project
1. Launch Unity Hub
2. Click "Open Project"
3. Select the `unityavatar/` folder
4. Wait for project to load (may take 5-10 minutes for first import)

#### Step 2: Configure Avatar Scene
1. Open `Assets/Scenes/Main.unity`
2. Select the Avatar GameObject

#### Step 3: Attach UnityClient Script
1. Copy [UnityClient.cs](llm_digital_human/UnityClient.cs) to `Assets/Scripts/`
2. Attach script to Avatar GameObject:
   - Drag script onto Avatar in hierarchy, OR
   - Select Avatar → Drag UnityClient.cs to Inspector's script field
3. Configure Inspector fields:
   - Assign Headmesh to the Facerenderer

#### Step 4: Build & Run
```
File → Build Settings
Platform: Select "PC, Mac & Linux Standalone" (or Android/iOS)
Build & Run
```

---



## Project Execution


### Terminal 1: Start Python Backend
```bash
cd llm_digital_human
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac
python server.py
```

Expected output:
```
Start Digital Human Server...
Device: cuda (or cpu)
Load Whisper model...
Whisper model loaded
Emotion model loaded
 * Running on http://0.0.0.0:5000
```

### Terminal 2: Run Unity Application
1. Open Unity project
2. Click Play button in Unity Editor, OR
3. Run the built executable: `unityavatar/Build/UnityAvatar.exe`

### Interact with Avatar
1. Click **"Start Recording"** button in Unity
2. Speak into your microphone (10 seconds)
3. Speak something like: *"What is Computer Science?"*
4. Wait for processing
5. Avatar speaks response with synchronized lip animation

### Full Pipeline Walkthrough

```
User Speech Input
    ↓
[UnityClient.cs] Captures 10-second audio → WAV format → Base64 encode
    ↓
POST request to http://localhost:5000/process
    ↓
[server.py] Receives audio
    ↓
[STT] Whisper transcribes audio → "Hello how are you"
    ↓
[LLM] Ollama processes text → "I'm doing well, thank you!"
    ↓
[EMOTION] BERT classifier → Emotion: "neutral" (confidence: 0.95)
    ↓
[TTS] Piper synthesizes → "I'm doing well, thank you!" → WAV audio file
    ↓
[MFCC] Extract 14-channel features (13 MFCC + energy) at 100 FPS
    ↓
[1D CNN] Neural network predicts 8 blendshapes per frame → 100 frames
    ↓
[KEYFRAMES] Generate complete 52-blendshape animation at 100 FPS
    ↓
[JSON] Serialize animation + metadata
    ↓
HTTP Response (JSON + audio file + animation data)
    ↓
[UnityClient.cs] Parses JSON animation
    ↓
[Animation Loop] Applies 52 blendshapes per frame, synced to audio playback
    ↓
[Neutral Reset] After animation ends, returns avatar to neutral pose
    ↓
Ready for next input
```



## License

This project uses the following open-source models and libraries:

- Whisper: OpenAI, MIT License.
- Ollama: MIT License.
- Piper TTS
- Transformers: Hugging Face Transformers.
- PyTorch: BSD-style license.

---

**Version:** 1.0  
**Last Updated:** April 2026  

