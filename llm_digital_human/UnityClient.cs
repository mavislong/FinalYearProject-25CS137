using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class DigitalHumanClient : MonoBehaviour
{
    [SerializeField] private string SERVER_URL ="http://localhost:5000/process";
    [SerializeField] private AudioSource AudioSource_play;
    [SerializeField] private SkinnedMeshRenderer facerenderer;
    
    private bool recording_active =false;
    private const int  sample_rate = 16000;
    private const float RecordDuration = 10f;
    private string Status_Text  ="Click button or Press SPACE"; 
    
    void Start()
    {
        if (AudioSource_play ==null)
        {
            AudioSource_play = GetComponent<AudioSource>();
            if (AudioSource_play ==null)
            {
                AudioSource_play = gameObject.AddComponent<AudioSource>();
                Debug.Log("AudioSource auto-created");
            }
        }
        AudioSource_play.playOnAwake = false;
        AudioSource_play.spatialBlend = 0f;
        if (facerenderer == null)
        {
            SkinnedMeshRenderer[] meshes = GetComponentsInChildren<SkinnedMeshRenderer>();
            if (meshes.Length > 0)
                facerenderer = meshes[0];
        }
        
        Debug.Log("Ready! Click button or press SPACE to record");
    }
    
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space) && !recording_active)
        {
            StartRecording();
        }
    }
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 150));
        
        GUI.skin.box.fontSize = 16;
        GUI.Box(new Rect(0, 0, 280, 140), "Audio Recording");
        GUI.skin.label.fontSize = 12;
        GUI.Label(new Rect(10, 30, 260, 40), Status_Text);
        GUI.skin.button.fontSize = 14;
        if (!recording_active)
        {
            if (GUI.Button(new Rect(10, 75, 260, 50), "▶ START RECORDING (10s)"))
            {
                StartRecording();
            }
        }
        else
        {
            GUI.skin.button.normal.textColor = Color.red;
            if (GUI.Button(new Rect(10, 75, 260, 50), "⏹ STOP RECORDING"))
            {
                StopRecording();
            }
            GUI.skin.button.normal.textColor = Color.white;
        }
        
        GUILayout.EndArea();
    }
    
    private void StartRecording()
    {
        if (recording_active)
            return;
        
        recording_active = true;
        UpdateStatusText("Recording... (10 sec)");
        Debug.Log("Recording started for 10 seconds...");
        
        if (Microphone.devices.Length > 0)
        {
            AudioSource_play.clip = Microphone.Start(Microphone.devices[0], false, (int)RecordDuration, sample_rate);
            StartCoroutine(RecordingCoroutine());
        }
        else
        {
            Debug.LogError("No microphone detected");
            UpdateStatusText("ERROR: No microphone");
            recording_active = false;
        }
    }
    
    private void StopRecording()
    {
        if (!recording_active) return;
        
        recording_active = false;
        Microphone.End(Microphone.devices[0]);
        
        AudioClip clip_audio = AudioSource_play.clip;
        if (clip_audio == null || clip_audio.length == 0)
        {
            Debug.LogError(" No audio recorded!");
            UpdateStatusText("ERROR: No audio recorded");
            return;
        }
        
        Debug.Log($"Recording stopped. Duration: {clip_audio.length}s, Samples: {clip_audio.samples}");
        
        UpdateStatusText("Converting to WAV");
        byte[] WAV_data = WavUtility.FromAudioClip(clip_audio);
        Debug.Log($"WAV data size: {WAV_data.Length} bytes");
        
        if (WAV_data.Length == 0)
        {
            Debug.LogError("WAV conversion failed!");
            UpdateStatusText("ERROR: WAV conversion failed");
            return;
        }
        
        StartCoroutine(ProcessAudioCoroutine(WAV_data));
    }
    
    private IEnumerator RecordingCoroutine()
    {
        yield return new WaitForSeconds(RecordDuration);
        
        StopRecording();
    }
    
    private IEnumerator ProcessAudioCoroutine(byte[] audio_data)
    {
        string Base64_AUDIO = System.Convert.ToBase64String(audio_data);
        Debug.Log($"Base64 audio length: {Base64_AUDIO.Length} chars");
        string request_body = "{\"audio\":\"" + Base64_AUDIO + "\"}";
        Debug.Log($"Request JSON size: {request_body.Length} chars");
        Debug.Log($"JSON preview: {request_body.Substring(0, Mathf.Min(100, request_body.Length))}...");

        byte[] BodyRaw = System.Text.Encoding.UTF8.GetBytes(request_body);
        Debug.Log($"JSON payload size: {BodyRaw.Length} bytes");
        
        using (UnityWebRequest web_request = new UnityWebRequest(SERVER_URL, "POST"))
        {
            web_request.uploadHandler = new UploadHandlerRaw(BodyRaw);
            web_request.downloadHandler = new DownloadHandlerBuffer();
            web_request.SetRequestHeader("Content-Type", "application/json");
            
            Debug.Log("Sending audio to server...");
            Debug.Log($"Sending to: {SERVER_URL}");
            
            yield return web_request.SendWebRequest();
            
            HandleProcessResponse(web_request);
        }
    }

    private void HandleProcessResponse(UnityWebRequest response_req)
    {
        if (response_req.result == UnityWebRequest.Result.Success)
        {
            string response_text = response_req.downloadHandler.text;
            Debug.Log($"Server response: {response_text.Substring(0, Mathf.Min(200, response_text.Length))}");
            
            ProcessResponse ProcessResp = null;
            try
            {
                ProcessResp = JsonUtility.FromJson<ProcessResponse>(response_text);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to parse response JSON: {e.Message}");
                UpdateStatusText($"Error: Response parse failed");
                return;
            }
            
            if (ProcessResp.status == "success")
            {
                Debug.Log($"User said: {ProcessResp.user_text}");
                Debug.Log($"Response: {ProcessResp.response_text}");
                Debug.Log($"Emotion: {ProcessResp.emotion}");
                Debug.Log($"Audio URL: {ProcessResp.audio_url}");
                Debug.Log($"Animation URL: {ProcessResp.animation_url}");
                StoreLastResponse(ProcessResp);
                
                UpdateStatusText($"Response: {ProcessResp.response_text}");
                
                StartCoroutine(PlayResponseAudioCoroutine(ProcessResp.audio_url));
            }
            else
            {
                Debug.LogError($"Server error: {ProcessResp.message}");
                UpdateStatusText($"Error: {ProcessResp.message}");
            }
        }
        else
        {
            Debug.LogError($"Request error: {response_req.error}");
            Debug.LogError($"Response code: {response_req.responseCode}");
            if (!string.IsNullOrEmpty(response_req.downloadHandler?.text))
                Debug.LogError($"Response: {response_req.downloadHandler.text}");
            
            UpdateStatusText($"Network Error: {response_req.error}");
        }
    }

    private IEnumerator PlayResponseAudioCoroutine(string Audio_URL)
    {
        Debug.Log($"Downloading audio from: {Audio_URL}");
        
        using (UnityWebRequest audio_request = UnityWebRequestMultimedia.GetAudioClip(Audio_URL, AudioType.WAV))
        {
            yield return audio_request.SendWebRequest();
            
            HandleAudioResponse(audio_request);
        }
    }

    private void HandleAudioResponse(UnityWebRequest req_audio)
    {
        if (req_audio.result == UnityWebRequest.Result.Success)
        {
            AudioClip clip_loaded = DownloadHandlerAudioClip.GetContent(req_audio);
            
            if (clip_loaded != null)
            {
                Debug.Log($"Audio loaded: {clip_loaded.length}s, {clip_loaded.frequency}Hz, {clip_loaded.channels}ch");
                
                AudioSource_play.clip = clip_loaded;
                AudioSource_play.Play();
                
                UpdateStatusText($"Playing audio ({clip_loaded.length:F1}s)...");
                ProcessResponse data_response = GetLastResponse();
                if (data_response != null && !string.IsNullOrEmpty(data_response.animation_url))
                {
                    Debug.Log($"Found animation URL, download");
                    StartCoroutine(AnimateFaceCoroutine(data_response.animation_url, clip_loaded.length));
                }
                else
                {
                    if (data_response == null)
                        Debug.LogWarning("lastResponse is null!");
                    else
                        Debug.LogWarning($"animation_url is empty: '{data_response.animation_url}'");
                }
            }
            else
            {
                Debug.LogError("Failed to load audio clip");
                UpdateStatusText("Error: Audio decode failed");
            }
        }
        else
        {
            Debug.LogError($"Download error: {req_audio.error}");
            UpdateStatusText($"Download error: {req_audio.error}");
        }
    }

    private ProcessResponse last_response;
    
    private void StoreLastResponse(ProcessResponse response)
    {
        last_response = response;
    }
    
    private ProcessResponse GetLastResponse()
    {
        return last_response;
    }

    private IEnumerator AnimateFaceCoroutine(string animURL, float audio_duration)
    {
        Debug.Log($"Starting animation coroutine");
        Debug.Log($"Downloading from: {animURL}");
        Debug.Log($"Audio duration: {audio_duration}s");
        
        using (UnityWebRequest web_req = UnityWebRequest.Get(animURL))
        {
            yield return web_req.SendWebRequest();
            
            if (web_req.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"Download successful!");
                HandleAnimationResponse(web_req, audio_duration);
            }
            else
            {
                Debug.LogError($"Download FAILED: {web_req.error}");
                Debug.LogError($"Response code: {web_req.responseCode}");
                UpdateStatusText($"Animation download failed: {web_req.error}");
            }
        }
    }

    private void HandleAnimationResponse(UnityWebRequest req, float audio_dur)
    {
        if (req.result == UnityWebRequest.Result.Success)
        {
            string JSON_anim = req.downloadHandler.text;
            Debug.Log($"JSON received: {JSON_anim.Length} bytes");
            
            if (string.IsNullOrEmpty(JSON_anim))
            {
                Debug.LogError("Empty JSON response!");
                UpdateStatusText("Error: Empty animation data");
                return;
            }
            
            Debug.Log($"JSON preview: {JSON_anim.Substring(0, Mathf.Min(300, JSON_anim.Length))}...");
            
            AnimationData AnimData = null;
            try
            {
                AnimData = JsonUtility.FromJson<AnimationData>(JSON_anim);
                Debug.Log("JSON parsed successfully");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to parse animation JSON: {e.Message}");
                Debug.LogError($"Animation JSON (full): {JSON_anim}");
                UpdateStatusText("Error: Animation parse failed");
                return;
            }
            
            if (AnimData != null && AnimData.frames != null && AnimData.frames.Length > 0)
            {
                Debug.Log($"Animation frames: {AnimData.frames.Length}");
                Debug.Log($"FPS: {AnimData.fps}");
                Debug.Log($"Total Duration: {AnimData.total_duration}s");
                
                Frame frame_0 = AnimData.frames[0];
                Debug.Log($"Frame 0 blendshapes:");
                Debug.Log($"  jawOpen={frame_0.jawOpen:F3}, mouthSmileLeft={frame_0.mouthSmileLeft:F3}, mouthSmileRight={frame_0.mouthSmileRight:F3}");
                Debug.Log($"  browInnerUp={frame_0.browInnerUp:F3}, cheekPuff={frame_0.cheekPuff:F3}");
                
                Debug.Log("Starting ApplyAnimationCoroutine...");
                StartCoroutine(ApplyAnimationCoroutine(AnimData, audio_dur));
            }
            else
            {
                Debug.LogError($"Invalid animation data - frames null or empty!");
                UpdateStatusText("Error: Invalid animation data");
            }
        }
        else
        {
            Debug.LogError($"Download error: {req.error}");
            Debug.LogError($"Response code: {req.responseCode}");
            UpdateStatusText($"Download error: {req.error}");
        }
    }

    private IEnumerator ApplyAnimationCoroutine(AnimationData Anim_Data, float audio_Duration)
    {
        Debug.Log($"Starting animation with {Anim_Data.frames.Length} frames at {Anim_Data.fps} FPS");
        
        float frame_time = 1.0f / Anim_Data.fps;
        int FRAME_INDEX = 0;
        float elapsed_time = 0f;
        if (facerenderer != null && facerenderer.sharedMesh != null && FRAME_INDEX == 0)
        {
            int blend_count = facerenderer.sharedMesh.blendShapeCount;
            Debug.Log($"Available blendshapes on mesh: {blend_count}");
            for (int i = 0; i < Mathf.Min(10, blend_count); i++)
            {
                string blend_name = facerenderer.sharedMesh.GetBlendShapeName(i);
                Debug.Log($"  [{i}] {blend_name}");
            }
        }
        
        while (FRAME_INDEX < Anim_Data.frames.Length && elapsed_time < Anim_Data.total_duration)
        {
            Frame current_Frame = Anim_Data.frames[FRAME_INDEX];
            
            if (FRAME_INDEX == 0 || FRAME_INDEX % 10 == 0)
            {
                Debug.Log($"Frame {FRAME_INDEX}: jawOpen={current_Frame.jawOpen}, mouthSmileLeft={current_Frame.mouthSmileLeft}, browInnerUp={current_Frame.browInnerUp}");
            }
            
            ApplyBlendshapeFrame(current_Frame);
            FRAME_INDEX++;
            elapsed_time += frame_time;
            
            yield return new WaitForSeconds(frame_time);
        }
        
        ResetBlendshapesToNeutral();
        Debug.Log("Animation complete - Reset to neutral");
        UpdateStatusText("Complete! Press to Record (10 sec)");
    }

    private void ApplyBlendshapeFrame(Frame current_frame)
    {
        if (facerenderer == null || facerenderer.sharedMesh == null)
        {
            Debug.LogWarning("faceMesh is null!");
            return;
        }
        
        System.Type frame_class = typeof(Frame);
        System.Reflection.FieldInfo[] field_list = frame_class.GetFields();
        
        foreach (System.Reflection.FieldInfo Blend_Field in field_list)
        {
            if (Blend_Field.FieldType == typeof(float))
            {
                string blend_shape = Blend_Field.Name;
                float blend_value = (float)Blend_Field.GetValue(current_frame);
                
                ApplyBlendshape(blend_shape, blend_value);
            }
        }
    }
    
    private void ApplyBlendshape(string ShapeName, float Weight_Val)
    {
        if (facerenderer == null || facerenderer.sharedMesh == null)
            return;
        
        int mesh_index = facerenderer.sharedMesh.GetBlendShapeIndex(ShapeName);
        if (mesh_index >= 0)
        {
            float clamped_weight = Mathf.Clamp01(Weight_Val);
            facerenderer.SetBlendShapeWeight(mesh_index, clamped_weight);
        }
    }
    
    private void ResetBlendshapesToNeutral()
    {
        if (facerenderer == null || facerenderer.sharedMesh == null)
            return;
        
        int total_shape_count = facerenderer.sharedMesh.blendShapeCount;
        for (int shape_idx = 0; shape_idx < total_shape_count; shape_idx++)
        {
            facerenderer.SetBlendShapeWeight(shape_idx, 0f);
        }
    }
    
    private void UpdateStatusText(string message_txt)
    {
        Status_Text = message_txt;
        Debug.Log(message_txt);
    }
}


[System.Serializable]
public class ProcessResponse
{
    public string status;
    public string message;
    public string user_text;
    public string response_text;
    public string emotion;
    public float confidence;
    public float processing_time;
    public float audio_duration;
    public string audio_url;
    public string animation_url;
}

[System.Serializable]
public class AnimationData
{
    public Frame[] frames;
    public int fps;
    public float total_duration;
}

[System.Serializable]
public class Frame
{
    public int frame;
    public float browDownLeft;
    public float browDownRight;
    public float browInnerUp;
    public float browOuterUpLeft;
    public float browOuterUpRight;
    public float cheekPuff;
    public float cheekSquintLeft;
    public float cheekSquintRight;
    public float eyeBlinkLeft;
    public float eyeBlinkRight;
    public float eyeLookDownLeft;
    public float eyeLookDownRight;
    public float eyeLookInLeft;
    public float eyeLookInRight;
    public float eyeLookOutLeft;
    public float eyeLookOutRight;
    public float eyeLookUpLeft;
    public float eyeLookUpRight;
    public float eyeSquintLeft;
    public float eyeSquintRight;
    public float eyeWideLeft;
    public float eyeWideRight;
    public float jawForward;
    public float jawLeft;
    public float jawRight;
    public float jawOpen;
    public float mouthClose;
    public float mouthDimpleLeft;
    public float mouthDimpleRight;
    public float mouthFrownLeft;
    public float mouthFrownRight;
    public float mouthFunnel;
    public float mouthLeft;
    public float mouthLowerDownLeft;
    public float mouthLowerDownRight;
    public float mouthPressLeft;
    public float mouthPressRight;
    public float mouthPucker;
    public float mouthRight;
    public float mouthRollLower;
    public float mouthRollUpper;
    public float mouthShrugLower;
    public float mouthShrugUpper;
    public float mouthSmileLeft;
    public float mouthSmileRight;
    public float mouthStretchLeft;
    public float mouthStretchRight;
    public float mouthUpperUpLeft;
    public float mouthUpperUpRight;
    public float noseSneerLeft;
    public float noseSneerRight;
    public float tongueOut;
}

public static class WavUtility
{
    public static byte[] FromAudioClip(AudioClip clip)
    {
        int sampleCount = clip.samples;
        int channelCount = clip.channels;
        int sampleRate = clip.frequency;
        
        float[] samples = new float[sampleCount * channelCount];
        clip.GetData(samples, 0);
        
        return EncodeToWav(samples, sampleRate, channelCount);
    }
    
    private static byte[] EncodeToWav(float[] samples, int sampleRate, int channels)
    {
        int fileLength = 36 + samples.Length * 2;
        System.IO.MemoryStream memoryStream = new System.IO.MemoryStream(44 + samples.Length * 2);
        using (System.IO.BinaryWriter writer = new System.IO.BinaryWriter(memoryStream))
        {
            
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(fileLength);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));
        
            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16); 
            writer.Write((ushort)1); 
            writer.Write((ushort)channels);
            writer.Write(sampleRate);
            writer.Write(sampleRate * channels * 2); 
            writer.Write((ushort)(channels * 2)); 
            writer.Write((ushort)16); 
            
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(samples.Length * 2);
            
            foreach (float sample in samples)
            {
                short sampleData = (short)(sample * 32767f);
                writer.Write(sampleData);
            }
        }
        
        return memoryStream.ToArray();
    }
    
    public static AudioClip ToAudioClip(byte[] wavBytes)
    {
        System.IO.MemoryStream memoryStream = new System.IO.MemoryStream(wavBytes);
        using (System.IO.BinaryReader reader = new System.IO.BinaryReader(memoryStream))
        {

            string riffHeader = new string(reader.ReadChars(4));
            if (riffHeader != "RIFF")
            {
                Debug.LogError("Invalid RIFF header");
                return null;
            }
            
            int riffLength = reader.ReadInt32();
            string waveHeader = new string(reader.ReadChars(4));
            if (waveHeader != "WAVE")
            {
                Debug.LogError("Invalid WAVE header");
                return null;
            }
            
            int sampleRate = 0;
            int channels = 0;
            int bitsPerSample = 0;
            byte[] audioData = null;
            
            while (memoryStream.Position < memoryStream.Length - 8)
            {
                string chunkId = new string(reader.ReadChars(4));
                int chunkSize = reader.ReadInt32();
                long chunkStart = memoryStream.Position;
                
                if (chunkId == "fmt ")
                {
                    ushort audioFormat = reader.ReadUInt16();
                    if (audioFormat != 1)
                    {
                        Debug.LogError($"Unsupported audio format: {audioFormat}");
                        return null;
                    }
                    
                    channels = reader.ReadUInt16();
                    sampleRate = reader.ReadInt32();
                    int byteRate = reader.ReadInt32();
                    ushort blockAlign = reader.ReadUInt16();
                    bitsPerSample = reader.ReadUInt16();
                    
                    Debug.Log($"fmt chunk: {sampleRate}Hz, {channels}ch, {bitsPerSample}bit");
                }
                else if (chunkId == "data")
                {
                    audioData = reader.ReadBytes(chunkSize);
                    Debug.Log($"data chunk: {audioData.Length} bytes");
                }
                else
                {
                    Debug.Log($"Skipping chunk: {chunkId}");
                }
                
                memoryStream.Seek(chunkStart + chunkSize, System.IO.SeekOrigin.Begin);
            }
            
            if (sampleRate == 0 || channels == 0 || audioData == null)
            {
                Debug.LogError("Missing required chunks in WAV file");
                return null;
            }
            int sampleCount = audioData.Length / (bitsPerSample / 8);
            float[] samples = new float[sampleCount];
            
            if (bitsPerSample == 16)
            {
                for (int i = 0; i < sampleCount; i++)
                {
                    short sample = System.BitConverter.ToInt16(audioData, i * 2);
                    samples[i] = sample / 32768f;
                }
            }
            else
            {
                Debug.LogError($"Unsupported bits per sample: {bitsPerSample}");
                return null;
            }
            
            
            AudioClip clip = AudioClip.Create("WavAudio", sampleCount / channels, channels, sampleRate, false);
            clip.SetData(samples, 0);
            
            Debug.Log($"WAV decoded: {sampleRate}Hz, {channels}ch, {sampleCount / channels} samples");
            return clip;
        }
    }
}
