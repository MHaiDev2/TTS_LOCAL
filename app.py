# import spaces  # Hugging Face Spaces specific, not needed locally
from kokoro import KModel, KPipeline  # Original import
# from kokoro_mock import KModel, KPipeline  # Better mock for local testing
import gradio as gr
import os
import random
import torch
import numpy as np
import datetime
import io
import wave
from huggingface_hub import HfApi
import tempfile
import time
import threading

IS_DUPLICATE = not os.getenv('SPACE_ID', '').startswith('hexgrad/')
CUDA_AVAILABLE = torch.cuda.is_available()
if not IS_DUPLICATE:
    import kokoro
    import misaki
    # import kokoro_mock as kokoro
    # import misaki_mock as misaki
    print('DEBUG', kokoro.__version__, CUDA_AVAILABLE, misaki.__version__)
    # print('DEBUG', 'mock-kokoro-0.9.4', CUDA_AVAILABLE, 'mock-misaki-0.9.4')

CHAR_LIMIT = None if IS_DUPLICATE else 5000
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Local storage configuration
RAWMATERIAL_FOLDER = 'rawmaterial'
OUTPUT_FOLDER = 'output'
DONE_FOLDER = os.path.join(RAWMATERIAL_FOLDER, 'done')

# Ensure folders exist
os.makedirs(RAWMATERIAL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DONE_FOLDER, exist_ok=True)

# Enhanced auto-processing state
auto_processing = False
processing_status = {
    "current": "", 
    "completed": [], 
    "remaining": [], 
    "errors": [],
    "start_time": None,
    "current_file_start": None,
    "progress_percent": 0,
    "total_files": 0,
    "processed_files": 0,
    "live_logs": []
}

def get_txt_files_from_rawmaterial():
    """Get all .txt files from the local rawmaterial folder"""
    try:
        files = os.listdir(RAWMATERIAL_FOLDER)
        txt_files = [f for f in files if f.endswith('.txt') and os.path.isfile(os.path.join(RAWMATERIAL_FOLDER, f))]
        print(f"Found {len(txt_files)} txt files in {RAWMATERIAL_FOLDER}: {txt_files}")
        return txt_files
    except Exception as e:
        print(f"Error listing files from rawmaterial folder: {e}")
        return []

def read_txt_content_from_rawmaterial(filename):
    """Read content from txt file in the local rawmaterial folder"""
    try:
        file_path = os.path.join(RAWMATERIAL_FOLDER, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        print(f"Successfully read content from {filename} ({len(content)} characters)")
        return content
        
    except Exception as e:
        print(f"Error reading {filename} from rawmaterial folder: {e}")
        return None

def move_txt_to_done(filename):
    """Move a txt file to the done folder after successful processing"""
    try:
        source_path = os.path.join(RAWMATERIAL_FOLDER, filename)
        done_path = os.path.join(DONE_FOLDER, filename)
        
        # Move file to done folder
        os.rename(source_path, done_path)
        
        print(f"Successfully moved {filename} to done folder")
        return True
        
    except Exception as e:
        print(f"Error moving {filename} to done folder: {e}")
        return False

def save_audio_to_output(audio_data, filename, sample_rate=24000):
    """Save audio data to local output folder"""
    try:
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        print(f"Saving audio to: {output_path}")
        print(f"Audio shape: {audio_data.shape if hasattr(audio_data, 'shape') else type(audio_data)}")
        
        # Convert numpy array to WAV
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"Audio saved successfully: {output_path}")
        return True
            
    except Exception as e:
        print(f"Error saving audio: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def process_text_to_audio_with_progress(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, max_chars=None, filename=""):
    """Process text to audio with progress tracking"""
    if max_chars:
        text = text[:max_chars]
    
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    audio_chunks = []
    
    # Split text into chunks for progress tracking
    text_chunks = list(pipeline(text, voice, speed))
    total_chunks = len(text_chunks)
    
    print(f"🎵 Starting audio generation for {filename}")
    print(f"📊 Total text chunks to process: {total_chunks}")
    
    for i, (_, ps, _) in enumerate(text_chunks):
        ref_s = pack[len(ps)-1]
        
        # Progress logging every ~10% or every chunk if less than 10 chunks
        progress_interval = max(1, total_chunks // 10)
        if i % progress_interval == 0 or i == total_chunks - 1:
            chunk_progress = int((i + 1) / total_chunks * 100)
            print(f"🔄 Processing chunk {i+1}/{total_chunks} ({chunk_progress}%) for {filename}")
        
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except Exception as e:
            print(f"GPU error, switching to CPU: {e}")
            audio = models[False](ps, ref_s, speed)
        
        audio_chunks.append(audio.numpy())
    
    if audio_chunks:
        print(f"✅ Audio generation completed for {filename}")
        return np.concatenate(audio_chunks)
    return None

def format_duration(seconds):
    """Format duration in a readable way"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {int(remaining_seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def log_message(message):
    """Add message to both console and live logs"""
    print(message)
    if "live_logs" in processing_status:
        processing_status["live_logs"].append(message)
        # Keep only last 50 messages to prevent memory issues
        if len(processing_status["live_logs"]) > 50:
            processing_status["live_logs"] = processing_status["live_logs"][-50:]

def auto_process_files():
    """Automatically process all txt files from dataset with enhanced progress tracking"""
    global auto_processing, processing_status
    
    auto_processing = True
    
    # Initialize processing status with timestamps
    start_time = datetime.datetime.now()
    processing_status = {
        "current": "", 
        "completed": [], 
        "remaining": [], 
        "errors": [],
        "start_time": start_time,
        "current_file_start": None,
        "progress_percent": 0,
        "total_files": 0,
        "processed_files": 0,
        "live_logs": []  # Store live log messages
    }
    
    try:
        # Get all txt files from rawmaterial folder
        txt_files = get_txt_files_from_rawmaterial()
        processing_status["remaining"] = txt_files.copy()
        processing_status["total_files"] = len(txt_files)
        
        log_message(f"🚀 AUTO-PROCESSING STARTED: {start_time.strftime('%Y-%m-%d at %H:%M:%S')}")
        log_message(f"📁 Found files in {RAWMATERIAL_FOLDER}: {len(txt_files)}")
        log_message("=" * 50)
        
        for file_index, txt_file in enumerate(txt_files):
            if not auto_processing:  # Stop if cancelled
                log_message("⏹️ Auto-Processing stopped")
                break
            
            # Update current file status
            processing_status["current"] = txt_file
            processing_status["current_file_start"] = datetime.datetime.now()
            current_progress = int((file_index / len(txt_files)) * 100)
            processing_status["progress_percent"] = current_progress
            
            log_message(f"\n📖 FILE {file_index + 1}/{len(txt_files)} ({current_progress}%)")
            log_message(f"📄 Processing: {txt_file}")
            log_message(f"⏰ Started: {processing_status['current_file_start'].strftime('%H:%M:%S')}")
            
            # Read content from rawmaterial folder
            content = read_txt_content_from_rawmaterial(txt_file)
            if not content:
                error_msg = f"Could not read {txt_file} from rawmaterial folder"
                processing_status["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                continue
            
            log_message(f"📝 Text length: {len(content)} characters")
            
            # Generate filename for audio
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"{base_name}.wav"
            
            try:
                # Process text to audio with progress tracking
                audio_data = process_text_to_audio_with_progress(
                    content, 
                    voice='af_nicole', 
                    speed=1.0,
                    filename=os.path.basename(txt_file)
                )
                
                if audio_data is not None:
                    print(f"💾 Saving audio as {audio_filename}...")
                    # Save to output folder
                    success = save_audio_to_output(audio_data, audio_filename)
                    
                    if success:
                        # Move txt file to done folder after successful processing
                        if move_txt_to_done(txt_file):
                            processing_status["completed"].append(txt_file)
                            processing_status["remaining"].remove(txt_file)
                            processing_status["processed_files"] += 1
                            
                            # Calculate file processing time
                            file_duration = (datetime.datetime.now() - processing_status["current_file_start"]).total_seconds()
                            
                            print(f"✅ SUCCESS: {txt_file}")
                            print(f"⏱️  File processing time: {format_duration(file_duration)}")
                            print(f"📁 File moved to done/")
                        else:
                            processing_status["errors"].append(f"Processed {txt_file} but could not move to done folder")
                    else:
                        processing_status["errors"].append(f"Failed to save audio for {txt_file}")
                        print(f"❌ Error saving {txt_file}")
                else:
                    processing_status["errors"].append(f"Failed to generate audio for {txt_file}")
                    print(f"❌ Error generating audio for {txt_file}")
                    
            except Exception as e:
                error_msg = f"Error processing {txt_file}: {str(e)}"
                processing_status["errors"].append(error_msg)
                print(f"❌ {error_msg}")
            
            # Small delay between files
            time.sleep(2)
    
    except Exception as e:
        print(f"❌ Auto-processing error: {e}")
    
    finally:
        # Final status update
        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        processing_status["current"] = ""
        processing_status["progress_percent"] = 100
        auto_processing = False
        
        log_message("\n" + "=" * 50)
        log_message(f"🏁 AUTO-PROCESSING FINISHED: {end_time.strftime('%Y-%m-%d at %H:%M:%S')}")
        log_message(f"⏱️  Total runtime: {format_duration(total_duration)}")
        log_message(f"✅ Successfully processed: {len(processing_status['completed'])}")
        log_message(f"❌ Errors: {len(processing_status['errors'])}")
        
        if processing_status['completed']:
            avg_time = total_duration / len(processing_status['completed'])
            log_message(f"📊 Average time per file: {format_duration(avg_time)}")

def start_auto_processing():
    """Start auto-processing in background thread"""
    if not auto_processing:
        thread = threading.Thread(target=auto_process_files)
        thread.daemon = True
        thread.start()
        return "🚀 Auto-Processing started..."
    return "⚠️ Already running..."

def stop_auto_processing():
    """Stop auto-processing"""
    global auto_processing
    auto_processing = False
    return "⏹️ Auto-Processing stopped"

def get_processing_status():
    """Get current processing status with enhanced information"""
    if auto_processing:
        current_time = datetime.datetime.now()
        
        # Calculate overall progress
        if processing_status["total_files"] > 0:
            overall_progress = int((processing_status["processed_files"] / processing_status["total_files"]) * 100)
        else:
            overall_progress = 0
        
        # Calculate elapsed time
        if processing_status["start_time"]:
            elapsed = (current_time - processing_status["start_time"]).total_seconds()
            elapsed_str = format_duration(elapsed)
        else:
            elapsed_str = "0s"
        
        # Calculate current file time
        current_file_time = ""
        if processing_status["current_file_start"]:
            file_elapsed = (current_time - processing_status["current_file_start"]).total_seconds()
            current_file_time = f" (running for {format_duration(file_elapsed)})"
        
        # Estimate remaining time
        remaining_estimate = ""
        if processing_status["processed_files"] > 0 and elapsed > 0:
            avg_time_per_file = elapsed / processing_status["processed_files"]
            remaining_files = processing_status["total_files"] - processing_status["processed_files"]
            estimated_remaining = avg_time_per_file * remaining_files
            remaining_estimate = f"\n⏳ Estimated remaining: {format_duration(estimated_remaining)}"
        
        status = f"""🔄 AUTO-PROCESSING RUNNING

📊 PROGRESS: {overall_progress}% ({processing_status['processed_files']}/{processing_status['total_files']})
⏰ Total runtime: {elapsed_str}{remaining_estimate}

📄 Current file: {os.path.basename(processing_status['current']) if processing_status['current'] else 'None'}{current_file_time}
📁 Source: {RAWMATERIAL_FOLDER}
💾 Output: {OUTPUT_FOLDER}

✅ Completed: {len(processing_status['completed'])}
📋 Remaining: {len(processing_status['remaining'])}
❌ Errors: {len(processing_status['errors'])}"""

        if processing_status['errors']:
            status += f"\n\n🚨 Recent errors:\n" + "\n".join([f"• {error}" for error in processing_status['errors'][-3:]])
    else:
        if processing_status.get("start_time"):
            # Show final summary
            end_time = datetime.datetime.now()
            if processing_status.get("processed_files", 0) > 0:
                total_time = format_duration((end_time - processing_status["start_time"]).total_seconds())
                status = f"""✅ AUTO-PROCESSING FINISHED

⏰ Total runtime: {total_time}
✅ Successful: {len(processing_status.get('completed', []))}
❌ Errors: {len(processing_status.get('errors', []))}
📁 Source: {RAWMATERIAL_FOLDER}

🎉 All available files have been processed!"""
            else:
                status = "⏹️ Auto-Processing stopped"
        else:
            status = f"⭕ Ready to start\n📁 Source: {RAWMATERIAL_FOLDER}\n💾 Output: {OUTPUT_FOLDER}"
    
    return status

def get_live_logs():
    """Get current live logs as formatted string"""
    if processing_status.get("live_logs"):
        # Show last 20 messages to keep it readable
        recent_logs = processing_status["live_logs"][-20:]
        return "\n".join(recent_logs)
    else:
        return "Waiting for auto-processing to start..."

# @spaces.GPU(duration=30)  # Removed for local usage
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        # Return the generated audio
        return (24000, audio.numpy()), ps
    return None, ''

def predict(text, voice='af_heart', speed=1):
    return None  # Completely disabled audio output

def tokenize_first(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        # Don't yield any audio to completely eliminate noise
        pass
        if first:
            first = False
            # Removed torch.zeros noise generator

def generate_and_save(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, save_to_storage_flag=False):
    """Generate audio with optional storage saving"""
    text = text if CHAR_LIMIT is None else text.strip()[:CHAR_LIMIT]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    audio_chunks = []
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audiobook_{voice}.wav"
    
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        
        audio_np = audio.numpy()
        audio_chunks.append(audio_np)
        yield 24000, audio_np
        
        if first:
            first = False
    
    # Save complete audio if requested
    if save_to_storage_flag and audio_chunks:
        try:
            full_audio = np.concatenate(audio_chunks)
            success = save_audio_to_output(full_audio, filename)
            if success:
                gr.Info(f"Audio saved as {filename}")
            else:
                gr.Warning("Could not save audio to output folder")
        except Exception as e:
            gr.Warning(f"Error saving audio: {e}")

with open('en.txt', 'r') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_gatsby():
    with open('gatsby5k.md', 'r') as r:
        return r.read().strip()

def get_frankenstein():
    with open('frankenstein5k.md', 'r') as r:
        return r.read().strip()

CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚹 Michael': 'am_michael',
'🇺🇸 🚹 Fenrir': 'am_fenrir',
'🇺🇸 🚹 Puck': 'am_puck',
'🇺🇸 🚹 Echo': 'am_echo',
'🇺🇸 🚹 Eric': 'am_eric',
'🇺🇸 🚹 Liam': 'am_liam',
'🇺🇸 🚹 Onyx': 'am_onyx',
'🇺🇸 🚹 Santa': 'am_santa',
'🇺🇸 🚹 Adam': 'am_adam',
'🇬🇧 🚺 Emma': 'bf_emma',
'🇬🇧 🚺 Isabella': 'bf_isabella',
'🇬🇧 🚺 Alice': 'bf_alice',
'🇬🇧 🚺 Lily': 'bf_lily',
'🇬🇧 🚹 George': 'bm_george',
'🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}
for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
💬 To adjust intonation, try punctuation `;:,.!?—…"()""` or stress `ˈ` and `ˌ`
⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`
⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 context length.')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button('Predict', variant='secondary', visible=False)

STREAM_NOTE = ['⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`.']
if CHAR_LIMIT is not None:
    STREAM_NOTE.append(f'✂️ Each stream is capped at {CHAR_LIMIT} characters.')
    STREAM_NOTE.append('🚀 Want more characters? You can [use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) or duplicate this space:')
STREAM_NOTE = '\n\n'.join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    save_checkbox = gr.Checkbox(label="Save to Output", value=False, info="Save complete audio to local output folder")
    with gr.Accordion('Note', open=True):
        gr.Markdown(STREAM_NOTE)
        storage_info = gr.Markdown("**Storage:** ✅ Local output folder configured")

# Enhanced Auto-Processing Tab
with gr.Blocks() as auto_tab:
    gr.Markdown("## 🚀 Auto-Process Text Files")
    gr.Markdown(f"Automatically converts all .txt files from **{RAWMATERIAL_FOLDER}/** to audiobooks and saves them to **{OUTPUT_FOLDER}/**.")
    
    with gr.Row():
        start_auto_btn = gr.Button('▶️ Start Auto-Processing', variant='primary')
        stop_auto_btn = gr.Button('⏹️ Stop', variant='stop')
    
    with gr.Row():
        with gr.Column():
            status_display = gr.Textbox(
                label="📊 Processing Status", 
                interactive=False, 
                lines=8,
                value=f"⭕ Ready to start\n📁 Source: {RAWMATERIAL_FOLDER}\n💾 Output: {OUTPUT_FOLDER}"
            )
        with gr.Column():
            live_logs = gr.Textbox(
                label="📝 Live Logs", 
                interactive=False, 
                lines=8,
                value="Waiting for auto-processing to start...",
                max_lines=50
            )
    refresh_status_btn = gr.Button('🔄 Refresh Status', variant='secondary')
    
    gr.Markdown("### 📋 Features:")
    gr.Markdown(f"""
    - 📁 **Local Processing**: Reads .txt files from {RAWMATERIAL_FOLDER}/
    - ⏰ **Timestamps**: Start and end times are logged  
    - 📊 **Progress Display**: Percentage display and processed files
    - ⚡ **Speed Measurement**: Time per file and estimated remaining time
    - 🔍 **Detailed Logs**: Progress every ~10% during audio generation
    - 💾 **Automatic Saving**: As [filename].wav to {OUTPUT_FOLDER}/
    - 📁 **Smart Moving**: .txt files moved to done/ after successful processing
    """)
    
    gr.Markdown("### 🛠️ Instructions:")
    gr.Markdown(f"""
    1. Place your text files in the **{RAWMATERIAL_FOLDER}/** folder
    2. Click '▶️ Start Auto-Processing'
    3. Monitor progress with '🔄 Refresh Status'
    4. Find generated audio files in the **{OUTPUT_FOLDER}/** folder
    5. Processed .txt files are automatically moved to **{DONE_FOLDER}/**
    """)

BANNER_TEXT = '''
[***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://huggingface.co/hexgrad/Kokoro-82M)
This local application supports manual streaming and automatic batch processing of text files with enhanced progress tracking.
'''
API_OPEN = os.getenv('SPACE_ID') != 'hexgrad/Kokoro-TTS'
API_NAME = None if API_OPEN else False

with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown(BANNER_TEXT, container=True)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info=f"Up to ~500 characters per Generate, or {'∞' if CHAR_LIMIT is None else CHAR_LIMIT} characters per Stream")
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice', info='Quality and availability vary by language')
                use_gpu = gr.Dropdown(
                    [('ZeroGPU 🚀', True), ('CPU 🐌', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info='GPU is usually faster, but has a usage quota',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            random_btn = gr.Button('🎲 Random Quote 💬', variant='secondary')
            with gr.Row():
                gatsby_btn = gr.Button('🥂 Gatsby 📕', variant='secondary')
                frankenstein_btn = gr.Button('💀 Frankenstein 📗', variant='secondary')
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab, auto_tab], ['Generate', 'Stream', 'Auto-Process'])
    
    # Event handlers
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text], api_name=API_NAME)
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text], api_name=API_NAME)
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text], api_name=API_NAME)
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps], api_name=API_NAME)
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps], api_name=API_NAME)
    
    # Stream events
    stream_event = stream_btn.click(
        fn=generate_and_save, 
        inputs=[text, voice, speed, use_gpu, save_checkbox], 
        outputs=[out_stream], 
        api_name=API_NAME
    )
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio], api_name=API_NAME)
    
    # Auto-processing events  
    start_auto_btn.click(fn=start_auto_processing, outputs=[status_display])
    stop_auto_btn.click(fn=stop_auto_processing, outputs=[status_display])
    
    # Refresh both status and logs
    def refresh_all():
        return get_processing_status(), get_live_logs()
    
    refresh_status_btn.click(fn=refresh_all, outputs=[status_display, live_logs])

if __name__ == '__main__':
    import sys
    # Fix encoding issues on Windows
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    app.queue(api_open=API_OPEN).launch(show_api=API_OPEN, ssr_mode=False)