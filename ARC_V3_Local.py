import os
import sys
import time
import json
import re
import asyncio
import sqlite3
import audioop
import pyaudio
import winsound
import ollama
import pyttsx3
import psutil
import platform
from datetime import datetime as dt
from vosk import Model, KaldiRecognizer
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "vosk-model-small-en-us-0.15")
DB_PATH = os.path.join(SCRIPT_DIR, "arc_memory.db")
SAMPLE_RATE = 16000
SILENCE_TIMEOUT = 1.8  # seconds

# UI Colors
class C:
    USER = Fore.CYAN
    ARC = Fore.LIGHTGREEN_EX
    SYSTEM = Fore.LIGHTYELLOW_EX
    TEXT = Fore.LIGHTWHITE_EX
    DEV = Fore.LIGHTBLACK_EX
    MUTED = Style.DIM
    RESET = Style.RESET_ALL

# ==========================================
# SQLITE FTS5 MEMORY DATABASE
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create virtual FTS5 table for session memory
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
            timestamp,
            topic,
            summary
        )
    """)
    conn.commit()
    conn.close()

def add_memory(topic, summary):
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO memories (timestamp, topic, summary) VALUES (?, ?, ?)",
        (timestamp, topic, summary)
    )
    conn.commit()
    conn.close()
    print(f"{C.DEV}[MEMORY STORED] Topic: {topic} | Summary: {summary}{C.RESET}")

def recall_memory(query, limit=2):
    if not query:
        return ""
    # Clean query of non-alphanumeric chars for FTS5 safety
    clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
    if not clean_query:
        return ""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Search using BM25 ranking
        cursor.execute(
            "SELECT timestamp, topic, summary FROM memories WHERE memories MATCH ? ORDER BY bm25(memories) LIMIT ?",
            (clean_query, limit)
        )
        results = cursor.fetchall()
    except sqlite3.OperationalError:
        # Fallback if match query syntax fails
        cursor.execute(
            "SELECT timestamp, topic, summary FROM memories LIMIT ?",
            (limit,)
        )
        results = cursor.fetchall()
    finally:
        conn.close()
        
    if not results:
        return ""
    
    mem_blocks = []
    for row in results:
        mem_blocks.append(f"[{row[0]}] Topic: {row[1]} | Summary: {row[2]}")
    return "\n".join(mem_blocks)

# ==========================================
# SYSTEM MONITORING & UTILITIES
# ==========================================
def get_system_report():
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(os.path.splitdrive(SCRIPT_DIR)[0] + '\\').percent
    battery = psutil.sensors_battery()
    
    battery_str = "N/A"
    if battery:
        plugged = "Plugged In" if battery.power_plugged else "Discharging"
        battery_str = f"{battery.percent}% ({plugged})"
        
    return {
        "cpu": f"{cpu}%",
        "ram": f"{ram}%",
        "disk": f"{disk}%",
        "battery": battery_str
    }

# ==========================================
# ADAPTIVE VAD (VOICE ACTIVITY DETECTION)
# ==========================================
class AdaptiveVAD:
    def __init__(self, initial_threshold=180, alpha=0.98):
        self.noise_floor = initial_threshold
        self.alpha = alpha
        self.speech_active = False

    def process_frame(self, frame_data):
        rms = audioop.rms(frame_data, 2)
        # Adapt noise floor on lower volumes
        if rms < self.noise_floor * 1.5:
            self.noise_floor = self.alpha * self.noise_floor + (1 - self.alpha) * rms
            
        threshold = max(self.noise_floor * 2.0, 150)
        is_speech = rms > threshold
        return is_speech, rms, threshold

# ==========================================
# ASYNC SPEECH PLAYER (TTS WORKER)
# ==========================================
def speak_sync(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"TTS Error: {e}")

# ==========================================
# PERSONALITY & AGENT SYSTEM PROMPTS
# ==========================================
DETECTED_OS = platform.system()

SYSTEM_PROMPT = f"""You are ARC: a local-first, voice-operated operating system assistant running on {DETECTED_OS}. Precision of J.A.R.V.I.S., calm tone of F.R.I.D.A.Y., dry wit of T.A.R.S.
Optimize all responses for speech speed. Keep responses under 250 characters.
Do NOT start responses with 'As ARC', 'ARC:', or repeat your own role/system description. Speak directly and naturally to the user.

You must generate command actions compatible ONLY with the host operating system: {DETECTED_OS}.
For example, since the host is {DETECTED_OS}, do not generate shell commands for other operating systems (e.g., use 'dir' instead of 'ls' if on Windows).
Always check the 'Current workspace directory' and 'Available files/folders' provided in context to avoid using dummy or placeholder paths. Use the actual paths and files from the context.

If the user requests system monitoring, command execution, launching applications/browsers, or file/git operations:
Respond with a single line starting with 'ACTION:' followed by a JSON object containing the command to execute, a brief description, and an explanation.
Example: ACTION: {{"command": "git status", "description": "Check git repository status", "explanation": "I will inspect the active git workspace status."}}

Otherwise, respond directly in a conversational, technical, and concise manner.
"""

# ==========================================
# MAIN SYSTEM CLASS
# ==========================================
class ARCSystem:
    def __init__(self):
        self.audio_queue = None
        self.speak_queue = None
        self.loop = None
        self.working_context = []
        self.max_window = 10
        self.vad = AdaptiveVAD()
        
        # Initialize Vosk
        if not os.path.exists(MODEL_PATH):
            print(f"{C.SYSTEM}Vosk model not found at: {MODEL_PATH}{C.RESET}")
            print(f"{C.SYSTEM}Please download a model from https://alphacephei.com/vosk/models and extract it there.{C.RESET}")
            sys.exit(1)
            
        self.model = Model(MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.rec.SetWords(True)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.loop and self.audio_queue:
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    async def tts_worker(self):
        while True:
            text = await self.speak_queue.get()
            # Run blocking pyttsx3 code in executor
            await asyncio.to_thread(speak_sync, text)
            self.speak_queue.task_done()

    def start_recording(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=2000,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    async def add_to_context(self, role, content):
        self.working_context.append({"role": role, "content": content})
        if len(self.working_context) >= self.max_window:
            # Summarize and save to FTS5 memory
            await self.condense_and_store_memory()

    async def condense_and_store_memory(self):
        convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.working_context)
        summary_prompt = [
            {"role": "system", "content": "Summarize this conversation. Return ONLY: Topic: <topic> | Summary: <max 20 words>"},
            {"role": "user", "content": convo_text}
        ]
        try:
            # Quick summary query using qwen or llama
            response = await asyncio.to_thread(
                ollama.chat, model="llama3.1:latest", messages=summary_prompt
            )
            summary_content = response["message"]["content"].strip()
            
            # Parse topic & summary
            topic = "General"
            summary_text = summary_content
            if "|" in summary_content:
                parts = summary_content.split("|")
                for p in parts:
                    if p.strip().lower().startswith("topic:"):
                        topic = p.split(":")[1].strip()
                    elif p.strip().lower().startswith("summary:"):
                        summary_text = p.split(":")[1].strip()
                        
            add_memory(topic, summary_text)
        except Exception as e:
            print(f"Memory storage error: {e}")
        finally:
            self.working_context = []

    async def execute_action(self, action_json):
        try:
            action = json.loads(action_json)
            command = action.get("command")
            description = action.get("description")
            explanation = action.get("explanation")
            
            # Command Safety Guardrails
            cmd_lower = command.lower()
            blocklist = [
                "format", "shutdown", "net user", "net localgroup",
                "reg delete", "del /s", "rmdir /s", "rm -rf", "mkfs"
            ]
            is_unsafe = False
            for pattern in blocklist:
                if pattern in cmd_lower:
                    is_unsafe = True
                    break
                    
            if is_unsafe:
                warn_msg = "Action blocked. The command violates safety guardrails."
                print(f"\n{C.SYSTEM}[BLOCKED] Unsafe command rejected:{C.RESET} {command}")
                await self.speak_queue.put(warn_msg)
                return "Execution blocked by safety guardrails (unsafe system utility or recursive delete keyword)."
            
            # Notify user and request confirmation
            speak_prompt = f"I need to run the following action: {description}. Please say yes to confirm or no to cancel."
            print(f"\n{C.SYSTEM}ARC REQUESTS ACTION:{C.RESET} {description}")
            print(f"{C.DEV}Command line: {command}{C.RESET}")
            
            await self.speak_queue.put(speak_prompt)
            # Wait for TTS to finish speaking prompt
            await self.speak_queue.join()
            
            # Start listening for confirmation
            confirmed = await self.get_voice_confirmation()
            if confirmed:
                print(f"{C.SYSTEM}Executing: {command}...{C.RESET}")
                winsound.Beep(440, 200)
                
                # Run the command asynchronously
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                output = stdout.decode().strip()
                error = stderr.decode().strip()
                
                res_content = f"Output:\n{output}" if output else ""
                if error:
                    res_content += f"\nError:\n{error}"
                if not res_content:
                    res_content = "Command completed with no output."
                    
                print(f"{C.DEV}Command Result:{C.RESET}\n{res_content}")
                return res_content
            else:
                print(f"{C.SYSTEM}Action cancelled by user.{C.RESET}")
                return "Action was cancelled by the user."
        except Exception as e:
            return f"Failed to execute command: {e}"

    async def get_voice_confirmation(self):
        # Empty audio queue first
        if self.audio_queue:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
            
        print(f"{C.SYSTEM}[Listening for confirmation (yes/no)...]{C.RESET}")
        speaking_active = False
        last_voice_time = time.time()
        
        while True:
            data = await self.audio_queue.get()
            is_speech, _, _ = self.vad.process_frame(data)
            self.rec.AcceptWaveform(data)
            
            if is_speech:
                speaking_active = True
                last_voice_time = time.time()
                
            if speaking_active and (time.time() - last_voice_time) > 1.2:
                result = json.loads(self.rec.FinalResult())
                text = result.get("text", "").strip().lower()
                self.rec.Reset()
                
                if text:
                    print(f"{C.USER}YOU >> {text}{C.RESET}")
                    if any(word in text for word in ["yes", "yep", "sure", "confirm", "go ahead"]):
                        return True
                    if any(word in text for word in ["no", "nope", "cancel", "stop"]):
                        return False
                
                # If neither recognized, ask again
                await self.speak_queue.put("Please say yes or no to confirm.")
                await self.speak_queue.join()
                speaking_active = False

    async def run_proactive_checks(self):
        # Proactively check system load
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            if ram > 90.0:
                await self.speak_queue.put("System memory usage is critically high, exceeding ninety percent. Consider closing idle background processes.")
            elif cpu > 90.0:
                await self.speak_queue.put("Processor utilization is currently above ninety percent.")
        except Exception as e:
            pass

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        self.speak_queue = asyncio.Queue()
        init_db()
        self.start_recording()
        
        # Start TTS consumer task in background
        tts_task = asyncio.create_task(self.tts_worker())
        
        ASCII_BANNER = r"""
       :@@@:      %@@@@@@@@%        %@@@@@=                         
      :@@@@@      %@@@@@@@@@@%   o@@@@@%@@@@%                       
     .@@@ @@@     %@@      .@@@ %@>       @                        
     @@%   @@@    %@@       @@@@@@-                                 
    @@@     @@%   %@@  :::"@@@"@@@                                  
   @@@   ...<@@%  %@@  o@@@@%  @@@                                  
  @@@   %@@@@@@@@ %@@    @@@    @@@                                 
 @@@:         "@@@@@@     @@@o   @@@@o  :%@@@.                      
"@@:           "@@@@@      o@@%_  :%@@@@@@@o    .AI                 
              Autonomous Reasoning Core - v3 (Async Engine)
        """
        print(C.ARC + ASCII_BANNER + C.RESET)
        winsound.Beep(760, 500)
        print(f"{C.SYSTEM}ARC is Listening...{C.RESET}\n")
        
        speaking_active = False
        last_voice_time = time.time()
        
        # Periodic monitoring checker
        asyncio.create_task(self.proactive_monitoring_scheduler())
        
        try:
            while True:
                data = await self.audio_queue.get()
                is_speech, rms, threshold = self.vad.process_frame(data)
                
                # Check for audio trigger
                self.rec.AcceptWaveform(data)
                
                if is_speech:
                    if not speaking_active:
                        speaking_active = True
                        # Beep to indicate user speech started
                        winsound.Beep(500, 100)
                    last_voice_time = time.time()
                    
                # Process speech completion after pause
                if speaking_active and (time.time() - last_voice_time) > SILENCE_TIMEOUT:
                    result = json.loads(self.rec.FinalResult())
                    text = result.get("text", "").strip()
                    
                    if text:
                        winsound.Beep(600, 300)
                        print(f"\n{C.USER}YOU >> {C.RESET}{C.TEXT}{text}{C.RESET}")
                        
                        # Add user input to working context FIRST (chronological order)
                        await self.add_to_context("user", text)
                        
                        # Build memory context and query SQLite FTS5 database
                        memory_context = ""
                        if any(keyword in text.lower() for keyword in ["recall", "remember", "history", "past"]):
                            # Clean query terms for retrieval
                            query_terms = text.replace("recall", "").replace("remember", "").strip()
                            memory_context = recall_memory(query_terms)
                        
                        # Build messages list starting with instructions
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                        
                        # Add system status dynamically for context awareness (placed near top as prefix info)
                        sys_stats = get_system_report()
                        messages.append({
                            "role": "system",
                            "content": f"Current host stats: CPU: {sys_stats['cpu']} | RAM: {sys_stats['ram']} | Battery: {sys_stats['battery']}"
                        })
                        
                        # Add local workspace file context to prevent dummy/placeholder path generation
                        cwd = os.getcwd()
                        try:
                            local_files = os.listdir(cwd)
                        except Exception:
                            local_files = []
                        messages.append({
                            "role": "system",
                            "content": f"Current workspace directory: {cwd}\nAvailable files/folders in this directory: {local_files}"
                        })
                        
                        # Add memory context if found
                        if memory_context:
                            messages.append({
                                "role": "system",
                                "content": f"Relevant historical memories retrieved from local storage:\n{memory_context}"
                            })
                            print(f"{C.DEV}[RETRIEVED MEMORY]:\n{memory_context}{C.RESET}")
                            
                        # Add working window context (ends with the current user message)
                        messages.extend(self.working_context)
                        
                        # Retrieve LLM reply with streaming
                        t_start = time.time()
                        
                        try:
                            # Run chat streaming
                            response_stream = await asyncio.to_thread(
                                ollama.chat,
                                model="llama3.1:latest",
                                messages=messages,
                                stream=True
                            )
                            
                            print(f"\n {C.ARC}ARC >>{C.RESET} ", end="", flush=True)
                            
                            accumulated_response = ""
                            sentence_buffer = ""
                            first_token_latency = None
                            
                            # Read chunks and split into sentences for low perceived TTS latency
                            for chunk in response_stream:
                                if first_token_latency is None:
                                    first_token_latency = time.time() - t_start
                                    
                                content = chunk['message']['content']
                                accumulated_response += content
                                sentence_buffer += content
                                print(C.TEXT + content, end="", flush=True)
                                
                                # Send sentences to TTS player queue as they complete
                                while True:
                                    m = re.search(r'[.!?](\s+|$)', sentence_buffer)
                                    if not m:
                                        break
                                    end_idx = m.end()
                                    sentence = sentence_buffer[:end_idx].strip()
                                    sentence_buffer = sentence_buffer[end_idx:]
                                    
                                    if sentence and not sentence.startswith("ACTION:"):
                                        await self.speak_queue.put(sentence)
                                        
                            # Send final remaining sentence segment
                            remaining = sentence_buffer.strip()
                            if remaining and not remaining.startswith("ACTION:"):
                                await self.speak_queue.put(remaining)
                            
                            print()  # final newline
                            
                            # Parse actions if suggested by model
                            if "ACTION:" in accumulated_response:
                                action_part = accumulated_response.split("ACTION:")[1].strip()
                                action_result = await self.execute_action(action_part)
                                
                                # Feed execution outcome back into context for final wrap-up
                                followup_messages = messages + [
                                    {"role": "assistant", "content": accumulated_response},
                                    {"role": "system", "content": f"Execution outcome: {action_result}"}
                                ]
                                
                                final_res = await asyncio.to_thread(
                                    ollama.chat, model="llama3.1:latest", messages=followup_messages
                                )
                                final_text = final_res["message"]["content"]
                                print(f"\n {C.ARC}ARC >>{C.RESET} {C.TEXT}{final_text}{C.RESET}")
                                await self.speak_queue.put(final_text)
                                await self.add_to_context("assistant", final_text)
                            else:
                                await self.add_to_context("assistant", accumulated_response)
                                
                            # Check if context exceeds limit and condense at the very end of the turn
                            if len(self.working_context) >= self.max_window:
                                await self.condense_and_store_memory()
                            
                            # Print metrics
                            total_latency = time.time() - t_start
                            print(f"\n{C.DEV}[METRICS] First-token latency: {first_token_latency:.2f}s | Total response time: {total_latency:.2f}s{C.RESET}")
                            
                        except Exception as e:
                            print(f"{C.SYSTEM}Inference error: {e}{C.RESET}")
                            await self.speak_queue.put("I encountered an inference error.")
                            
                    self.rec.Reset()
                    speaking_active = False

        except KeyboardInterrupt:
            print("\nShutting down ARC...")
        finally:
            self.stop_recording()
            tts_task.cancel()

    async def proactive_monitoring_scheduler(self):
        # Run every 5 minutes
        while True:
            await asyncio.sleep(300)
            await self.run_proactive_checks()

if __name__ == "__main__":
    init_db()
    system = ARCSystem()
    asyncio.run(system.run())
