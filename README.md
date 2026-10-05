# English Conversation Trainer (v2 — Real Voice MVP)

A focused, terminal-based English speaking practice application. This MVP validates a structured conversational learning method: practicing realistic, pre-scripted dialogues turn-by-turn with AI voice synthesis, interactive microphone recording, high-accuracy speech-to-text, semantic evaluation, and an end-of-session diagnostic report card.

---

## Table of Contents

1. [Overview & Learning Method](#1-overview--learning-method)
2. [What We Used to Build This (Tech Stack)](#2-what-we-used-to-build-this-tech-stack)
3. [How It Was Built (Architecture & Codebase)](#3-how-it-was-built-architecture--codebase)
4. [Installation & Setup (How to Run)](#4-installation--setup-how-to-run)
5. [User Guide (Step-by-Step Walkthrough)](#5-user-guide-step-by-step-walkthrough)
6. [Scenarios Catalog](#6-scenarios-catalog)
7. [Troubleshooting & FAQs](#7-troubleshooting--faqs)

---

## 1. Overview & Learning Method

The application is built around a progressive learning progression:

$$\text{SCRIPTED} \longrightarrow \text{GUIDED (Full Hint)} \longrightarrow \text{INDEPENDENT (No Hint)} \longrightarrow \text{SPONTANEOUS}$$

- **Scripted Control:** Conversations follow realistic, structured dialogues defined in [`levels.json`](levels.json). The curriculum is controlled and deterministic—the AI does not randomly invent scenarios.
- **Intelligent Evaluation:** The LLM evaluates your spoken responses dynamically. It assesses communication, grammar, and naturalness **without** penalizing you for speaking natural variations (e.g., saying *"Can I get a large iced caramel latte?"* instead of the exact line *"I would like to order a large iced caramel latte, please"* is recognized as a valid, high-scoring answer).
- **Diagnostic Feedback:** Concise feedback is delivered during the conversation so your speaking flow isn't interrupted. Detailed strengths, grammar patterns, and recommendations appear in a comprehensive report card at the end.

---

## 2. What We Used to Build This (Tech Stack)

The system is built entirely in Python using modern, fast AI APIs and local audio libraries:

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.10+ (tested on Python 3.14) | Core application logic and execution |
| **Speech-to-Text (STT)** | **Groq Whisper** (`whisper-large-v3`) | Fast, accurate transcription of spoken English via Groq Python SDK |
| **Text-to-Speech (TTS)** | **Groq TTS** (`canopylabs/orpheus-v1-english`) | Natural, expressive conversational character voice output |
| **AI Evaluator & Reporter** | **Groq Chat Models** (`qwen/qwen3.8-27b` / `llama-3.1-8b-instant`) | Turn-by-turn conversational scoring and end-of-session diagnostic analysis |
| **Audio Recording & Playback** | `sounddevice` | Cross-platform PortAudio bindings for low-latency microphone capture and WAV playback |
| **Audio Processing** | `numpy` & `scipy.io.wavfile` | 16 kHz mono audio buffer concatenation and WAV encoding |
| **Environment Management** | `python-dotenv` | Clean, secure loading of configuration and API keys from `.env` |
| **Curriculum Database** | `levels.json` | Human-editable structured JSON scenarios containing turns, speakers, and expected lines |

> **Why we replaced earlier libraries:**
> - Completely removed `SpeechRecognition` and Google Speech backend due to unreliability, network dropping, and silent failures.
> - Completely removed `pyttsx3` to eliminate robotic synthesizer stuttering and platform audio device locking.
> - Switched to Groq's official unified cloud audio & LLM infrastructure for speed, accuracy, and natural voice delivery.

---

## 3. How It Was Built (Architecture & Codebase)

The codebase follows a modular design where each file has a single responsibility:

```text
english-conversation-trainer/
│
├── main.py                  # CLI entrypoint, startup banner, and menu routing
├── conversation_engine.py   # State machine managing turns, retries, and session history
├── audio_recorder.py        # Interactive microphone recording with Start/Stop controls
├── speech_to_text.py        # Groq Whisper integration and transcription error handling
├── text_to_speech.py        # Groq TTS client and local sounddevice audio player
├── evaluator.py             # Strict JSON evaluator assessing communication, grammar & naturalness
├── report.py                # Final report card generator analyzing complete conversation transcript
├── levels.json              # Pre-scripted conversational scenarios and turns
├── requirements.txt         # Minimal required Python dependencies
├── .env.example             # Template for required environment variables
├── .gitignore               # Excludes secrets, virtualenvs, caches, and audio files
└── README.md                # Project documentation and user guide
```

### Turn-by-Turn Execution Pipeline:

```text
REAL-WORLD SCENARIO
       │
       ▼
BOT LINE DISPLAYED & SPOKEN
(Groq TTS via canopylabs/orpheus-v1-english ➔ sounddevice playback)
       │
       ▼
LEARNER SEES HINT (Full Hint mode) OR LISTENS ONLY (No Hint mode)
       │
       ▼
INTERACTIVE MICROPHONE RECORDING
(Press Enter to start ➔ speak naturally ➔ Press Enter to stop)
       │
       ▼
LOCAL WAV BUFFER (16 kHz mono)
       │
       ▼
GROQ WHISPER STT (whisper-large-v3)
       │
       ▼
YOU: TRANSCRIPTION DISPLAYED
       │
       ▼
GROQ EVALUATION (Communication, Grammar, Naturalness, Relevance: 0-10)
       │
       ▼
IMMEDIATE CONCISE FEEDBACK & SCORE (✓ Good! / Almost correct.)
       │
       ▼
NEXT SCRIPTED TURN (Repeats until all turns complete)
       │
       ▼
COMPREHENSIVE FINAL REPORT CARD
(Groq analyzes full transcript: strengths, weaknesses with examples, and next practice focus)
```

---

## 4. Installation & Setup (How to Run)

### 1. Prerequisites
- **Python:** Python 3.10 or higher.
- **Hardware:** Working microphone and speakers/headphones.
- **Linux Audio Libraries (Ubuntu/Debian):**
  ```bash
  sudo apt-get install -y portaudio19-dev
  ```
  *(Note: PortAudio and JACK shared libraries are pre-configured in local user environment if building without root privileges).*

### 2. Set Up Virtual Environment & Dependencies
Navigate to the project directory:

```bash
cd /home/sadik-khan/SadikWork/Hodor/econ_trainer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Your API Key (.env)
Copy the example environment file to `.env`:

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your details:

```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_STT_MODEL=whisper-large-v3
GROQ_TTS_MODEL=canopylabs/orpheus-v1-english
GROQ_CHAT_MODEL=qwen/qwen3.8-27b
```

> **API Key:** Get a free key from the [Groq Console](https://console.groq.com).  
> **Chat Model:** Supported fast models include `qwen/qwen3.8-27b`, `llama-3.1-8b-instant`, or `llama-3.3-70b-versatile` depending on your account tier.  
> **TTS Model:** `canopylabs/orpheus-v1-english` requires a one-time terms acceptance on the [Groq Playground](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english).

### 4. Launch the Application
Run the main script:

```bash
python main.py
```

---

## 5. User Guide (Step-by-Step Walkthrough)

### Step 1: Choose a Situation
When launched, you will see the startup banner and scenario list:

```text
========================================
     ENGLISH CONVERSATION TRAINER
========================================

Practice English through realistic conversations.

Choose a situation:

1. Family - Morning
2. College - Talking to a Friend
3. Office - Morning Meeting
4. Coffee Shop - Ordering
5. Shopping - Buying Something

Enter choice: 4
```

Type the number of the scenario you want to practice and press **Enter**.

---

### Step 2: Choose Your Practice Mode
Next, select your practice assistance level:

```text
How would you like to practice?

1. Full Hint
2. No Hint

Enter choice: 1
```

- **1. Full Hint (Guided Practice):** Shows the target English sentence (`YOUR LINE:`) before you speak. Recommended for beginners who want to build vocabulary, learn conversational structures, and practice vocalization.
- **2. No Hint (Independent Practice):** Only plays the bot's speech and displays the context. You must formulate and speak your own response based on the conversation so far.

---

### Step 3: Listen to the Bot
The character speaks the line aloud through your speakers/headphones and prints the dialogue in the terminal:

```text
----------------------------------------

BOT:
"Hi there! Welcome to Daily Roast Cafe. What can I get started for you today?"
```

In Full Hint mode, your target sentence is shown:

```text
YOUR LINE:
"Hi! I would like to order a large iced caramel latte, please."
```

---

### Step 4: Record Your Voice (Interactive Controls)
When it is your turn to speak, you will see the prompt:

```text
----------------------------------------
🎤 YOUR TURN
Press Enter to start recording.
----------------------------------------
```

1. Press **Enter** once to begin recording.
2. The terminal displays:
   ```text
   🔴 Recording... Speak now. (Press Enter to stop)
   ```
3. Speak your sentence clearly into your microphone at a natural pace.
4. Press **Enter** again when you finish speaking.
5. The terminal confirms:
   ```text
   ⏹ Recording stopped.
   🧠 Transcribing...
   ```

---

### Step 5: Read Feedback & Continue
The application transcribes what you said, submits it to Groq for evaluation, and displays instant feedback:

```text
YOU:
"Can I get a large iced caramel latte, please?"

🧠 Evaluating...

FEEDBACK:
✓ Excellent! Your order was clear, polite, and perfectly natural.

Score: 10/10
----------------------------------------
```

If you made a grammatical error, the system provides an encouraging correction without stopping the conversation:

```text
FEEDBACK:
Almost correct.

Correction:
"Yesterday, I went to college after breakfast."

Note: Great job! Just remember to use past tense 'went' instead of 'go' for actions yesterday.

Score: 8/10
----------------------------------------
```

The bot then speaks the next line in the script, and the process repeats until all turns are complete.

---

### Step 6: Review the Diagnostic Report Card
At the end of the dialogue, Groq analyzes your complete transcript and renders a diagnostic Report Card:

```text
========================================
          CONVERSATION REPORT
========================================

Scenario:
Coffee Shop - Ordering

Turns:
10 / 10

Overall:
9.8 / 10

Communication:
10 / 10

Grammar:
10 / 10

Naturalness:
9 / 10

Relevance:
10 / 10

----------------------------------------

WHAT YOU DID WELL

- Clear and polite communication of order details
- Natural use of courteous requests ('please', 'could I get')

----------------------------------------

MAIN WEAKNESSES

1. Slightly abrupt phrasing in Turn 4 ('warm it up' vs 'that would be wonderful')

----------------------------------------

EXAMPLES

You said:
"Yes please, warm it up."

Better:
"Yes, please. That would be wonderful."

----------------------------------------

NEXT PRACTICE FOCUS

Your ordering skills are strong. Try practicing polite phrases for accepting additional services (e.g., 'That would be great!').

========================================
```

---

### Fallback & In-Session Controls

- **Retry Audio:** If no speech was captured or background noise garbled your words:
  ```text
  ⚠️ I couldn't understand your speech.
  Please try again.

  [Enter] Retry
  [T] Type response directly
  [Q] Quit scenario
  ```
  Press **Enter** to retry speaking the turn immediately.
- **Keyboard Fallback (`T`):** If your microphone hardware disconnects or malfunctions, press `T` to type your response in the terminal so you can finish your session without interruption.
- **Quit (`Q`):** Press `Q` at any retry prompt or press `Ctrl + C` at any time to safely exit.

---

## 6. Scenarios Catalog

All scenarios in [`levels.json`](levels.json) are designed around realistic, everyday spoken English:

1. **Family - Morning (10 turns):**
   - *Topics:* Waking up, asking how the other person slept, breakfast choices, morning weather, and daily plans.
   - *Skill focus:* Daily routines, conversational greetings, and future plans.
2. **College - Talking to a Friend (10 turns):**
   - *Topics:* Heading to lectures, discussing difficult assignments, comparing answers, and arranging to meet for lunch.
   - *Skill focus:* Casual peer conversation, school subjects, and scheduling.
3. **Office - Morning Meeting (10 turns):**
   - *Topics:* Stand-up meeting greeting, reporting completed tasks, outlining today's priorities, and checking blockers.
   - *Skill focus:* Professional business English, past and present work status.
4. **Coffee Shop - Ordering (10 turns):**
   - *Topics:* Placing drink orders, specifying milk options, ordering bakery items, warming preferences, and paying by card.
   - *Skill focus:* Service encounters, polite requests, food/drink specifications.
5. **Shopping - Buying Something (10 turns):**
   - *Topics:* Inquiring about jackets, size checks, color availability, asking about discounts, and checkout.
   - *Skill focus:* Retail transactions, sizing inquiries, pricing questions.

---

## 7. Troubleshooting & FAQs

### Q: Why do I see a notice about `model_terms_required` for Orpheus TTS?
**A:** Groq's `canopylabs/orpheus-v1-english` voice model requires a one-time terms acceptance.
1. Visit [Groq Playground](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english).
2. Click **Accept Terms** while logged into your Groq organization.
3. Relaunch `python main.py`.

*(Note: If you have not accepted terms yet, the application will display the bot text in the terminal and continue without crashing).*

### Q: Why did I receive `GROQ_CHAT_MODEL is unavailable`?
**A:** Different Groq API keys and tiers have access to different models. Check your available models on the Groq Console or run:
```bash
python3 -c "import os; from groq import Groq; print([m.id for m in Groq().models.list().data])"
```
Then set `GROQ_CHAT_MODEL` in your `.env` file to any active chat model (e.g. `GROQ_CHAT_MODEL=qwen/qwen3.8-27b`).

### Q: How do I change the recording volume or select a different microphone?
**A:** The application uses your system's default input device. You can verify available devices in python:
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```
Adjust your microphone input gain in your OS sound settings (e.g., `pavucontrol` or GNOME Sound Settings on Linux).
