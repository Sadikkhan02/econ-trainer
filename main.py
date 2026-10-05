"""English Conversation Trainer MVP (v2).

Terminal-based voice-enabled English speaking trainer powered by Groq.
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

from audio_recorder import AudioRecorder
from conversation_engine import ConversationEngine
from evaluator import GroqEvaluator
from report import ReportGenerator
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech


def print_banner() -> None:
    """Display startup banner."""
    print("========================================")
    print("     ENGLISH CONVERSATION TRAINER")
    print("========================================")
    print("\nPractice English through realistic conversations.\n")


def get_api_key() -> str:
    """Validate presence of GROQ_API_KEY."""
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("\nERROR:")
        print("GROQ_API_KEY is missing.\n")
        print("Please create a .env file and add:\n")
        print("GROQ_API_KEY=your_key_here\n")
        print("Then run the application again.")
        sys.exit(1)
    return key


def select_scenario(engine: ConversationEngine):
    """Prompt user to choose a situation."""
    print("Choose a situation:\n")
    for idx, scenario in enumerate(engine.scenarios, 1):
        print(f"{idx}. {scenario.get('name')}")

    while True:
        try:
            choice_str = input("\nEnter choice: ").strip()
            if not choice_str:
                continue
            choice_num = int(choice_str)
            scenario = engine.get_scenario_by_index(choice_num - 1)
            if scenario:
                return scenario
            print(f"Please enter a number between 1 and {len(engine.scenarios)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(engine.scenarios)}.")


def select_hint_mode() -> bool:
    """Prompt user to choose hint mode."""
    print("\nHow would you like to practice?\n")
    print("1. Full Hint")
    print("2. No Hint\n")

    while True:
        choice = input("Enter choice: ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Invalid choice. Please enter 1 for Full Hint or 2 for No Hint.")


def main():
    try:
        print_banner()

        api_key = get_api_key()
        client = Groq(api_key=api_key)

        chat_model = os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant").strip()
        stt_model = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3").strip()
        tts_model = os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english").strip()

        recorder = AudioRecorder(sample_rate=16000)
        stt = SpeechToText(client=client, model=stt_model)
        tts = TextToSpeech(client=client, model=tts_model)

        try:
            evaluator = GroqEvaluator(client=client, model=chat_model)
        except Exception as e:
            print(f"\nERROR:\n{e}\n")
            sys.exit(1)

        reporter = ReportGenerator(client=client, model=chat_model)

        levels_path = os.path.join(os.path.dirname(__file__), "levels.json")
        try:
            engine = ConversationEngine(
                levels_file=levels_path,
                recorder=recorder,
                stt=stt,
                tts=tts,
                evaluator=evaluator,
                reporter=reporter,
            )
        except Exception as e:
            print(f"\nERROR:\nFailed to initialize conversation engine: {e}\n")
            sys.exit(1)

        selected_scenario = select_scenario(engine)
        full_hint = select_hint_mode()

        engine.run_session(scenario=selected_scenario, full_hint=full_hint)

    except KeyboardInterrupt:
        print("\n\nSession cancelled by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
