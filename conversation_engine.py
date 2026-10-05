"""Conversation Engine for English Conversation Trainer (v2).

Coordinates scripted turns, recording, Whisper transcription, Groq TTS,
AI turn evaluation, and session reporting.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from audio_recorder import AudioRecorder
from evaluator import GroqEvaluator
from report import ReportGenerator
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech


class ConversationSession:
    """Tracks state and metrics for an individual conversation practice session."""

    def __init__(self, scenario: Dict[str, Any], full_hint: bool):
        self.scenario = scenario
        self.scenario_name = scenario.get("name", "English Conversation")
        self.turns = scenario.get("turns", [])
        self.full_hint = full_hint
        self.current_turn_index = 0
        self.history: List[Dict[str, Any]] = []
        self.retries = 0
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def record_turn(
        self,
        bot_line: str,
        expected_line: str,
        user_line: str,
        evaluation: Dict[str, Any],
    ) -> None:
        """Add a completed turn to session history."""
        self.history.append({
            "bot": bot_line,
            "expected": expected_line,
            "user": user_line,
            "score": evaluation.get("score", 8),
            "communication": evaluation.get("communication", 8),
            "grammar": evaluation.get("grammar", 8),
            "naturalness": evaluation.get("naturalness", 8),
            "relevance": evaluation.get("relevance", 8),
            "passed": evaluation.get("passed", True),
            "feedback": evaluation.get("feedback", ""),
            "correction": evaluation.get("correction", ""),
        })


class ConversationEngine:
    """Orchestrates scenario execution and audio/LLM integrations."""

    def __init__(
        self,
        levels_file: str,
        recorder: AudioRecorder,
        stt: SpeechToText,
        tts: TextToSpeech,
        evaluator: GroqEvaluator,
        reporter: ReportGenerator,
    ):
        self.levels_file = levels_file
        self.recorder = recorder
        self.stt = stt
        self.tts = tts
        self.evaluator = evaluator
        self.reporter = reporter
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load available scenarios from levels.json."""
        if not os.path.exists(self.levels_file):
            raise FileNotFoundError(f"Levels file '{self.levels_file}' not found.")
        with open(self.levels_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("scenarios", [])

    def get_scenario_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Return scenario by 0-based index."""
        if 0 <= index < len(self.scenarios):
            return self.scenarios[index]
        return None

    def run_session(self, scenario: Dict[str, Any], full_hint: bool) -> bool:
        """Run a full turn-by-turn conversation session."""
        session = ConversationSession(scenario=scenario, full_hint=full_hint)
        turns = session.turns
        total_turns = len(turns)

        print("\n========================================")
        print("     ENGLISH CONVERSATION TRAINER")
        print("========================================")
        print(f"\nScenario:\n{session.scenario_name}")
        print(f"\nMode:\n{'Full Hint' if full_hint else 'No Hint'}")

        last_bot_line = ""
        completed_count = 0

        for turn_idx, turn in enumerate(turns):
            speaker = turn.get("speaker")

            if speaker == "bot":
                bot_line = turn.get("line", "")
                last_bot_line = bot_line

                print("\n----------------------------------------")
                print(f"\nBOT:\n\"{bot_line}\"\n")

                # Bot speaks clearly with Groq TTS and local playback
                self.tts.speak(bot_line)
                completed_count += 1
                time.sleep(0.5)

            elif speaker == "user":
                expected_line = turn.get("expected", "")

                if full_hint:
                    print("----------------------------------------")
                    print(f"\nYOUR LINE:\n\"{expected_line}\"\n")

                # Learner speech recording and transcription loop with retries
                actual_text = None
                while not actual_text:
                    wav_path = self.recorder.record_interactive()

                    if wav_path == "QUIT":
                        print("\nSession stopped by user.")
                        return False

                    if not wav_path:
                        session.retries += 1
                        print("\n⚠️ I couldn't understand your speech.")
                        print("Please try again.\n")
                        print("[Enter] Retry")
                        print("[T] Type response directly (if microphone issue)")
                        print("[Q] Quit scenario")
                        choice = input("Choice: ").strip().lower()
                        if choice == "q":
                            print("\nSession cancelled.")
                            return False
                        elif choice == "t":
                            typed = input("Type your response: ").strip()
                            if typed:
                                actual_text = typed
                        continue

                    # Transcribe with Groq Whisper
                    print("\n🧠 Transcribing...")
                    stt_result = self.stt.transcribe_file(wav_path)
                    self.recorder.cleanup_file(wav_path)

                    if not stt_result.get("success"):
                        session.retries += 1
                        err = stt_result.get("error", "Unknown error")
                        print(f"\n⚠️ I couldn't understand the audio: {err}")
                        print("Please try again.\n")
                        print("[Enter] Retry")
                        print("[T] Type response directly")
                        print("[Q] Quit scenario")
                        choice = input("Choice: ").strip().lower()
                        if choice == "q":
                            print("\nSession cancelled.")
                            return False
                        elif choice == "t":
                            typed = input("Type your response: ").strip()
                            if typed:
                                actual_text = typed
                        continue

                    actual_text = stt_result.get("text", "").strip()

                print(f"\nYOU:\n\"{actual_text}\"\n")

                # AI Evaluation with Groq
                print("🧠 Evaluating...")
                recent_ctx = [
                    {"bot": h["bot"], "learner": h["user"]}
                    for h in session.history[-3:]
                ]
                evaluation = self.evaluator.evaluate_turn(
                    scenario_name=session.scenario_name,
                    bot_line=last_bot_line,
                    expected_line=expected_line,
                    learner_transcript=actual_text,
                    recent_context=recent_ctx,
                )

                # Show concise immediate feedback
                score = evaluation.get("score", 8)
                passed = evaluation.get("passed", True)
                correction = evaluation.get("correction", "")

                print("\nFEEDBACK:")
                if passed and score >= 8:
                    print(f"✓ {evaluation.get('feedback', 'Good response!')}")
                elif correction:
                    print("Almost correct.\n")
                    print(f"Correction:\n\"{correction}\"\n")
                    if evaluation.get("feedback"):
                        print(f"Note: {evaluation.get('feedback')}")
                else:
                    print(evaluation.get("feedback", "Good response."))

                print(f"\nScore: {score}/10")
                print("----------------------------------------")

                session.record_turn(
                    bot_line=last_bot_line,
                    expected_line=expected_line,
                    user_line=actual_text,
                    evaluation=evaluation,
                )
                completed_count += 1
                time.sleep(0.5)

        session.end_time = time.time()

        # Generate and display final report
        print("\n🧠 Generating comprehensive Conversation Report...")
        report_data = self.reporter.generate_report(
            scenario_name=session.scenario_name,
            total_turns=total_turns,
            completed_turns=completed_count,
            turn_history=session.history,
        )
        self.reporter.display_report_card(
            scenario_name=session.scenario_name,
            total_turns=total_turns,
            completed_turns=completed_count,
            report_data=report_data,
        )
        return True
