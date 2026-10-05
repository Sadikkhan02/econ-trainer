"""Diagnostic report generator for English Conversation Trainer.

Analyzes full session dialogue transcripts using Groq and renders the structured report card.
"""

import json
from typing import Any, Dict, List
from groq import Groq


class ReportGenerator:
    """Generates comprehensive end-of-session evaluation reports."""

    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model = model

    def generate_report(
        self,
        scenario_name: str,
        total_turns: int,
        completed_turns: int,
        turn_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze the complete conversation transcript with Groq and return report data."""
        system_prompt = (
            "You are an expert English conversation examiner analyzing a full practice session transcript.\n"
            "Review the student's actual spoken turns and performance.\n"
            "CRITICAL RULES:\n"
            "1. Base your evaluation ONLY on the actual dialogue provided. Do NOT invent mistakes that do not appear.\n"
            "2. Identify genuine strengths, recurring grammar/vocabulary patterns, and the main weakness.\n"
            "3. If an error occurred, provide the exact sentence spoken by the learner and a natural correction.\n"
            "4. Return strict JSON with numeric ratings from 0 to 10.\n\n"
            "JSON SCHEMA:\n"
            "{\n"
            '  "overall": float (e.g. 8.1),\n'
            '  "communication": integer (0-10),\n'
            '  "grammar": integer (0-10),\n'
            '  "naturalness": integer (0-10),\n'
            '  "relevance": integer (0-10),\n'
            '  "strengths": [list of 2-3 brief strings],\n'
            '  "weaknesses": [list of 1-2 brief strings],\n'
            '  "example_user": string or null,\n'
            '  "example_better": string or null,\n'
            '  "recommendation": string\n'
            "}"
        )

        formatted_dialogue = []
        for i, turn in enumerate(turn_history, 1):
            formatted_dialogue.append(
                f"Turn {i}:\n"
                f"BOT: {turn.get('bot')}\n"
                f"YOU: {turn.get('user')}\n"
                f"(Expected: {turn.get('expected')})\n"
                f"(Score: {turn.get('score')}/10, Grammar: {turn.get('grammar')}/10)\n"
            )

        user_content = (
            f"Scenario: {scenario_name}\n"
            f"Turns Completed: {completed_turns} / {total_turns}\n\n"
            "Session Transcript:\n" + "\n".join(formatted_dialogue)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data
        except Exception:
            # Fallback calculation if Groq call is interrupted
            scores = [t.get("score", 8) for t in turn_history] or [8]
            avg = round(sum(scores) / len(scores), 1)
            comm = [t.get("communication", 8) for t in turn_history] or [8]
            gram = [t.get("grammar", 8) for t in turn_history] or [8]
            nat = [t.get("naturalness", 8) for t in turn_history] or [8]
            rel = [t.get("relevance", 8) for t in turn_history] or [8]

            return {
                "overall": avg,
                "communication": round(sum(comm) / len(comm)),
                "grammar": round(sum(gram) / len(gram)),
                "naturalness": round(sum(nat) / len(nat)),
                "relevance": round(sum(rel) / len(rel)),
                "strengths": [
                    "You responded appropriately.",
                    "Your meaning was clear.",
                    "You maintained the flow of the conversation.",
                ],
                "weaknesses": ["Minor sentence structure variations"],
                "example_user": turn_history[0].get("user") if turn_history else None,
                "example_better": turn_history[0].get("expected") if turn_history else None,
                "recommendation": "Continue practicing with realistic scenarios to build spontaneous fluency.",
            }

    @staticmethod
    def display_report_card(
        scenario_name: str,
        total_turns: int,
        completed_turns: int,
        report_data: Dict[str, Any],
    ) -> None:
        """Display the formatted report card in the terminal."""
        overall = report_data.get("overall", 8.0)
        communication = report_data.get("communication", 8)
        grammar = report_data.get("grammar", 8)
        naturalness = report_data.get("naturalness", 8)
        relevance = report_data.get("relevance", 8)
        strengths = report_data.get("strengths", ["Responded appropriately."])
        weaknesses = report_data.get("weaknesses", ["None observed"])
        example_user = report_data.get("example_user")
        example_better = report_data.get("example_better")
        recommendation = report_data.get("recommendation", "Practice again to increase fluency.")

        print("\n" + "=" * 40)
        print("          CONVERSATION REPORT")
        print("=" * 40)
        print(f"\nScenario:\n{scenario_name}")
        print(f"\nTurns:\n{completed_turns} / {total_turns}")
        print(f"\nOverall:\n{overall} / 10")
        print(f"\nCommunication:\n{communication} / 10")
        print(f"\nGrammar:\n{grammar} / 10")
        print(f"\nNaturalness:\n{naturalness} / 10")
        print(f"\nRelevance:\n{relevance} / 10")
        print("\n" + "-" * 40)

        print("\nWHAT YOU DID WELL\n")
        for item in strengths:
            print(f"- {item}")

        print("\n" + "-" * 40)

        print("\nMAIN WEAKNESSES\n")
        for i, item in enumerate(weaknesses, 1):
            print(f"{i}. {item}")

        if example_user and example_better:
            print("\n" + "-" * 40)
            print("\nEXAMPLES\n")
            print(f"You said:\n\"{example_user}\"\n")
            print(f"Better:\n\"{example_better}\"")

        print("\n" + "-" * 40)
        print(f"\nNEXT PRACTICE FOCUS\n")
        print(f"{recommendation}")
        print("\n" + "=" * 40 + "\n")
