"""AI Evaluator module using Groq chat completions for English Conversation Trainer.

Evaluates user conversational turns across communication, grammar, naturalness, and relevance.
"""

import json
import os
from typing import Any, Dict, List, Optional
from groq import Groq


class GroqEvaluator:
    """Evaluates conversational English turns using Groq chat completion models."""

    def __init__(self, client: Groq, model: Optional[str] = None):
        self.client = client
        self.model = model or os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant").strip()
        self._validate_model()

    def _validate_model(self) -> None:
        """Validate that the configured chat model is accessible."""
        try:
            # Quick validation call
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except Exception as e:
            err_msg = str(e)
            if "model_not_found" in err_msg or "does not exist" in err_msg:
                try:
                    available = [m.id for m in self.client.models.list().data if "whisper" not in m.id and "orpheus" not in m.id]
                    hint = f" Available chat models on your account: {', '.join(available)}."
                except Exception:
                    hint = ""
                raise RuntimeError(
                    f"The configured GROQ_CHAT_MODEL '{self.model}' is unavailable or not accessible on your Groq account.{hint}\n"
                    f"Please update GROQ_CHAT_MODEL in your .env file."
                )
            # If rate limited or other transient error, allow initialization to proceed
            pass

    def evaluate_turn(
        self,
        scenario_name: str,
        bot_line: str,
        expected_line: str,
        learner_transcript: str,
        recent_context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a learner's spoken response.

        Accepts natural variations, idioms, and synonyms as long as the communicative intent fits.
        Returns strict JSON matching:
        {
          "score": int (0-10),
          "communication": int (0-10),
          "grammar": int (0-10),
          "naturalness": int (0-10),
          "relevance": int (0-10),
          "passed": bool,
          "feedback": str,
          "correction": str,
          "strength": str,
          "weakness": str
        }
        """
        system_prompt = (
            "You are an encouraging, expert English conversation tutor evaluating a learner in an English-speaking MVP.\n"
            "Scenario: " + scenario_name + "\n\n"
            "CRITICAL EVALUATION RULES:\n"
            "1. Do NOT require a word-for-word verbatim repetition of the expected line.\n"
            "2. Accept natural variations, polite contractions, and synonymous expressions.\n"
            "   Example: If expected is 'I would like a large iced caramel latte, please', and the learner says\n"
            "   'Can I get a large iced caramel latte?', this is an excellent, natural response (score 9 or 10).\n"
            "3. Assess five criteria from 0 to 10:\n"
            "   - communication: Was the intended message clear and understandable?\n"
            "   - grammar: Is the sentence structure acceptable for spoken conversational English?\n"
            "   - naturalness: Does it sound like authentic everyday speech?\n"
            "   - relevance: Does it appropriately answer the bot's prompt?\n"
            "   - score: Overall combined quality score (0-10).\n"
            "4. If meaning is clear even with small grammar slips, 'passed' should be true. The goal is speaking practice.\n"
            "5. Feedback must be concise and constructive (1-2 sentences maximum). Do not overwhelm with grammar rules.\n"
            "6. If a correction is needed, provide a simple, clean, natural sentence. Otherwise leave correction empty.\n\n"
            "You MUST return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "score": integer (0-10),\n'
            '  "communication": integer (0-10),\n'
            '  "grammar": integer (0-10),\n'
            '  "naturalness": integer (0-10),\n'
            '  "relevance": integer (0-10),\n'
            '  "passed": boolean,\n'
            '  "feedback": string,\n'
            '  "correction": string,\n'
            '  "strength": string,\n'
            '  "weakness": string\n'
            "}"
        )

        context_lines = []
        if recent_context:
            for item in recent_context[-3:]:
                context_lines.append(f"Bot: {item.get('bot', '')}")
                context_lines.append(f"Learner: {item.get('learner', '')}")

        context_str = "\n".join(context_lines) if context_lines else "None"

        user_content = (
            f"Recent Context:\n{context_str}\n\n"
            f"Current Bot Prompt: \"{bot_line}\"\n"
            f"Expected Learner Line/Intent: \"{expected_line}\"\n"
            f"Learner Spoke (Transcribed): \"{learner_transcript}\""
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

            return {
                "score": int(data.get("score", 8)),
                "communication": int(data.get("communication", 8)),
                "grammar": int(data.get("grammar", 8)),
                "naturalness": int(data.get("naturalness", 8)),
                "relevance": int(data.get("relevance", 8)),
                "passed": bool(data.get("passed", True)),
                "feedback": str(data.get("feedback", "Good response!")),
                "correction": str(data.get("correction", "")).strip(),
                "strength": str(data.get("strength", "")).strip(),
                "weakness": str(data.get("weakness", "")).strip(),
            }

        except Exception as e:
            # Fallback if chat API temporarily fails during conversation
            is_close = learner_transcript.strip().lower() == expected_line.strip().lower()
            return {
                "score": 9 if is_close else 7,
                "communication": 9 if is_close else 7,
                "grammar": 8,
                "naturalness": 8,
                "relevance": 9,
                "passed": True,
                "feedback": "Your response was understood." if is_close else f"Understood. (Evaluation note: {e})",
                "correction": "" if is_close else expected_line,
                "strength": "Clear message",
                "weakness": "",
            }
