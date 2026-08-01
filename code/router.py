"""router.py
High-level orchestration for routing decisions.

The Router reads messages, retrieves user/message history, applies detectors and scorers,
and emits routing decisions.

Notes:
- Actions are normalized to the HackerRank required set: notify, digest, mute.
- message_type is normalized to allowed values (text, image, audio, video, file, unknown).
"""

import logging
from pathlib import Path
import pandas as pd

from code.retriever import Retriever
from code.scoring import Scoring
from code.vision import Vision
from code.voice import Voice
from code.utils import safe_read_csv, guess_message_type, normalize_message_type
from code.prompts import REASONS


class Router:
    def __init__(self, dataset_dir: Path, n_evidence: int = 3):
        self.logger = logging.getLogger("Router")
        self.dataset_dir = Path(dataset_dir)
        self.n_evidence = n_evidence

        # Load all CSVs in dataset dir
        self.tables = self._load_tables()

        # Components
        self.retriever = Retriever(self.tables)
        self.scoring = Scoring(self.tables)
        self.vision = Vision()
        self.voice = Voice()

        # load messages
        self.messages = safe_read_csv(self.dataset_dir / "messages.csv")

    def _load_tables(self):
        # Read all CSVs in dataset directory into a dict
        tables = {}
        for p in sorted(self.dataset_dir.glob("*.csv")):
            name = p.stem
            tables[name] = safe_read_csv(p)
        return tables

    def process_all(self):
        results = []
        for _, msg in self.messages.iterrows():
            try:
                res = self.process_message(msg)
            except Exception as e:
                self.logger.exception("Failed processing message %s: %s", msg.get("message_id", "?"), e)
                res = {
                    "message_id": msg.get("message_id"),
                    "action": "notify",
                    "message_type": "unknown",
                    "reason": "processing_error",
                    "confidence": 0.0,
                    "evidence_message_ids": "",
                }
            results.append(res)
        return results

    def process_message(self, msg_row):
        message_id = msg_row.get("message_id")
        sender = msg_row.get("sender_id")
        text = (msg_row.get("text") or "")
        media = msg_row.get("media")  # path relative to dataset/media/

        raw_type = guess_message_type(text, media)
        message_type = normalize_message_type(raw_type)

        # retrieve history and candidate evidence
        history = self.retriever.get_user_history(sender)
        evidence = self.retriever.find_evidence_for_text(text, topk=self.n_evidence)

        # features and initial scores
        trust_score = self.scoring.business_trust_score(sender)
        group_priority = self.scoring.group_priority_score(msg_row)

        reason = "unknown"
        final_action = "notify"
        confidence = 0.0

        # Media processing
        media_insights = None
        if message_type == "image":
            media_path = self.dataset_dir / "media" / media
            media_insights = self.vision.analyze_image(media_path)
        elif message_type == "audio":
            media_path = self.dataset_dir / "media" / media
            transcript, t_conf = self.voice.transcribe_audio(media_path)
            text = (text or "") + " " + (transcript or "")

        # rule-based detectors + lightweight ML where applicable
        is_scam, scam_score = self.scoring.detect_scam(text, media_insights)
        is_spam, spam_score = self.scoring.detect_spam(text, media_insights)
        is_promo, promo_score = self.scoring.detect_promotion(text, media_insights)
        in_quiet_hours, quiet_score = self.scoring.detect_quiet_hours(msg_row)
        personalized, personal_score = self.scoring.personalization_score(sender, text)

        # decision logic: combine scores with business trust and group priority
        # Preference order: scam > spam > promotion > quiet-hours > personal
        scores = {
            "scam": scam_score,
            "spam": spam_score,
            "promotion": promo_score,
            "personal": personal_score,
            "quiet": quiet_score,
        }

        # choose reason and map to one of notify/digest/mute
        if is_scam:
            reason = "scam"
            # strong scams -> mute; weaker -> mute to be conservative
            final_action = "mute"
            confidence = self.scoring.combine_confidence(scam_score, trust_score, group_priority)
        elif is_spam:
            reason = "spam"
            final_action = "mute"
            confidence = self.scoring.combine_confidence(spam_score, trust_score, group_priority)
        elif is_promo:
            reason = "promotion"
            final_action = "digest"
            confidence = self.scoring.combine_confidence(promo_score, trust_score, group_priority)
        elif in_quiet_hours and not personalized:
            reason = "quiet_hours"
            # defer non-urgent content into digest
            final_action = "digest"
            confidence = self.scoring.combine_confidence(quiet_score, trust_score, group_priority)
        elif personalized:
            reason = "personal"
            final_action = "notify"
            confidence = self.scoring.combine_confidence(personal_score, trust_score, group_priority)
        else:
            reason = "unknown"
            final_action = "notify"
            confidence = self.scoring.combine_confidence(max(scores.values()), trust_score, group_priority)

        # format evidence ids
        evidence_ids = ";".join([str(int(x)) for x in (evidence or [])])

        return {
            "message_id": int(message_id) if message_id is not None else None,
            "action": final_action,
            "message_type": message_type,
            "reason": reason,
            "confidence": round(float(confidence), 4),
            "evidence_message_ids": evidence_ids,
        }
