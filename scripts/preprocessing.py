"""
Text Preprocessing, Normalization & Spam Detection Module
Real-Time Social Listening & Brand Sentiment Pipeline

Features:
- Regex-based URL, mention, and hashtag extraction/normalization
- Emoji translation / demojization (e.g., :fire: -> 'fire')
- Language detection with fallback mechanisms
- Heuristic-based spam & bot message filtering
- Pydantic models for type safety and validation
"""

import re
import html
import unicodedata
from typing import Optional, Tuple, Dict, Any, List
from pydantic import BaseModel, Field
import emoji
from langdetect import detect, DetectorFactory

# Enforce deterministic results for langdetect
DetectorFactory.seed = 0

from scripts.utils import setup_logger

logger = setup_logger("preprocessing")


# -----------------------------------------------------------------------------
# Data Schemas
# -----------------------------------------------------------------------------
class RawSocialPost(BaseModel):
    """Unified inbound raw post schema."""
    post_id: str
    platform: str
    author_id: str
    author_handle: Optional[str] = None
    text: str
    created_at: str
    url: Optional[str] = None
    reply_to_id: Optional[str] = None
    reply_to_user: Optional[str] = None
    engagement: Dict[str, Any] = Field(default_factory=dict)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class CleanedSocialPost(BaseModel):
    """Enriched post after text normalization, language detection, and spam filtering."""
    post_id: str
    platform: str
    author_id: str
    author_handle: Optional[str]
    raw_text: str
    clean_text: str
    language: str
    is_spam: bool
    spam_reason: Optional[str]
    created_at: str
    url: Optional[str]
    reply_to_id: Optional[str]
    reply_to_user: Optional[str]
    engagement: Dict[str, Any]


# -----------------------------------------------------------------------------
# Text Preprocessor Class
# -----------------------------------------------------------------------------
class SocialTextPreprocessor:
    """Production-grade text sanitizer and quality filter for social streams."""

    # Regex patterns
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    MENTION_PATTERN = re.compile(r"@[\w_.-]+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{3,}")  # e.g., 'soooooooo' -> 'soo'
    MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
    SPECIAL_SYMBOLS_PATTERN = re.compile(r"[^\w\s.,!?'\"-]")

    # Known spam trigger phrases (crypto pumps, bot promotions, free giveaways)
    SPAM_TRIGGERS = [
        "airdrop", "free crypto", "giveaway alert", "dm to buy", "earn $",
        "whatsapp me", "telegram channel", "100x gem", "guaranteed profit",
        "work from home", "passive income", "click here to claim"
    ]

    def __init__(self, replace_emoji: bool = True, supported_languages: Optional[List[str]] = None):
        self.replace_emoji = replace_emoji
        self.supported_languages = supported_languages or ["en"]

    def clean_text(self, text: str) -> str:
        """Applies comprehensive text normalization pipeline."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Unescape HTML entities (&amp; -> &, &lt; -> <)
        text = html.unescape(text)

        # 2. Unicode normalization (NFKC decomposes compatibility chars)
        text = unicodedata.normalize("NFKC", text)

        # 3. Handle emojis (convert to descriptive textual tokens)
        if self.replace_emoji:
            text = emoji.demojize(text, delimiters=(" ", " "))
            text = text.replace("_", " ")

        # 4. Remove URLs
        text = self.URL_PATTERN.sub("", text)

        # 5. Remove Email addresses
        text = self.EMAIL_PATTERN.sub("", text)

        # 6. Normalize Mentions to standard token
        text = self.MENTION_PATTERN.sub("@user", text)

        # 7. Unpack Hashtags (e.g. #ApexCloud -> ApexCloud)
        text = self.HASHTAG_PATTERN.sub(r"\1", text)

        # 8. Compress repeated characters (e.g. 'terribleeeeee' -> 'terrible')
        text = self.REPEATED_CHARS_PATTERN.sub(r"\1\1", text)

        # 9. Clean excessive non-standard symbols
        text = self.SPECIAL_SYMBOLS_PATTERN.sub(" ", text)

        # 10. Collapse redundant whitespace
        text = self.MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

        return text

    def detect_language(self, text: str) -> str:
        """Detects language with safety fallbacks for short texts."""
        clean_sample = self.clean_text(text)
        if len(clean_sample.split()) < 3:
            return "en"  # Default assumption for ultra-short brand mentions

        try:
            detected_lang = detect(clean_sample)
            return detected_lang
        except Exception:
            return "unknown"

    def evaluate_spam(self, raw_text: str, clean_text: str) -> Tuple[bool, Optional[str]]:
        """Evaluates heuristic spam risk based on structural and semantic triggers."""
        # 1. Length bounds
        words = clean_text.split()
        if len(words) < 2:
            return True, "TOO_SHORT"
        if len(raw_text) > 3000:
            return True, "EXCESSIVE_LENGTH"

        # 2. URL density ratio
        urls = self.URL_PATTERN.findall(raw_text)
        if len(urls) >= 3:
            return True, "EXCESSIVE_LINKS"

        # 3. Uppercase shouting ratio (e.g., BUY NOW SCAM FREE FREE)
        alpha_chars = [c for c in raw_text if c.isalpha()]
        if len(alpha_chars) > 20:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.75:
                return True, "SHOUTING_RATIO"

        # 4. Repeated character flood (e.g., 'aaaaaaaaaaaaaaaaaaaa')
        if len(raw_text) > 20:
            most_common_char_count = max(raw_text.count(c) for c in set(raw_text))
            if most_common_char_count / len(raw_text) > 0.45:
                return True, "CHARACTER_FLOODING"

        # 5. Spam phrase matching
        lower_raw = raw_text.lower()
        for phrase in self.SPAM_TRIGGERS:
            if phrase in lower_raw:
                return True, f"SPAM_KEYWORD:{phrase}"

        return False, None

    def process(self, raw_post: RawSocialPost) -> CleanedSocialPost:
        """Executes full normalization, language filtering, and spam grading."""
        cleaned = self.clean_text(raw_post.text)
        lang = self.detect_language(raw_post.text)
        is_spam, spam_reason = self.evaluate_spam(raw_post.text, cleaned)

        # Flag unsupported languages as spam/noise for sentiment inference
        if lang not in self.supported_languages and lang != "unknown":
            is_spam = True
            spam_reason = f"UNSUPPORTED_LANGUAGE:{lang}"

        return CleanedSocialPost(
            post_id=raw_post.post_id,
            platform=raw_post.platform,
            author_id=raw_post.author_id,
            author_handle=raw_post.author_handle,
            raw_text=raw_post.text,
            clean_text=cleaned,
            language=lang,
            is_spam=is_spam,
            spam_reason=spam_reason,
            created_at=raw_post.created_at,
            url=raw_post.url,
            reply_to_id=raw_post.reply_to_id,
            reply_to_user=raw_post.reply_to_user,
            engagement=raw_post.engagement,
        )


# Global preprocessor instance
default_preprocessor = SocialTextPreprocessor()


def preprocess_post(raw_dict: Dict[str, Any]) -> CleanedSocialPost:
    """Convenience functional wrapper for processing dictionary inputs."""
    post_obj = RawSocialPost(**raw_dict)
    return default_preprocessor.process(post_obj)
