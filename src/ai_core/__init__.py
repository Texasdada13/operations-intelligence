"""
Operations Intelligence - AI Core Module

Claude AI integration for COO consultation.
"""

from .chat_engine import ChatEngine, ConversationMode
from .claude_client import ClaudeClient

__all__ = ['ChatEngine', 'ConversationMode', 'ClaudeClient']
