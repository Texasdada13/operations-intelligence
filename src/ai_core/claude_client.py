"""
Claude Client - Operations Intelligence

Wrapper for Anthropic Claude API interactions.
"""

import os
import logging
from typing import Optional, Generator, Dict, Any, List
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for Claude AI API interactions."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude client."""
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("No Anthropic API key found")
            self.client = None
        else:
            self.client = Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return self.client is not None

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7
    ) -> str:
        """Generate a complete response from Claude."""
        if not self.client:
            return "AI assistant is not available. Please configure the Anthropic API key."

        try:
            response = self.client.messages.create(
                model=self.DEFAULT_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature
            )

            if response.content and len(response.content) > 0:
                return response.content[0].text

            return "I apologize, but I couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    def stream_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Stream response from Claude token by token."""
        if not self.client:
            yield "AI assistant is not available. Please configure the Anthropic API key."
            return

        try:
            with self.client.messages.stream(
                model=self.DEFAULT_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            yield f"I encountered an error: {str(e)}"

    def analyze_operations(
        self,
        data: Dict[str, Any],
        analysis_type: str = "general"
    ) -> str:
        """Perform operational analysis using Claude."""
        system_prompt = """You are an expert operations analyst specializing in:
- Process optimization and efficiency
- Capacity planning and resource allocation
- KPI analysis and benchmarking
- Operational risk assessment
- Continuous improvement methodologies

Analyze the provided operational data and provide actionable insights."""

        user_message = f"""Please analyze the following operational data:

Analysis Type: {analysis_type}

Data:
{self._format_data(data)}

Provide:
1. Key observations
2. Performance assessment
3. Identified issues or risks
4. Specific recommendations
5. Priority actions"""

        return self.generate_response(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt
        )

    def _format_data(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Format data for inclusion in prompts."""
        lines = []
        prefix = "  " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_data(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value[:10]:  # Limit list items
                    if isinstance(item, dict):
                        lines.append(self._format_data(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines)
