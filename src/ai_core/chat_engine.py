"""
Chat Engine - Operations Intelligence

AI-powered COO consultation with operational context.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Generator
import logging

from .claude_client import ClaudeClient

logger = logging.getLogger(__name__)


class ConversationMode(Enum):
    """Conversation modes for COO consultation."""
    GENERAL = "general"
    PROCESS_ANALYSIS = "process_analysis"
    CAPACITY_PLANNING = "capacity_planning"
    KPI_REVIEW = "kpi_review"
    BENCHMARK_DISCUSSION = "benchmark_discussion"
    IMPROVEMENT_PLANNING = "improvement_planning"
    RISK_ASSESSMENT = "risk_assessment"
    RESOURCE_OPTIMIZATION = "resource_optimization"


SYSTEM_PROMPTS = {
    ConversationMode.GENERAL: """You are an AI-powered Chief Operating Officer (COO) consultant specializing in operational excellence.
You help organizations optimize their processes, improve efficiency, reduce costs, and enhance quality.

Your expertise includes:
- Process optimization and lean methodologies
- Capacity planning and resource allocation
- KPI development and performance management
- Operational risk management
- Continuous improvement (Kaizen, Six Sigma)
- Supply chain optimization
- Workforce productivity

Provide practical, actionable advice based on operational data and industry best practices.
Be direct and specific in your recommendations. Use data to support your insights when available.""",

    ConversationMode.PROCESS_ANALYSIS: """You are an expert process analyst with deep knowledge of:
- Value stream mapping
- Process flow analysis
- Bottleneck identification
- Cycle time optimization
- Lean manufacturing principles
- Six Sigma methodologies

Focus on analyzing process efficiency, identifying waste (muda), and recommending improvements.
Use specific metrics like OEE, cycle time, throughput, and first-pass yield in your analysis.""",

    ConversationMode.CAPACITY_PLANNING: """You are a capacity planning specialist helping organizations:
- Forecast demand and plan resources
- Balance workload across resources
- Identify capacity constraints
- Optimize resource utilization
- Plan for seasonal variations
- Make investment decisions

Provide data-driven recommendations for capacity decisions.
Consider both short-term adjustments and long-term strategic planning.""",

    ConversationMode.KPI_REVIEW: """You are a KPI and performance management expert specializing in:
- Operational metrics development
- KPI target setting
- Performance trend analysis
- Balanced scorecard approaches
- Leading vs lagging indicators
- Industry benchmarking

Help interpret KPI data and provide insights on performance improvement opportunities.
Explain the significance of metrics and their interconnections.""",

    ConversationMode.BENCHMARK_DISCUSSION: """You are a benchmarking specialist with knowledge of:
- Industry performance standards
- Best-in-class operational metrics
- Comparative analysis methodologies
- Gap analysis and improvement planning
- World-class manufacturing standards

Help organizations understand how they compare to industry standards and identify improvement priorities.
Reference specific benchmark values and industry norms when relevant.""",

    ConversationMode.IMPROVEMENT_PLANNING: """You are a continuous improvement consultant skilled in:
- Kaizen and continuous improvement
- A3 problem solving
- Root cause analysis
- Project prioritization (impact vs effort)
- Change management
- Implementation roadmaps

Help develop practical improvement plans with clear priorities and measurable outcomes.
Focus on quick wins as well as strategic improvements.""",

    ConversationMode.RISK_ASSESSMENT: """You are an operational risk management expert specializing in:
- Risk identification and assessment
- Business continuity planning
- Supply chain risk management
- Quality risk analysis
- Capacity and demand risk
- Mitigation strategy development

Help identify operational risks and develop mitigation strategies.
Prioritize risks based on likelihood and impact.""",

    ConversationMode.RESOURCE_OPTIMIZATION: """You are a resource optimization specialist focused on:
- Labor planning and scheduling
- Equipment utilization
- Space optimization
- Material flow efficiency
- Cost reduction strategies
- Productivity improvement

Provide recommendations for optimizing resource allocation and reducing operational costs.
Balance efficiency with flexibility and employee satisfaction."""
}


SUGGESTED_PROMPTS = {
    ConversationMode.GENERAL: [
        "What are our biggest operational challenges?",
        "How can we improve overall efficiency?",
        "Where should we focus our improvement efforts?",
    ],
    ConversationMode.PROCESS_ANALYSIS: [
        "Analyze our process efficiency and identify bottlenecks",
        "Where are we experiencing the most waste?",
        "How can we reduce cycle time in our operations?",
    ],
    ConversationMode.CAPACITY_PLANNING: [
        "Do we have enough capacity to meet demand?",
        "Where should we invest in additional capacity?",
        "How should we plan resources for next quarter?",
    ],
    ConversationMode.KPI_REVIEW: [
        "How are we performing against our KPI targets?",
        "Which KPIs need the most attention?",
        "What's driving the change in our performance?",
    ],
    ConversationMode.BENCHMARK_DISCUSSION: [
        "How do we compare to industry benchmarks?",
        "What's our biggest gap vs world-class performance?",
        "Which areas are we performing above average?",
    ],
    ConversationMode.IMPROVEMENT_PLANNING: [
        "What improvements should we prioritize?",
        "Create an improvement roadmap for next quarter",
        "What quick wins can we implement this month?",
    ],
    ConversationMode.RISK_ASSESSMENT: [
        "What are our top operational risks?",
        "How resilient is our supply chain?",
        "What capacity risks should we address?",
    ],
    ConversationMode.RESOURCE_OPTIMIZATION: [
        "How can we improve labor productivity?",
        "Are our resources being used efficiently?",
        "Where can we reduce operational costs?",
    ]
}


class ChatEngine:
    """AI chat engine for COO consultation."""

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        """Initialize chat engine."""
        self.claude = claude_client or ClaudeClient()
        self.conversation_history: List[Dict[str, str]] = []
        self.current_mode = ConversationMode.GENERAL
        self.context_data: Dict[str, Any] = {}

    def set_mode(self, mode: ConversationMode):
        """Set conversation mode."""
        self.current_mode = mode

    def set_context(self, context: Dict[str, Any]):
        """Set operational context for the conversation."""
        self.context_data = context

    def get_suggested_prompts(self) -> List[str]:
        """Get suggested prompts for current mode."""
        return SUGGESTED_PROMPTS.get(self.current_mode, SUGGESTED_PROMPTS[ConversationMode.GENERAL])

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def detect_mode(self, message: str) -> ConversationMode:
        """Detect appropriate conversation mode from message."""
        message_lower = message.lower()

        if any(word in message_lower for word in ['process', 'bottleneck', 'cycle time', 'flow', 'waste']):
            return ConversationMode.PROCESS_ANALYSIS
        elif any(word in message_lower for word in ['capacity', 'demand', 'forecast', 'resource plan']):
            return ConversationMode.CAPACITY_PLANNING
        elif any(word in message_lower for word in ['kpi', 'metric', 'performance', 'target']):
            return ConversationMode.KPI_REVIEW
        elif any(word in message_lower for word in ['benchmark', 'compare', 'industry', 'standard']):
            return ConversationMode.BENCHMARK_DISCUSSION
        elif any(word in message_lower for word in ['improve', 'kaizen', 'roadmap', 'plan']):
            return ConversationMode.IMPROVEMENT_PLANNING
        elif any(word in message_lower for word in ['risk', 'resilience', 'continuity', 'threat']):
            return ConversationMode.RISK_ASSESSMENT
        elif any(word in message_lower for word in ['labor', 'utilization', 'cost', 'productivity', 'optimize']):
            return ConversationMode.RESOURCE_OPTIMIZATION

        return ConversationMode.GENERAL

    def build_context_prompt(self) -> str:
        """Build context injection for the system prompt."""
        if not self.context_data:
            return ""

        context_parts = ["\n\n--- OPERATIONAL CONTEXT ---"]

        # Organization info
        if 'organization' in self.context_data:
            org = self.context_data['organization']
            context_parts.append(f"\nOrganization: {org.get('name', 'Unknown')}")
            context_parts.append(f"Industry: {org.get('industry', 'N/A')}")
            context_parts.append(f"Operation Type: {org.get('operation_type', 'N/A')}")
            context_parts.append(f"Employee Count: {org.get('employee_count', 'N/A')}")

        # Current KPIs
        if 'kpis' in self.context_data:
            kpis = self.context_data['kpis']
            context_parts.append("\nCurrent KPIs:")
            for key, value in kpis.items():
                if value is not None:
                    context_parts.append(f"  - {key}: {value}")

        # Process metrics
        if 'processes' in self.context_data:
            context_parts.append("\nProcess Summary:")
            for process in self.context_data['processes'][:5]:
                context_parts.append(f"  - {process.get('name')}: OEE {process.get('oee', 'N/A')}%")

        # Resource utilization
        if 'resources' in self.context_data:
            context_parts.append("\nResource Utilization:")
            for resource in self.context_data['resources'][:5]:
                context_parts.append(f"  - {resource.get('name')}: {resource.get('utilization', 'N/A')}%")

        # Benchmark results
        if 'benchmark' in self.context_data:
            bench = self.context_data['benchmark']
            context_parts.append(f"\nBenchmark Score: {bench.get('overall_score', 'N/A')} ({bench.get('overall_rating', 'N/A')})")
            if bench.get('strengths'):
                context_parts.append(f"Strengths: {', '.join(bench['strengths'][:3])}")
            if bench.get('improvements'):
                context_parts.append(f"Areas to Improve: {', '.join(bench['improvements'][:3])}")

        context_parts.append("\n--- END CONTEXT ---\n")

        return "\n".join(context_parts)

    def get_system_prompt(self) -> str:
        """Get full system prompt with context."""
        base_prompt = SYSTEM_PROMPTS.get(self.current_mode, SYSTEM_PROMPTS[ConversationMode.GENERAL])
        context_prompt = self.build_context_prompt()
        return base_prompt + context_prompt

    def chat(self, message: str, auto_detect_mode: bool = True) -> str:
        """Send a message and get a response."""
        if auto_detect_mode:
            detected_mode = self.detect_mode(message)
            if detected_mode != self.current_mode:
                self.current_mode = detected_mode

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Get response
        response = self.claude.generate_response(
            messages=self.conversation_history,
            system_prompt=self.get_system_prompt()
        )

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response

    def stream_chat(self, message: str, auto_detect_mode: bool = True) -> Generator[str, None, None]:
        """Send a message and stream the response."""
        if auto_detect_mode:
            detected_mode = self.detect_mode(message)
            if detected_mode != self.current_mode:
                self.current_mode = detected_mode

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Stream response
        full_response = ""
        for token in self.claude.stream_response(
            messages=self.conversation_history,
            system_prompt=self.get_system_prompt()
        ):
            full_response += token
            yield token

        # Add complete response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

        # Keep history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def analyze_with_context(self, analysis_type: str, data: Dict[str, Any]) -> str:
        """Perform analysis with full operational context."""
        # Merge provided data with context
        full_context = {**self.context_data, **data}

        prompt = f"""Based on the operational context and the following specific data, provide a {analysis_type} analysis:

{self._format_analysis_data(data)}

Please provide:
1. Key findings
2. Performance assessment
3. Root causes (if applicable)
4. Specific recommendations
5. Priority actions with expected impact"""

        return self.chat(prompt, auto_detect_mode=False)

    def _format_analysis_data(self, data: Dict[str, Any]) -> str:
        """Format analysis data for prompts."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}:")
                for item in value[:10]:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)


# Factory function
def create_chat_engine(api_key: Optional[str] = None) -> ChatEngine:
    """Create a configured chat engine."""
    client = ClaudeClient(api_key)
    return ChatEngine(client)
