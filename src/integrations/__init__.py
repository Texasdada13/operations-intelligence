"""
External Data Integrations for Operations Intelligence

Provides connections to:
- Jira (Atlassian)
- Azure DevOps
- GitHub (future)
"""

from .jira_client import JiraClient, JiraConfig
from .azure_devops_client import AzureDevOpsClient, AzureDevOpsConfig
from .integration_manager import OperationsIntegrationManager, IntegrationType

__all__ = [
    'JiraClient',
    'JiraConfig',
    'AzureDevOpsClient',
    'AzureDevOpsConfig',
    'OperationsIntegrationManager',
    'IntegrationType'
]
