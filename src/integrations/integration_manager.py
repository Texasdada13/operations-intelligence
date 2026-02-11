"""
Integration Manager for Operations Intelligence

Provides a unified interface for managing DevOps integrations.
Supports Jira and Azure DevOps with extensibility for additional platforms.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .jira_client import JiraClient, JiraConfig, JiraDemoClient
from .azure_devops_client import AzureDevOpsClient, AzureDevOpsConfig, AzureDevOpsDemoClient

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    JIRA = "jira"
    AZURE_DEVOPS = "azure_devops"
    GITHUB = "github"  # Future
    DEMO = "demo"


@dataclass
class IntegrationStatus:
    """Status of an integration connection"""
    integration_type: IntegrationType
    is_connected: bool
    last_sync: Optional[datetime] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    error_message: Optional[str] = None


class OperationsIntegrationManager:
    """
    Manages DevOps integrations for Operations Intelligence.

    Provides a unified interface for:
    - API authentication
    - Data fetching from multiple sources
    - Normalized data formats
    - Combined metrics and reports
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(__file__), '..', '..', 'instance', 'integrations'
        )
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

        self._jira_client: Optional[JiraClient] = None
        self._azure_client: Optional[AzureDevOpsClient] = None
        self._demo_mode = False

    # ========== Configuration ==========

    def configure_jira(self, config: Optional[JiraConfig] = None):
        """Configure Jira integration"""
        config = config or JiraConfig.from_env()
        if config.domain and config.email and config.api_token:
            self._jira_client = JiraClient(config)

    def configure_azure_devops(self, config: Optional[AzureDevOpsConfig] = None):
        """Configure Azure DevOps integration"""
        config = config or AzureDevOpsConfig.from_env()
        if config.organization and config.project and config.personal_access_token:
            self._azure_client = AzureDevOpsClient(config)

    def enable_demo_mode(self):
        """Enable demo mode with mock data"""
        self._demo_mode = True
        self._jira_client = JiraDemoClient()
        self._azure_client = AzureDevOpsDemoClient()

    # ========== Status ==========

    def get_status(self, integration_type: IntegrationType) -> IntegrationStatus:
        """Get the status of an integration"""
        is_connected = False
        organization = None
        project = None
        error_message = None

        try:
            if integration_type == IntegrationType.JIRA and self._jira_client:
                # Test connection by fetching projects
                try:
                    projects = self._jira_client.get_projects()
                    is_connected = True
                    organization = self._jira_client.config.domain
                except Exception as e:
                    error_message = str(e)

            elif integration_type == IntegrationType.AZURE_DEVOPS and self._azure_client:
                try:
                    # Test connection
                    self._azure_client.get_iterations()
                    is_connected = True
                    organization = self._azure_client.config.organization
                    project = self._azure_client.config.project
                except Exception as e:
                    error_message = str(e)

            elif integration_type == IntegrationType.DEMO:
                is_connected = self._demo_mode
                organization = "Demo Organization"

        except Exception as e:
            error_message = str(e)

        return IntegrationStatus(
            integration_type=integration_type,
            is_connected=is_connected,
            organization=organization,
            project=project,
            error_message=error_message
        )

    def get_all_statuses(self) -> List[IntegrationStatus]:
        """Get status for all configured integrations"""
        statuses = []

        if self._jira_client:
            statuses.append(self.get_status(IntegrationType.JIRA))
        if self._azure_client:
            statuses.append(self.get_status(IntegrationType.AZURE_DEVOPS))
        if self._demo_mode:
            statuses.append(self.get_status(IntegrationType.DEMO))

        return statuses

    def disconnect(self, integration_type: IntegrationType):
        """Disconnect an integration"""
        if integration_type == IntegrationType.JIRA:
            self._jira_client = None
        elif integration_type == IntegrationType.AZURE_DEVOPS:
            self._azure_client = None

    # ========== Jira Data Access ==========

    def get_jira_summary(self, project_key: str) -> Optional[Dict[str, Any]]:
        """Get Jira operations summary"""
        if not self._jira_client:
            return None

        try:
            return self._jira_client.get_operations_summary(project_key)
        except Exception as e:
            logger.error(f"Error fetching Jira summary: {e}")
            return {'error': str(e)}

    def get_jira_projects(self) -> Optional[List[Dict[str, Any]]]:
        """Get list of Jira projects"""
        if not self._jira_client:
            return None

        try:
            projects = self._jira_client.get_projects()
            return [
                {
                    'key': p.key,
                    'name': p.name,
                    'type': p.project_type,
                    'lead': p.lead
                }
                for p in projects
            ]
        except Exception as e:
            logger.error(f"Error fetching Jira projects: {e}")
            return None

    def get_jira_velocity(self, project_key: str, num_sprints: int = 6) -> Optional[List[Dict[str, Any]]]:
        """Get Jira velocity trend"""
        if not self._jira_client:
            return None

        try:
            boards = self._jira_client.get_boards(project_key)
            scrum_board = next((b for b in boards if b.get('type') == 'scrum'), None)
            if not scrum_board:
                return []

            metrics = self._jira_client.get_velocity_trend(scrum_board['id'], num_sprints)
            return [
                {
                    'sprint': m.sprint_name,
                    'committed': m.committed_points,
                    'completed': m.completed_points,
                    'completion_rate': m.completion_rate,
                    'avg_cycle_time': m.avg_cycle_time
                }
                for m in metrics
            ]
        except Exception as e:
            logger.error(f"Error fetching Jira velocity: {e}")
            return None

    # ========== Azure DevOps Data Access ==========

    def get_azure_summary(self) -> Optional[Dict[str, Any]]:
        """Get Azure DevOps operations summary"""
        if not self._azure_client:
            return None

        try:
            return self._azure_client.get_operations_summary()
        except Exception as e:
            logger.error(f"Error fetching Azure DevOps summary: {e}")
            return {'error': str(e)}

    def get_azure_builds(self, days: int = 30) -> Optional[Dict[str, Any]]:
        """Get Azure DevOps build metrics"""
        if not self._azure_client:
            return None

        try:
            return self._azure_client.get_build_metrics(days)
        except Exception as e:
            logger.error(f"Error fetching Azure DevOps builds: {e}")
            return None

    # ========== Combined Operations Summary ==========

    def get_unified_operations_summary(self) -> Dict[str, Any]:
        """Get combined operations summary from all connected sources"""
        summary = {
            'sources': [],
            'work_items': {
                'total_open': 0,
                'completed_last_30_days': 0,
                'by_status': {},
                'by_type': {}
            },
            'performance': {
                'avg_cycle_time_days': 0,
                'velocity_points': 0
            },
            'team': {
                'workload': []
            },
            'builds': {
                'total': 0,
                'success_rate': 0
            },
            'alerts': []
        }

        cycle_times = []
        workload_combined = {}

        # Aggregate Jira data
        if self._jira_client:
            try:
                projects = self._jira_client.get_projects()
                for project in projects[:3]:  # Limit to first 3 projects
                    jira_summary = self._jira_client.get_operations_summary(project.key)
                    if jira_summary:
                        summary['sources'].append({
                            'type': 'jira',
                            'project': project.name
                        })

                        # Aggregate work items
                        summary['work_items']['total_open'] += jira_summary['issues']['open']
                        summary['work_items']['completed_last_30_days'] += jira_summary['issues']['resolved_last_30_days']

                        for status, count in jira_summary['issues']['by_status'].items():
                            summary['work_items']['by_status'][status] = summary['work_items']['by_status'].get(status, 0) + count

                        for item_type, count in jira_summary['issues']['by_type'].items():
                            summary['work_items']['by_type'][item_type] = summary['work_items']['by_type'].get(item_type, 0) + count

                        # Performance
                        if jira_summary['performance']['avg_cycle_time_days']:
                            cycle_times.append(jira_summary['performance']['avg_cycle_time_days'])

                        # Team workload
                        for member in jira_summary['team']['workload']:
                            name = member['name']
                            workload_combined[name] = workload_combined.get(name, 0) + member['issues']

                        # Alerts
                        summary['alerts'].extend(jira_summary.get('alerts', []))

            except Exception as e:
                logger.error(f"Error aggregating Jira data: {e}")

        # Aggregate Azure DevOps data
        if self._azure_client:
            try:
                azure_summary = self._azure_client.get_operations_summary()
                if azure_summary and 'error' not in azure_summary:
                    summary['sources'].append({
                        'type': 'azure_devops',
                        'project': azure_summary['project']
                    })

                    # Aggregate work items
                    summary['work_items']['total_open'] += azure_summary['work_items']['open']
                    summary['work_items']['completed_last_30_days'] += azure_summary['work_items']['completed_last_30_days']

                    for status, count in azure_summary['work_items']['by_status'].items():
                        summary['work_items']['by_status'][status] = summary['work_items']['by_status'].get(status, 0) + count

                    for item_type, count in azure_summary['work_items']['by_type'].items():
                        summary['work_items']['by_type'][item_type] = summary['work_items']['by_type'].get(item_type, 0) + count

                    # Performance
                    if azure_summary['performance']['avg_cycle_time_days']:
                        cycle_times.append(azure_summary['performance']['avg_cycle_time_days'])
                    summary['performance']['velocity_points'] += azure_summary['performance']['velocity_points']

                    # Team workload
                    for member in azure_summary['team']['workload']:
                        name = member['name']
                        workload_combined[name] = workload_combined.get(name, 0) + member['items']

                    # Builds
                    summary['builds'] = azure_summary['builds']

                    # Alerts
                    summary['alerts'].extend(azure_summary.get('alerts', []))

            except Exception as e:
                logger.error(f"Error aggregating Azure DevOps data: {e}")

        # Calculate averages
        if cycle_times:
            summary['performance']['avg_cycle_time_days'] = round(sum(cycle_times) / len(cycle_times), 1)

        # Format workload
        summary['team']['workload'] = sorted(
            [{'name': k, 'items': v} for k, v in workload_combined.items()],
            key=lambda x: x['items'],
            reverse=True
        )[:15]

        return summary


# Factory function
def create_operations_integration_manager(demo_mode: bool = False) -> OperationsIntegrationManager:
    """Create an OperationsIntegrationManager with appropriate configuration"""
    manager = OperationsIntegrationManager()

    if demo_mode:
        manager.enable_demo_mode()
    else:
        # Configure from environment variables if available
        if os.getenv('JIRA_DOMAIN'):
            manager.configure_jira()
        if os.getenv('AZURE_DEVOPS_ORG'):
            manager.configure_azure_devops()

    return manager
