"""
Azure DevOps Integration Client for Operations Intelligence

Provides API access to Azure DevOps for:
- Work item tracking
- Sprint/iteration management
- Build and release pipelines
- Repository analytics
- Team velocity metrics
"""

import os
import logging
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import requests

logger = logging.getLogger(__name__)


@dataclass
class AzureDevOpsConfig:
    """Configuration for Azure DevOps API access"""
    organization: str
    project: str
    personal_access_token: str

    @property
    def base_url(self) -> str:
        return f"https://dev.azure.com/{self.organization}/{self.project}/_apis"

    @property
    def analytics_url(self) -> str:
        return f"https://analytics.dev.azure.com/{self.organization}/{self.project}/_odata/v3.0"

    @classmethod
    def from_env(cls) -> 'AzureDevOpsConfig':
        """Create config from environment variables"""
        return cls(
            organization=os.getenv('AZURE_DEVOPS_ORG', ''),
            project=os.getenv('AZURE_DEVOPS_PROJECT', ''),
            personal_access_token=os.getenv('AZURE_DEVOPS_PAT', '')
        )


class WorkItemType(Enum):
    """Azure DevOps work item types"""
    EPIC = "Epic"
    FEATURE = "Feature"
    USER_STORY = "User Story"
    TASK = "Task"
    BUG = "Bug"
    ISSUE = "Issue"


@dataclass
class WorkItem:
    """Azure DevOps work item"""
    id: int
    title: str
    work_item_type: str
    state: str
    assigned_to: Optional[str]
    created_date: datetime
    changed_date: datetime
    closed_date: Optional[datetime]
    story_points: Optional[float]
    priority: int
    iteration_path: Optional[str]
    area_path: Optional[str]
    tags: List[str] = field(default_factory=list)

    @property
    def cycle_time_days(self) -> Optional[float]:
        """Calculate cycle time if closed"""
        if self.closed_date:
            return (self.closed_date - self.created_date).total_seconds() / 86400
        return None


@dataclass
class Iteration:
    """Azure DevOps iteration/sprint"""
    id: str
    name: str
    path: str
    start_date: Optional[datetime]
    finish_date: Optional[datetime]
    time_frame: str  # past, current, future


@dataclass
class BuildDefinition:
    """Azure DevOps build pipeline"""
    id: int
    name: str
    path: str
    queue_status: str
    created_date: datetime


@dataclass
class Build:
    """Azure DevOps build run"""
    id: int
    build_number: str
    status: str
    result: Optional[str]
    start_time: datetime
    finish_time: Optional[datetime]
    source_branch: str
    definition_name: str
    requested_by: str

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.finish_time and self.start_time:
            return (self.finish_time - self.start_time).total_seconds() / 60
        return None


@dataclass
class ReleaseMetrics:
    """Release pipeline metrics"""
    total_deployments: int
    successful_deployments: int
    failed_deployments: int
    success_rate: float
    avg_deployment_time: float


class AzureDevOpsClient:
    """
    Azure DevOps API Client

    Provides methods for fetching DevOps data
    relevant to operations intelligence.
    """

    API_VERSION = "7.0"

    def __init__(self, config: AzureDevOpsConfig):
        self.config = config
        self._session = requests.Session()
        self._setup_auth()

    def _setup_auth(self):
        """Setup basic auth with PAT"""
        auth_string = f":{self.config.personal_access_token}"
        encoded = base64.b64encode(auth_string.encode()).decode()
        self._session.headers.update({
            'Authorization': f'Basic {encoded}',
            'Content-Type': 'application/json'
        })

    def _get(self, endpoint: str, params: Dict = None, use_analytics: bool = False) -> Dict[str, Any]:
        """GET request to Azure DevOps API"""
        base = self.config.analytics_url if use_analytics else self.config.base_url

        if params is None:
            params = {}
        if not use_analytics:
            params['api-version'] = self.API_VERSION

        url = f"{base}/{endpoint}"
        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """POST request to Azure DevOps API"""
        url = f"{self.config.base_url}/{endpoint}"
        params = {'api-version': self.API_VERSION}
        response = self._session.post(url, json=data, params=params)
        response.raise_for_status()
        return response.json()

    # ========== Work Items ==========

    def get_work_items(self, ids: List[int]) -> List[WorkItem]:
        """Get work items by IDs"""
        if not ids:
            return []

        data = self._get('wit/workitems', params={
            'ids': ','.join(map(str, ids)),
            '$expand': 'all'
        })

        items = []
        for wi in data.get('value', []):
            fields = wi.get('fields', {})

            created_date = datetime.fromisoformat(fields.get('System.CreatedDate', '').replace('Z', '+00:00'))
            changed_date = datetime.fromisoformat(fields.get('System.ChangedDate', '').replace('Z', '+00:00'))
            closed_date = None
            if fields.get('Microsoft.VSTS.Common.ClosedDate'):
                closed_date = datetime.fromisoformat(fields['Microsoft.VSTS.Common.ClosedDate'].replace('Z', '+00:00'))

            items.append(WorkItem(
                id=wi['id'],
                title=fields.get('System.Title', ''),
                work_item_type=fields.get('System.WorkItemType', ''),
                state=fields.get('System.State', ''),
                assigned_to=fields.get('System.AssignedTo', {}).get('displayName') if fields.get('System.AssignedTo') else None,
                created_date=created_date,
                changed_date=changed_date,
                closed_date=closed_date,
                story_points=fields.get('Microsoft.VSTS.Scheduling.StoryPoints'),
                priority=fields.get('Microsoft.VSTS.Common.Priority', 2),
                iteration_path=fields.get('System.IterationPath'),
                area_path=fields.get('System.AreaPath'),
                tags=fields.get('System.Tags', '').split('; ') if fields.get('System.Tags') else []
            ))

        return items

    def query_work_items(self, wiql: str) -> List[WorkItem]:
        """Query work items using WIQL"""
        data = self._post('wit/wiql', {'query': wiql})

        # Get IDs from query result
        ids = [wi['id'] for wi in data.get('workItems', [])]

        if not ids:
            return []

        # Fetch full work items (max 200 at a time)
        all_items = []
        for i in range(0, len(ids), 200):
            batch_ids = ids[i:i+200]
            all_items.extend(self.get_work_items(batch_ids))

        return all_items

    def get_open_work_items(self) -> List[WorkItem]:
        """Get all open work items"""
        wiql = f"""
        SELECT [System.Id]
        FROM WorkItems
        WHERE [System.TeamProject] = '{self.config.project}'
        AND [System.State] NOT IN ('Done', 'Closed', 'Removed', 'Resolved')
        ORDER BY [Microsoft.VSTS.Common.Priority] ASC
        """
        return self.query_work_items(wiql)

    def get_recent_work_items(self, days: int = 30) -> List[WorkItem]:
        """Get work items created in the last N days"""
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        wiql = f"""
        SELECT [System.Id]
        FROM WorkItems
        WHERE [System.TeamProject] = '{self.config.project}'
        AND [System.CreatedDate] >= '{start_date}'
        ORDER BY [System.CreatedDate] DESC
        """
        return self.query_work_items(wiql)

    def get_completed_work_items(self, days: int = 30) -> List[WorkItem]:
        """Get work items completed in the last N days"""
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        wiql = f"""
        SELECT [System.Id]
        FROM WorkItems
        WHERE [System.TeamProject] = '{self.config.project}'
        AND [System.State] IN ('Done', 'Closed', 'Resolved')
        AND [Microsoft.VSTS.Common.ClosedDate] >= '{start_date}'
        ORDER BY [Microsoft.VSTS.Common.ClosedDate] DESC
        """
        return self.query_work_items(wiql)

    # ========== Iterations ==========

    def get_iterations(self, time_frame: str = None) -> List[Iteration]:
        """Get project iterations"""
        data = self._get(f'work/teamsettings/iterations')

        iterations = []
        for it in data.get('value', []):
            start_date = None
            finish_date = None
            attrs = it.get('attributes', {})

            if attrs.get('startDate'):
                start_date = datetime.fromisoformat(attrs['startDate'].replace('Z', '+00:00'))
            if attrs.get('finishDate'):
                finish_date = datetime.fromisoformat(attrs['finishDate'].replace('Z', '+00:00'))

            iteration = Iteration(
                id=it['id'],
                name=it['name'],
                path=it['path'],
                start_date=start_date,
                finish_date=finish_date,
                time_frame=attrs.get('timeFrame', 'current')
            )

            if time_frame is None or iteration.time_frame == time_frame:
                iterations.append(iteration)

        return iterations

    def get_current_iteration(self) -> Optional[Iteration]:
        """Get the current iteration"""
        iterations = self.get_iterations(time_frame='current')
        return iterations[0] if iterations else None

    # ========== Builds ==========

    def get_build_definitions(self) -> List[BuildDefinition]:
        """Get build pipeline definitions"""
        data = self._get('build/definitions')

        definitions = []
        for d in data.get('value', []):
            definitions.append(BuildDefinition(
                id=d['id'],
                name=d['name'],
                path=d.get('path', '\\'),
                queue_status=d.get('queueStatus', 'enabled'),
                created_date=datetime.fromisoformat(d['createdDate'].replace('Z', '+00:00'))
            ))

        return definitions

    def get_builds(self, definition_id: int = None, days: int = 30,
                   top: int = 50) -> List[Build]:
        """Get build runs"""
        params = {
            'maxBuildsPerDefinition': top,
            'queryOrder': 'startTimeDescending'
        }

        if definition_id:
            params['definitions'] = str(definition_id)

        # Filter by date
        min_time = (datetime.utcnow() - timedelta(days=days)).isoformat() + 'Z'
        params['minTime'] = min_time

        data = self._get('build/builds', params=params)

        builds = []
        for b in data.get('value', []):
            start_time = datetime.fromisoformat(b['startTime'].replace('Z', '+00:00')) if b.get('startTime') else datetime.utcnow()
            finish_time = None
            if b.get('finishTime'):
                finish_time = datetime.fromisoformat(b['finishTime'].replace('Z', '+00:00'))

            builds.append(Build(
                id=b['id'],
                build_number=b['buildNumber'],
                status=b['status'],
                result=b.get('result'),
                start_time=start_time,
                finish_time=finish_time,
                source_branch=b.get('sourceBranch', ''),
                definition_name=b.get('definition', {}).get('name', ''),
                requested_by=b.get('requestedBy', {}).get('displayName', 'Unknown')
            ))

        return builds

    def get_build_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get build pipeline metrics"""
        builds = self.get_builds(days=days, top=100)

        completed = [b for b in builds if b.result]
        succeeded = [b for b in completed if b.result == 'succeeded']
        failed = [b for b in completed if b.result in ['failed', 'canceled']]

        # Average build duration
        durations = [b.duration_minutes for b in completed if b.duration_minutes]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            'total_builds': len(builds),
            'successful': len(succeeded),
            'failed': len(failed),
            'success_rate': (len(succeeded) / len(completed) * 100) if completed else 0,
            'avg_duration_minutes': round(avg_duration, 1),
            'by_definition': self._group_builds_by_definition(builds)
        }

    def _group_builds_by_definition(self, builds: List[Build]) -> List[Dict[str, Any]]:
        """Group build metrics by definition"""
        groups = {}
        for b in builds:
            name = b.definition_name
            if name not in groups:
                groups[name] = {'name': name, 'total': 0, 'succeeded': 0, 'failed': 0}
            groups[name]['total'] += 1
            if b.result == 'succeeded':
                groups[name]['succeeded'] += 1
            elif b.result in ['failed', 'canceled']:
                groups[name]['failed'] += 1

        # Calculate success rates
        for g in groups.values():
            completed = g['succeeded'] + g['failed']
            g['success_rate'] = (g['succeeded'] / completed * 100) if completed > 0 else 0

        return sorted(groups.values(), key=lambda x: x['total'], reverse=True)

    # ========== Operations Summary ==========

    def get_operations_summary(self) -> Dict[str, Any]:
        """Get comprehensive DevOps operations summary"""
        # Work items
        open_items = self.get_open_work_items()
        completed_items = self.get_completed_work_items(days=30)

        # Status breakdown
        status_counts = {}
        type_counts = {}
        workload = {}

        for item in open_items:
            status_counts[item.state] = status_counts.get(item.state, 0) + 1
            type_counts[item.work_item_type] = type_counts.get(item.work_item_type, 0) + 1
            assignee = item.assigned_to or 'Unassigned'
            workload[assignee] = workload.get(assignee, 0) + 1

        # Cycle time
        cycle_times = [i.cycle_time_days for i in completed_items if i.cycle_time_days]
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0

        # Build metrics
        build_metrics = self.get_build_metrics(days=30)

        # Current iteration
        current_iter = self.get_current_iteration()

        return {
            'project': self.config.project,
            'work_items': {
                'open': len(open_items),
                'completed_last_30_days': len(completed_items),
                'by_status': status_counts,
                'by_type': type_counts
            },
            'performance': {
                'avg_cycle_time_days': round(avg_cycle_time, 1),
                'velocity_points': sum(i.story_points or 0 for i in completed_items)
            },
            'team': {
                'workload': sorted(
                    [{'name': k, 'items': v} for k, v in workload.items()],
                    key=lambda x: x['items'],
                    reverse=True
                )[:10]
            },
            'builds': build_metrics,
            'current_iteration': {
                'name': current_iter.name if current_iter else None,
                'start': current_iter.start_date.isoformat() if current_iter and current_iter.start_date else None,
                'end': current_iter.finish_date.isoformat() if current_iter and current_iter.finish_date else None
            } if current_iter else None,
            'alerts': self._generate_alerts(open_items, build_metrics, avg_cycle_time)
        }

    def _generate_alerts(self, open_items: List[WorkItem], build_metrics: Dict,
                        avg_cycle_time: float) -> List[Dict[str, Any]]:
        """Generate DevOps alerts"""
        alerts = []

        # Low build success rate
        if build_metrics.get('success_rate', 100) < 80:
            alerts.append({
                'type': 'warning',
                'category': 'Builds',
                'message': f"Build success rate is {build_metrics['success_rate']:.1f}%",
                'recommendation': 'Review failing builds and fix pipeline issues'
            })

        # High cycle time
        if avg_cycle_time > 14:
            alerts.append({
                'type': 'warning',
                'category': 'Cycle Time',
                'message': f'Average cycle time is {avg_cycle_time:.1f} days',
                'recommendation': 'Review workflow and identify bottlenecks'
            })

        # Too many open bugs
        bugs = [i for i in open_items if i.work_item_type.lower() == 'bug']
        if len(bugs) > 20:
            alerts.append({
                'type': 'warning',
                'category': 'Quality',
                'message': f'{len(bugs)} open bugs in backlog',
                'recommendation': 'Prioritize bug fixes to reduce technical debt'
            })

        # High priority items aging
        high_priority_old = [
            i for i in open_items
            if i.priority <= 2 and (datetime.utcnow() - i.created_date.replace(tzinfo=None)).days > 14
        ]
        if high_priority_old:
            alerts.append({
                'type': 'info',
                'category': 'Priorities',
                'message': f'{len(high_priority_old)} high-priority items are over 2 weeks old',
                'recommendation': 'Review aging high-priority items'
            })

        return alerts


# Demo mode for testing
class AzureDevOpsDemoClient(AzureDevOpsClient):
    """Demo client with mock data for testing"""

    def __init__(self):
        config = AzureDevOpsConfig(organization='demo-org', project='Demo Project', personal_access_token='demo')
        super().__init__(config)

    def _get(self, endpoint: str, params: Dict = None, use_analytics: bool = False) -> Dict[str, Any]:
        """Return mock data"""
        return self._get_mock_data(endpoint)

    def _post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """Return mock data for queries"""
        return self._mock_query_result()

    def _get_mock_data(self, endpoint: str) -> Dict[str, Any]:
        """Generate mock Azure DevOps data"""
        import random

        if 'wit/workitems' in endpoint:
            return self._mock_work_items()
        elif 'iterations' in endpoint:
            return self._mock_iterations()
        elif 'build/definitions' in endpoint:
            return self._mock_build_definitions()
        elif 'build/builds' in endpoint:
            return self._mock_builds()

        return {'value': []}

    def _mock_query_result(self) -> Dict[str, Any]:
        """Mock WIQL query result"""
        return {'workItems': [{'id': i} for i in range(1, 31)]}

    def _mock_work_items(self) -> Dict[str, Any]:
        """Generate mock work items"""
        import random

        states = ['New', 'Active', 'In Progress', 'In Review', 'Done', 'Closed']
        types = ['User Story', 'Bug', 'Task', 'Feature', 'Epic']
        assignees = ['Alice Johnson', 'Bob Smith', 'Carol Williams', 'David Brown', None]

        items = []
        for i in range(1, 31):
            days_ago = random.randint(1, 60)
            created = datetime.utcnow() - timedelta(days=days_ago)
            closed = None
            state = random.choice(states)
            if state in ['Done', 'Closed']:
                closed = created + timedelta(days=random.randint(1, 14))

            items.append({
                'id': i,
                'fields': {
                    'System.Title': f'Work Item {i}: {random.choice(["Implement", "Fix", "Update", "Review"])} something',
                    'System.WorkItemType': random.choice(types),
                    'System.State': state,
                    'System.AssignedTo': {'displayName': random.choice(assignees)} if random.choice(assignees) else None,
                    'System.CreatedDate': created.isoformat() + 'Z',
                    'System.ChangedDate': (created + timedelta(days=random.randint(0, days_ago))).isoformat() + 'Z',
                    'Microsoft.VSTS.Common.ClosedDate': closed.isoformat() + 'Z' if closed else None,
                    'Microsoft.VSTS.Scheduling.StoryPoints': random.choice([1, 2, 3, 5, 8, 13, None]),
                    'Microsoft.VSTS.Common.Priority': random.randint(1, 4),
                    'System.IterationPath': 'Demo Project\\Sprint 20',
                    'System.AreaPath': 'Demo Project\\Backend',
                    'System.Tags': '; '.join(random.sample(['urgent', 'tech-debt', 'customer-request', 'security'], random.randint(0, 2)))
                }
            })

        return {'value': items}

    def _mock_iterations(self) -> Dict[str, Any]:
        """Generate mock iterations"""
        iterations = []
        for i in range(6):
            start = datetime.utcnow() - timedelta(weeks=i*2 + 2)
            end = start + timedelta(weeks=2)
            time_frame = 'past' if i > 0 else 'current'

            iterations.append({
                'id': f'iter-{i}',
                'name': f'Sprint {20 - i}',
                'path': f'Demo Project\\Sprint {20 - i}',
                'attributes': {
                    'startDate': start.isoformat() + 'Z',
                    'finishDate': end.isoformat() + 'Z',
                    'timeFrame': time_frame
                }
            })

        return {'value': iterations}

    def _mock_build_definitions(self) -> Dict[str, Any]:
        """Generate mock build definitions"""
        return {
            'value': [
                {'id': 1, 'name': 'CI Pipeline', 'path': '\\', 'queueStatus': 'enabled', 'createdDate': '2024-01-01T00:00:00Z'},
                {'id': 2, 'name': 'CD Pipeline', 'path': '\\', 'queueStatus': 'enabled', 'createdDate': '2024-01-01T00:00:00Z'},
                {'id': 3, 'name': 'PR Validation', 'path': '\\', 'queueStatus': 'enabled', 'createdDate': '2024-01-01T00:00:00Z'}
            ]
        }

    def _mock_builds(self) -> Dict[str, Any]:
        """Generate mock builds"""
        import random

        builds = []
        for i in range(30):
            days_ago = random.randint(0, 30)
            start = datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(0, 23))
            duration = random.randint(2, 30)
            result = random.choices(['succeeded', 'failed', 'canceled'], weights=[80, 15, 5])[0]

            builds.append({
                'id': 1000 + i,
                'buildNumber': f'20240{i:03d}.{random.randint(1, 10)}',
                'status': 'completed',
                'result': result,
                'startTime': start.isoformat() + 'Z',
                'finishTime': (start + timedelta(minutes=duration)).isoformat() + 'Z',
                'sourceBranch': random.choice(['refs/heads/main', 'refs/heads/develop', 'refs/heads/feature/test']),
                'definition': {'name': random.choice(['CI Pipeline', 'CD Pipeline', 'PR Validation'])},
                'requestedBy': {'displayName': random.choice(['Alice', 'Bob', 'Carol', 'System'])}
            })

        return {'value': builds}
