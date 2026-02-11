"""
Jira Integration Client for Operations Intelligence

Provides API access to Jira Cloud for:
- Project management metrics
- Sprint tracking and velocity
- Issue analysis and cycle time
- Team workload distribution
- Backlog health
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


@dataclass
class JiraConfig:
    """Configuration for Jira API access"""
    domain: str  # e.g., "yourcompany.atlassian.net"
    email: str
    api_token: str

    @property
    def base_url(self) -> str:
        return f"https://{self.domain}/rest/api/3"

    @property
    def agile_url(self) -> str:
        return f"https://{self.domain}/rest/agile/1.0"

    @classmethod
    def from_env(cls) -> 'JiraConfig':
        """Create config from environment variables"""
        return cls(
            domain=os.getenv('JIRA_DOMAIN', ''),
            email=os.getenv('JIRA_EMAIL', ''),
            api_token=os.getenv('JIRA_API_TOKEN', '')
        )


class IssueStatus(Enum):
    """Common Jira issue statuses"""
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    IN_REVIEW = "In Review"
    DONE = "Done"
    BLOCKED = "Blocked"


@dataclass
class JiraProject:
    """Jira project representation"""
    id: str
    key: str
    name: str
    project_type: str
    lead: Optional[str] = None
    issue_count: int = 0


@dataclass
class JiraIssue:
    """Jira issue representation"""
    id: str
    key: str
    summary: str
    issue_type: str
    status: str
    priority: str
    assignee: Optional[str]
    reporter: str
    created: datetime
    updated: datetime
    resolved: Optional[datetime]
    story_points: Optional[float]
    sprint: Optional[str]
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)

    @property
    def cycle_time_days(self) -> Optional[float]:
        """Calculate cycle time if resolved"""
        if self.resolved:
            return (self.resolved - self.created).total_seconds() / 86400
        return None


@dataclass
class JiraSprint:
    """Jira sprint representation"""
    id: int
    name: str
    state: str  # active, closed, future
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    complete_date: Optional[datetime]
    goal: Optional[str]
    board_id: int


@dataclass
class SprintMetrics:
    """Sprint velocity and metrics"""
    sprint_name: str
    committed_points: float
    completed_points: float
    completion_rate: float
    issues_completed: int
    issues_not_completed: int
    avg_cycle_time: float
    carryover_points: float


@dataclass
class TeamMetrics:
    """Team workload and performance metrics"""
    team_members: List[Dict[str, Any]]
    total_issues: int
    avg_issues_per_member: float
    workload_distribution: Dict[str, int]
    blocked_issues: int


class JiraClient:
    """
    Jira Cloud API Client

    Provides methods for fetching project management data
    relevant to operations intelligence.
    """

    def __init__(self, config: JiraConfig):
        self.config = config
        self._session = requests.Session()
        self._auth = HTTPBasicAuth(config.email, config.api_token)

    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        response = self._session.request(
            method, url, auth=self._auth, headers=headers, **kwargs
        )
        response.raise_for_status()
        return response.json() if response.text else {}

    def _get(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """GET request to Jira API"""
        url = f"{self.config.base_url}/{endpoint}"
        return self._make_request('GET', url, params=params)

    def _get_agile(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """GET request to Jira Agile API"""
        url = f"{self.config.agile_url}/{endpoint}"
        return self._make_request('GET', url, params=params)

    # ========== Projects ==========

    def get_projects(self) -> List[JiraProject]:
        """Get all accessible projects"""
        data = self._get('project')
        projects = []

        for proj in data:
            projects.append(JiraProject(
                id=proj['id'],
                key=proj['key'],
                name=proj['name'],
                project_type=proj.get('projectTypeKey', 'software'),
                lead=proj.get('lead', {}).get('displayName')
            ))

        return projects

    def get_project(self, project_key: str) -> JiraProject:
        """Get specific project details"""
        data = self._get(f'project/{project_key}')
        return JiraProject(
            id=data['id'],
            key=data['key'],
            name=data['name'],
            project_type=data.get('projectTypeKey', 'software'),
            lead=data.get('lead', {}).get('displayName')
        )

    # ========== Issues ==========

    def search_issues(self, jql: str, max_results: int = 100,
                      fields: List[str] = None) -> List[JiraIssue]:
        """Search issues using JQL"""
        if fields is None:
            fields = [
                'summary', 'issuetype', 'status', 'priority', 'assignee',
                'reporter', 'created', 'updated', 'resolutiondate',
                'customfield_10016',  # Story points (common field ID)
                'labels', 'components', 'sprint'
            ]

        data = self._get('search', params={
            'jql': jql,
            'maxResults': max_results,
            'fields': ','.join(fields)
        })

        issues = []
        for item in data.get('issues', []):
            fields_data = item.get('fields', {})

            # Parse dates
            created = datetime.fromisoformat(fields_data['created'].replace('Z', '+00:00'))
            updated = datetime.fromisoformat(fields_data['updated'].replace('Z', '+00:00'))
            resolved = None
            if fields_data.get('resolutiondate'):
                resolved = datetime.fromisoformat(fields_data['resolutiondate'].replace('Z', '+00:00'))

            # Get sprint name
            sprint_name = None
            sprints = fields_data.get('sprint') or fields_data.get('customfield_10020') or []
            if sprints and isinstance(sprints, list) and len(sprints) > 0:
                sprint_name = sprints[-1].get('name') if isinstance(sprints[-1], dict) else str(sprints[-1])

            issues.append(JiraIssue(
                id=item['id'],
                key=item['key'],
                summary=fields_data.get('summary', ''),
                issue_type=fields_data.get('issuetype', {}).get('name', 'Unknown'),
                status=fields_data.get('status', {}).get('name', 'Unknown'),
                priority=fields_data.get('priority', {}).get('name', 'Medium'),
                assignee=fields_data.get('assignee', {}).get('displayName') if fields_data.get('assignee') else None,
                reporter=fields_data.get('reporter', {}).get('displayName', 'Unknown'),
                created=created,
                updated=updated,
                resolved=resolved,
                story_points=fields_data.get('customfield_10016'),
                sprint=sprint_name,
                labels=fields_data.get('labels', []),
                components=[c.get('name') for c in fields_data.get('components', [])]
            ))

        return issues

    def get_project_issues(self, project_key: str, days: int = 30) -> List[JiraIssue]:
        """Get issues for a project from the last N days"""
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
        jql = f'project = {project_key} AND created >= "{start_date}" ORDER BY created DESC'
        return self.search_issues(jql)

    def get_open_issues(self, project_key: str) -> List[JiraIssue]:
        """Get all open issues for a project"""
        jql = f'project = {project_key} AND statusCategory != Done ORDER BY priority DESC'
        return self.search_issues(jql, max_results=500)

    def get_blocked_issues(self, project_key: str) -> List[JiraIssue]:
        """Get blocked issues"""
        jql = f'project = {project_key} AND (status = Blocked OR labels = blocked OR flagged = Impediment)'
        return self.search_issues(jql)

    # ========== Sprints ==========

    def get_boards(self, project_key: str = None) -> List[Dict[str, Any]]:
        """Get Scrum/Kanban boards"""
        params = {}
        if project_key:
            params['projectKeyOrId'] = project_key

        data = self._get_agile('board', params=params)
        return data.get('values', [])

    def get_sprints(self, board_id: int, state: str = None) -> List[JiraSprint]:
        """Get sprints for a board"""
        params = {}
        if state:
            params['state'] = state

        data = self._get_agile(f'board/{board_id}/sprint', params=params)
        sprints = []

        for s in data.get('values', []):
            start_date = None
            end_date = None
            complete_date = None

            if s.get('startDate'):
                start_date = datetime.fromisoformat(s['startDate'].replace('Z', '+00:00'))
            if s.get('endDate'):
                end_date = datetime.fromisoformat(s['endDate'].replace('Z', '+00:00'))
            if s.get('completeDate'):
                complete_date = datetime.fromisoformat(s['completeDate'].replace('Z', '+00:00'))

            sprints.append(JiraSprint(
                id=s['id'],
                name=s['name'],
                state=s['state'],
                start_date=start_date,
                end_date=end_date,
                complete_date=complete_date,
                goal=s.get('goal'),
                board_id=board_id
            ))

        return sprints

    def get_sprint_issues(self, sprint_id: int) -> List[JiraIssue]:
        """Get issues in a sprint"""
        data = self._get_agile(f'sprint/{sprint_id}/issue')
        issues = []

        for item in data.get('issues', []):
            fields_data = item.get('fields', {})
            created = datetime.fromisoformat(fields_data['created'].replace('Z', '+00:00'))
            updated = datetime.fromisoformat(fields_data['updated'].replace('Z', '+00:00'))
            resolved = None
            if fields_data.get('resolutiondate'):
                resolved = datetime.fromisoformat(fields_data['resolutiondate'].replace('Z', '+00:00'))

            issues.append(JiraIssue(
                id=item['id'],
                key=item['key'],
                summary=fields_data.get('summary', ''),
                issue_type=fields_data.get('issuetype', {}).get('name', 'Unknown'),
                status=fields_data.get('status', {}).get('name', 'Unknown'),
                priority=fields_data.get('priority', {}).get('name', 'Medium'),
                assignee=fields_data.get('assignee', {}).get('displayName') if fields_data.get('assignee') else None,
                reporter=fields_data.get('reporter', {}).get('displayName', 'Unknown'),
                created=created,
                updated=updated,
                resolved=resolved,
                story_points=fields_data.get('customfield_10016'),
                sprint=None,
                labels=fields_data.get('labels', []),
                components=[c.get('name') for c in fields_data.get('components', [])]
            ))

        return issues

    # ========== Metrics ==========

    def get_sprint_metrics(self, sprint_id: int) -> SprintMetrics:
        """Calculate sprint velocity and metrics"""
        issues = self.get_sprint_issues(sprint_id)

        completed = [i for i in issues if i.status.lower() in ['done', 'closed', 'resolved']]
        not_completed = [i for i in issues if i.status.lower() not in ['done', 'closed', 'resolved']]

        committed_points = sum(i.story_points or 0 for i in issues)
        completed_points = sum(i.story_points or 0 for i in completed)
        carryover_points = sum(i.story_points or 0 for i in not_completed)

        # Calculate average cycle time for completed issues
        cycle_times = [i.cycle_time_days for i in completed if i.cycle_time_days is not None]
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0

        # Get sprint name
        sprints = self.get_sprints(issues[0].sprint if issues else 0)
        sprint_name = next((s.name for s in sprints if s.id == sprint_id), f"Sprint {sprint_id}")

        return SprintMetrics(
            sprint_name=sprint_name,
            committed_points=committed_points,
            completed_points=completed_points,
            completion_rate=(completed_points / committed_points * 100) if committed_points > 0 else 0,
            issues_completed=len(completed),
            issues_not_completed=len(not_completed),
            avg_cycle_time=avg_cycle_time,
            carryover_points=carryover_points
        )

    def get_velocity_trend(self, board_id: int, num_sprints: int = 6) -> List[SprintMetrics]:
        """Get velocity trend for last N sprints"""
        sprints = self.get_sprints(board_id, state='closed')
        sprints = sorted(sprints, key=lambda s: s.complete_date or datetime.min, reverse=True)[:num_sprints]

        metrics = []
        for sprint in sprints:
            try:
                metric = self.get_sprint_metrics(sprint.id)
                metrics.append(metric)
            except Exception as e:
                logger.warning(f"Could not get metrics for sprint {sprint.id}: {e}")

        return metrics

    def get_team_metrics(self, project_key: str) -> TeamMetrics:
        """Get team workload metrics"""
        issues = self.get_open_issues(project_key)

        # Workload by assignee
        workload = {}
        for issue in issues:
            assignee = issue.assignee or 'Unassigned'
            workload[assignee] = workload.get(assignee, 0) + 1

        # Team members with their stats
        team_members = [
            {'name': name, 'issues': count}
            for name, count in sorted(workload.items(), key=lambda x: x[1], reverse=True)
        ]

        # Blocked issues
        blocked = [i for i in issues if 'blocked' in i.status.lower() or 'blocked' in [l.lower() for l in i.labels]]

        num_assignees = len([m for m in team_members if m['name'] != 'Unassigned'])

        return TeamMetrics(
            team_members=team_members,
            total_issues=len(issues),
            avg_issues_per_member=len(issues) / num_assignees if num_assignees > 0 else 0,
            workload_distribution=workload,
            blocked_issues=len(blocked)
        )

    def get_operations_summary(self, project_key: str) -> Dict[str, Any]:
        """Get comprehensive operations summary"""
        # Get project info
        project = self.get_project(project_key)

        # Get issues
        open_issues = self.get_open_issues(project_key)
        recent_issues = self.get_project_issues(project_key, days=30)

        # Status breakdown
        status_counts = {}
        priority_counts = {}
        type_counts = {}

        for issue in open_issues:
            status_counts[issue.status] = status_counts.get(issue.status, 0) + 1
            priority_counts[issue.priority] = priority_counts.get(issue.priority, 0) + 1
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1

        # Resolved in last 30 days
        resolved_recently = [i for i in recent_issues if i.resolved]

        # Average cycle time
        cycle_times = [i.cycle_time_days for i in resolved_recently if i.cycle_time_days]
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0

        # Team metrics
        team = self.get_team_metrics(project_key)

        # Get velocity if Scrum board exists
        velocity_data = []
        try:
            boards = self.get_boards(project_key)
            scrum_board = next((b for b in boards if b.get('type') == 'scrum'), None)
            if scrum_board:
                velocity_data = [
                    {
                        'sprint': m.sprint_name,
                        'committed': m.committed_points,
                        'completed': m.completed_points,
                        'completion_rate': m.completion_rate
                    }
                    for m in self.get_velocity_trend(scrum_board['id'], 6)
                ]
        except Exception as e:
            logger.warning(f"Could not fetch velocity data: {e}")

        return {
            'project': {
                'key': project.key,
                'name': project.name,
                'type': project.project_type,
                'lead': project.lead
            },
            'issues': {
                'open': len(open_issues),
                'resolved_last_30_days': len(resolved_recently),
                'by_status': status_counts,
                'by_priority': priority_counts,
                'by_type': type_counts
            },
            'performance': {
                'avg_cycle_time_days': round(avg_cycle_time, 1),
                'blocked_issues': team.blocked_issues,
                'velocity_trend': velocity_data
            },
            'team': {
                'total_members': len([m for m in team.team_members if m['name'] != 'Unassigned']),
                'avg_issues_per_member': round(team.avg_issues_per_member, 1),
                'workload': team.team_members[:10]  # Top 10
            },
            'alerts': self._generate_alerts(open_issues, team, avg_cycle_time)
        }

    def _generate_alerts(self, issues: List[JiraIssue], team: TeamMetrics,
                        avg_cycle_time: float) -> List[Dict[str, Any]]:
        """Generate operational alerts"""
        alerts = []

        # High blocked count
        if team.blocked_issues > 3:
            alerts.append({
                'type': 'warning',
                'category': 'Blockers',
                'message': f'{team.blocked_issues} issues are currently blocked',
                'recommendation': 'Review blocked issues and escalate impediments'
            })

        # High cycle time
        if avg_cycle_time > 14:
            alerts.append({
                'type': 'warning',
                'category': 'Cycle Time',
                'message': f'Average cycle time is {avg_cycle_time:.1f} days',
                'recommendation': 'Review workflow for bottlenecks'
            })

        # Unbalanced workload
        if team.team_members:
            workloads = [m['issues'] for m in team.team_members if m['name'] != 'Unassigned']
            if workloads:
                max_load = max(workloads)
                min_load = min(workloads) if min(workloads) > 0 else 1
                if max_load / min_load > 3:
                    alerts.append({
                        'type': 'info',
                        'category': 'Workload',
                        'message': 'Significant workload imbalance detected',
                        'recommendation': 'Consider redistributing work assignments'
                    })

        # Too many high priority issues
        high_priority = [i for i in issues if i.priority.lower() in ['highest', 'high', 'critical']]
        if len(high_priority) > 10:
            alerts.append({
                'type': 'warning',
                'category': 'Priorities',
                'message': f'{len(high_priority)} high-priority issues in backlog',
                'recommendation': 'Review prioritization - everything cannot be high priority'
            })

        return alerts


# Demo mode for testing
class JiraDemoClient(JiraClient):
    """Demo client with mock data for testing"""

    def __init__(self):
        config = JiraConfig(domain='demo.atlassian.net', email='demo@example.com', api_token='demo')
        super().__init__(config)

    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Return mock data"""
        return self._get_mock_data(url)

    def _get_mock_data(self, url: str) -> Dict[str, Any]:
        """Generate mock Jira data"""
        import random

        if '/project' in url and '/project/' not in url:
            return [
                {'id': '10001', 'key': 'PROJ', 'name': 'Main Project', 'projectTypeKey': 'software', 'lead': {'displayName': 'John Smith'}},
                {'id': '10002', 'key': 'OPS', 'name': 'Operations', 'projectTypeKey': 'software', 'lead': {'displayName': 'Jane Doe'}}
            ]
        elif '/project/' in url:
            return {'id': '10001', 'key': 'PROJ', 'name': 'Main Project', 'projectTypeKey': 'software', 'lead': {'displayName': 'John Smith'}}
        elif '/search' in url:
            return self._mock_issues()
        elif '/board' in url and '/sprint' in url:
            return self._mock_sprints()
        elif '/board' in url:
            return {'values': [{'id': 1, 'name': 'PROJ Board', 'type': 'scrum'}]}
        elif '/sprint/' in url and '/issue' in url:
            return self._mock_issues()

        return {}

    def _mock_issues(self) -> Dict[str, Any]:
        """Generate mock issues"""
        import random

        statuses = ['To Do', 'In Progress', 'In Review', 'Done', 'Blocked']
        priorities = ['Highest', 'High', 'Medium', 'Low', 'Lowest']
        types = ['Story', 'Bug', 'Task', 'Epic', 'Sub-task']
        assignees = ['Alice Johnson', 'Bob Smith', 'Carol Williams', 'David Brown', 'Eve Davis', None]

        issues = []
        for i in range(30):
            days_ago = random.randint(1, 60)
            created = datetime.utcnow() - timedelta(days=days_ago)
            resolved = None
            if random.random() > 0.4:
                resolved = created + timedelta(days=random.randint(1, 14))

            issues.append({
                'id': str(10000 + i),
                'key': f'PROJ-{100 + i}',
                'fields': {
                    'summary': f'Issue {i}: {random.choice(["Fix bug", "Add feature", "Update docs", "Refactor", "Test"])}',
                    'issuetype': {'name': random.choice(types)},
                    'status': {'name': random.choice(statuses)},
                    'priority': {'name': random.choice(priorities)},
                    'assignee': {'displayName': random.choice(assignees)} if random.choice(assignees) else None,
                    'reporter': {'displayName': 'Reporter Name'},
                    'created': created.isoformat() + 'Z',
                    'updated': (created + timedelta(days=random.randint(0, days_ago))).isoformat() + 'Z',
                    'resolutiondate': resolved.isoformat() + 'Z' if resolved else None,
                    'customfield_10016': random.choice([1, 2, 3, 5, 8, 13, None]),
                    'labels': random.sample(['backend', 'frontend', 'urgent', 'tech-debt'], random.randint(0, 2)),
                    'components': []
                }
            })

        return {'issues': issues, 'total': len(issues)}

    def _mock_sprints(self) -> Dict[str, Any]:
        """Generate mock sprints"""
        sprints = []
        for i in range(6):
            start = datetime.utcnow() - timedelta(weeks=i*2 + 2)
            end = start + timedelta(weeks=2)
            sprints.append({
                'id': 100 + i,
                'name': f'Sprint {20 - i}',
                'state': 'closed' if i > 0 else 'active',
                'startDate': start.isoformat() + 'Z',
                'endDate': end.isoformat() + 'Z',
                'completeDate': end.isoformat() + 'Z' if i > 0 else None,
                'goal': f'Sprint {20 - i} goals'
            })
        return {'values': sprints}
