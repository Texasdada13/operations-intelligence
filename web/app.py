"""
Flask Application - Operations Intelligence

Web application for the Fractional COO product.
"""

import os
import sys
import json
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database configuration
database_url = os.getenv('DATABASE_URL', 'sqlite:///operations_intelligence.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize database
from src.database.models import db
db.init_app(app)

# Initialize repository
from src.database.repository import OperationsRepository
repo = OperationsRepository()

# Initialize AI chat
from src.ai_core.chat_engine import ChatEngine, ConversationMode, create_chat_engine

# Initialize benchmark engines
from src.patterns.benchmark_engine import (
    create_operations_benchmarks,
    create_manufacturing_benchmarks,
    create_service_benchmarks
)

# Initialize scoring engines
from src.patterns.weighted_scoring import (
    create_process_efficiency_engine,
    create_resource_utilization_engine
)

# Initialize integrations
from src.integrations import OperationsIntegrationManager, IntegrationType
integration_manager = OperationsIntegrationManager()

# App metadata
APP_NAME = "Operations Intelligence"
APP_VERSION = "1.0.0"


@app.context_processor
def inject_globals():
    """Inject global variables into templates."""
    return {
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'current_year': datetime.now().year
    }


# Create tables on first request
@app.before_request
def create_tables():
    if not hasattr(app, '_tables_created'):
        db.create_all()
        app._tables_created = True


# ==================== Web Routes ====================

@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Main dashboard."""
    organizations = repo.get_all_organizations()
    return render_template('dashboard.html', organizations=organizations)


@app.route('/organization/<org_id>')
def organization_detail(org_id):
    """Organization detail page."""
    org = repo.get_organization(org_id)
    if not org:
        return render_template('404.html'), 404

    departments = repo.get_departments(org_id)
    processes = repo.get_processes(org_id)
    resources = repo.get_resources(org_id)
    kpi_entries = repo.get_kpi_entries(org_id, limit=12)
    latest_benchmark = repo.get_latest_benchmark(org_id)

    return render_template(
        'organization.html',
        organization=org,
        departments=departments,
        processes=processes,
        resources=resources,
        kpi_entries=kpi_entries,
        benchmark=latest_benchmark
    )


@app.route('/processes/<org_id>')
def processes_view(org_id):
    """Process analysis view."""
    org = repo.get_organization(org_id)
    if not org:
        return render_template('404.html'), 404

    processes = repo.get_processes(org_id)
    return render_template('processes.html', organization=org, processes=processes)


@app.route('/benchmarks/<org_id>')
def benchmarks_view(org_id):
    """Benchmarks view."""
    org = repo.get_organization(org_id)
    if not org:
        return render_template('404.html'), 404

    benchmarks = repo.get_benchmark_results(org_id)
    latest = benchmarks[0] if benchmarks else None

    return render_template(
        'benchmarks.html',
        organization=org,
        benchmarks=benchmarks,
        latest_benchmark=latest
    )


@app.route('/capacity/<org_id>')
def capacity_view(org_id):
    """Capacity planning view."""
    org = repo.get_organization(org_id)
    if not org:
        return render_template('404.html'), 404

    resources = repo.get_resources(org_id)
    capacity_plans = repo.get_capacity_plans(org_id)

    return render_template(
        'capacity.html',
        organization=org,
        resources=resources,
        capacity_plans=capacity_plans
    )


@app.route('/chat')
@app.route('/chat/<org_id>')
def chat_view(org_id=None):
    """AI COO Consultant chat view."""
    organizations = repo.get_all_organizations()
    org = repo.get_organization(org_id) if org_id else None

    return render_template(
        'chat.html',
        organization=org,
        organizations=organizations
    )


@app.route('/integrations')
def integrations_view():
    """DevOps integrations page."""
    return render_template('integrations.html')


# ==================== Integration API Routes ====================

@app.route('/api/integrations/status')
def api_integration_status():
    """Get integration connection status."""
    jira_status = integration_manager.get_status(IntegrationType.JIRA)
    azure_status = integration_manager.get_status(IntegrationType.AZURE_DEVOPS)
    demo_status = integration_manager.get_status(IntegrationType.DEMO)
    return jsonify({
        'jira': {
            'connected': jira_status.is_connected,
            'organization': jira_status.organization
        },
        'azure_devops': {
            'connected': azure_status.is_connected,
            'organization': azure_status.organization,
            'project': azure_status.project
        },
        'demo_mode': demo_status.is_connected
    })


@app.route('/api/integrations/demo/enable', methods=['POST'])
def api_enable_demo_mode():
    """Enable demo mode for integrations."""
    integration_manager.enable_demo_mode()
    return jsonify({'success': True, 'message': 'Demo mode enabled'})


@app.route('/api/integrations/jira/projects')
def api_jira_projects():
    """Get Jira projects."""
    projects = integration_manager.get_jira_projects()
    if not projects:
        return jsonify({'error': 'No Jira connection or data'}), 400
    return jsonify(projects)


@app.route('/api/integrations/jira/summary/<project_key>')
def api_jira_summary(project_key):
    """Get Jira operations summary for a project."""
    summary = integration_manager.get_jira_summary(project_key)
    if not summary:
        return jsonify({'error': 'No data available'}), 400
    return jsonify(summary)


@app.route('/api/integrations/jira/velocity/<project_key>')
def api_jira_velocity(project_key):
    """Get Jira velocity trend."""
    num_sprints = request.args.get('sprints', 6, type=int)
    velocity = integration_manager.get_jira_velocity(project_key, num_sprints)
    if not velocity:
        return jsonify({'error': 'No velocity data available'}), 400
    return jsonify(velocity)


@app.route('/api/integrations/azure/summary')
def api_azure_summary():
    """Get Azure DevOps operations summary."""
    summary = integration_manager.get_azure_summary()
    if not summary:
        return jsonify({'error': 'No Azure DevOps connection or data'}), 400
    return jsonify(summary)


@app.route('/api/integrations/azure/builds')
def api_azure_builds():
    """Get Azure DevOps build metrics."""
    days = request.args.get('days', 30, type=int)
    builds = integration_manager.get_azure_builds(days)
    if not builds:
        return jsonify({'error': 'No build data available'}), 400
    return jsonify(builds)


@app.route('/api/integrations/operations-summary')
def api_unified_operations_summary():
    """Get combined operations summary from all sources."""
    summary = integration_manager.get_unified_operations_summary()
    return jsonify(summary)


# ==================== API Routes ====================

@app.route('/api/organizations', methods=['GET'])
def api_list_organizations():
    """List all organizations."""
    orgs = repo.get_all_organizations()
    return jsonify({
        'success': True,
        'organizations': [org.to_dict() for org in orgs]
    })


@app.route('/api/organizations', methods=['POST'])
@limiter.limit("10 per hour")
def api_create_organization():
    """Create a new organization."""
    data = request.get_json()

    if not data or not data.get('name'):
        return jsonify({'success': False, 'error': 'Name is required'}), 400

    org = repo.create_organization(data)
    return jsonify({
        'success': True,
        'organization': org.to_dict()
    })


@app.route('/api/organizations/<org_id>', methods=['GET'])
def api_get_organization(org_id):
    """Get organization details."""
    summary = repo.get_organization_summary(org_id)
    if not summary:
        return jsonify({'success': False, 'error': 'Organization not found'}), 404

    return jsonify({
        'success': True,
        **summary
    })


@app.route('/api/organizations/<org_id>', methods=['PUT'])
def api_update_organization(org_id):
    """Update organization."""
    data = request.get_json()
    org = repo.update_organization(org_id, data)

    if not org:
        return jsonify({'success': False, 'error': 'Organization not found'}), 404

    return jsonify({
        'success': True,
        'organization': org.to_dict()
    })


@app.route('/api/organizations/<org_id>/kpis', methods=['POST'])
@limiter.limit("30 per hour")
def api_add_kpi_entry(org_id):
    """Add KPI entry."""
    data = request.get_json()

    if not data.get('period_date'):
        return jsonify({'success': False, 'error': 'Period date is required'}), 400

    # Parse date
    if isinstance(data['period_date'], str):
        data['period_date'] = datetime.strptime(data['period_date'], '%Y-%m-%d').date()

    entry = repo.create_kpi_entry(org_id, data)
    return jsonify({
        'success': True,
        'kpi_entry': entry.to_dict()
    })


@app.route('/api/organizations/<org_id>/kpis', methods=['GET'])
def api_get_kpi_entries(org_id):
    """Get KPI entries."""
    limit = request.args.get('limit', 12, type=int)
    entries = repo.get_kpi_entries(org_id, limit=limit)

    return jsonify({
        'success': True,
        'kpi_entries': [e.to_dict() for e in entries]
    })


@app.route('/api/organizations/<org_id>/processes', methods=['GET'])
def api_get_processes(org_id):
    """Get processes."""
    processes = repo.get_processes(org_id)
    return jsonify({
        'success': True,
        'processes': [p.to_dict() for p in processes]
    })


@app.route('/api/organizations/<org_id>/processes', methods=['POST'])
@limiter.limit("30 per hour")
def api_create_process(org_id):
    """Create a process."""
    data = request.get_json()

    if not data.get('name'):
        return jsonify({'success': False, 'error': 'Name is required'}), 400

    process = repo.create_process(org_id, data)
    return jsonify({
        'success': True,
        'process': process.to_dict()
    })


@app.route('/api/organizations/<org_id>/resources', methods=['GET'])
def api_get_resources(org_id):
    """Get resources."""
    resource_type = request.args.get('type')
    resources = repo.get_resources(org_id, resource_type=resource_type)

    return jsonify({
        'success': True,
        'resources': [r.to_dict() for r in resources]
    })


@app.route('/api/organizations/<org_id>/resources', methods=['POST'])
@limiter.limit("30 per hour")
def api_create_resource(org_id):
    """Create a resource."""
    data = request.get_json()

    if not data.get('name') or not data.get('resource_type'):
        return jsonify({'success': False, 'error': 'Name and resource_type are required'}), 400

    resource = repo.create_resource(org_id, data)
    return jsonify({
        'success': True,
        'resource': resource.to_dict()
    })


@app.route('/api/organizations/<org_id>/benchmark', methods=['POST'])
@limiter.limit("10 per hour")
def api_run_benchmark(org_id):
    """Run benchmark analysis."""
    data = request.get_json() or {}
    benchmark_type = data.get('benchmark_type', 'operations')

    # Get latest KPI entry
    latest_kpi = repo.get_latest_kpi_entry(org_id)
    if not latest_kpi:
        return jsonify({'success': False, 'error': 'No KPI data available for benchmarking'}), 400

    # Select benchmark engine
    if benchmark_type == 'manufacturing':
        engine = create_manufacturing_benchmarks()
    elif benchmark_type == 'service':
        engine = create_service_benchmarks()
    else:
        engine = create_operations_benchmarks()

    # Run benchmark analysis
    kpi_values = latest_kpi.get_kpi_values()
    # Filter out None values
    actual_values = {k: v for k, v in kpi_values.items() if v is not None}

    if not actual_values:
        return jsonify({'success': False, 'error': 'No valid KPI values for benchmarking'}), 400

    report = engine.analyze(actual_values, entity_id=org_id)

    # Save result
    result_data = report.to_dict()
    result_data['benchmark_type'] = benchmark_type
    result = repo.save_benchmark_result(org_id, result_data)

    # Update organization metrics
    org = repo.get_organization(org_id)
    if org:
        org.efficiency_score = report.overall_score
        if report.overall_score >= 80:
            org.risk_level = 'Low'
        elif report.overall_score >= 60:
            org.risk_level = 'Medium'
        elif report.overall_score >= 40:
            org.risk_level = 'High'
        else:
            org.risk_level = 'Critical'
        db.session.commit()

    return jsonify({
        'success': True,
        'benchmark': result.to_dict()
    })


# ==================== Chat API ====================

# Store chat engines per session (simplified - use Redis in production)
chat_sessions = {}


@app.route('/api/chat/session', methods=['POST'])
@limiter.limit("20 per hour")
def api_create_chat_session():
    """Create a new chat session."""
    data = request.get_json() or {}
    org_id = data.get('organization_id')

    # Create database session
    session = repo.create_chat_session(org_id)

    # Create chat engine
    chat_engine = create_chat_engine()

    # Set context if organization provided
    if org_id:
        context = _build_chat_context(org_id)
        chat_engine.set_context(context)

    chat_sessions[session.id] = chat_engine

    return jsonify({
        'success': True,
        'session_id': session.id,
        'suggested_prompts': chat_engine.get_suggested_prompts()
    })


@app.route('/api/chat/stream', methods=['POST'])
@limiter.limit("50 per hour")
def api_chat_stream():
    """Stream chat response."""
    data = request.get_json()
    session_id = data.get('session_id')
    message = data.get('message')

    if not session_id or not message:
        return jsonify({'success': False, 'error': 'session_id and message required'}), 400

    # Get or create chat engine
    chat_engine = chat_sessions.get(session_id)
    if not chat_engine:
        chat_engine = create_chat_engine()
        chat_sessions[session_id] = chat_engine

    # Save user message
    repo.add_chat_message(session_id, 'user', message)

    def generate():
        full_response = ""
        try:
            for token in chat_engine.stream_chat(message):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Save assistant response
            repo.add_chat_message(session_id, 'assistant', full_response)

            # Send completion with suggested prompts
            yield f"data: {json.dumps({'type': 'done', 'suggested_prompts': chat_engine.get_suggested_prompts()})}\n\n"

        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/chat/message', methods=['POST'])
@limiter.limit("50 per hour")
def api_chat_message():
    """Send chat message (non-streaming)."""
    data = request.get_json()
    session_id = data.get('session_id')
    message = data.get('message')

    if not session_id or not message:
        return jsonify({'success': False, 'error': 'session_id and message required'}), 400

    chat_engine = chat_sessions.get(session_id)
    if not chat_engine:
        chat_engine = create_chat_engine()
        chat_sessions[session_id] = chat_engine

    # Save user message
    repo.add_chat_message(session_id, 'user', message)

    # Get response
    response = chat_engine.chat(message)

    # Save assistant response
    repo.add_chat_message(session_id, 'assistant', response)

    return jsonify({
        'success': True,
        'response': response,
        'suggested_prompts': chat_engine.get_suggested_prompts()
    })


def _build_chat_context(org_id: str) -> dict:
    """Build operational context for chat."""
    org = repo.get_organization(org_id)
    if not org:
        return {}

    latest_kpi = repo.get_latest_kpi_entry(org_id)
    latest_benchmark = repo.get_latest_benchmark(org_id)
    processes = repo.get_processes(org_id)
    resources = repo.get_resources(org_id)

    context = {
        'organization': org.to_dict(),
        'kpis': latest_kpi.get_kpi_values() if latest_kpi else {},
        'processes': [p.to_dict() for p in processes[:10]],
        'resources': [r.to_dict() for r in resources[:10]]
    }

    if latest_benchmark:
        context['benchmark'] = {
            'overall_score': latest_benchmark.overall_score,
            'overall_rating': latest_benchmark.overall_rating,
            'grade': latest_benchmark.grade,
            'strengths': latest_benchmark.get_strengths(),
            'improvements': latest_benchmark.get_improvements()
        }

    return context


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded. Please try again later.'
    }), 429


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


# ==================== Health Check ====================

@app.route('/health')
@limiter.exempt
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'app': APP_NAME,
        'version': APP_VERSION
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5106))
    debug = os.getenv('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
