#!/usr/bin/env python3
"""
OpenFinOps Command Line Interface
==================================

Provides command-line tools for OpenFinOps operations.
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenFinOps - AI/ML Cost Observability Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="OpenFinOps 0.1.2"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start the OpenFinOps server")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8080, help="Port to run on")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize OpenFinOps configuration")
    init_parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if args.command == "server":
        server_command(args)
    elif args.command == "dashboard":
        dashboard_command(args)
    elif args.command == "init":
        init_command(args)
    else:
        parser.print_help()


def server_command(args=None):
    """Start the OpenFinOps server."""
    print("🚀 Starting OpenFinOps Server...")
    print(f"   Host: {args.host if args else '127.0.0.1'}")
    print(f"   Port: {args.port if args else 8080}")
    print("\n   Dashboard will be available at:")
    print(f"   http://{args.host if args else '127.0.0.1'}:{args.port if args else 8080}\n")

    # TODO: Implement server startup
    print("⚠️  Server implementation coming soon!")
    print("   This will start the Flask/Tornado web server for dashboards")


def dashboard_command(args=None):
    """Launch the dashboard."""
    port = args.port if args else 8080

    print("📊 Launching OpenFinOps Dashboard...")
    print(f"   Port: {port}")
    print(f"   URL: http://localhost:{port}")
    print()

    try:
        from openfinops.observability import ObservabilityHub
        from openfinops.dashboard import CFODashboard

        print("✅ OpenFinOps components loaded successfully")
        print()
        print("🎯 Available Features:")
        print("   • ObservabilityHub - Core observability platform")
        print("   • CFO Dashboard - Financial executive dashboard")
        print("   • Cost Observatory - Multi-cloud cost tracking")
        print("   • LLM Monitoring - AI/ML cost tracking")
        print()
        print("📝 Quick Start:")
        print("   from openfinops import ObservabilityHub")
        print("   hub = ObservabilityHub()")
        print()
        print("⚠️  Web UI server coming soon!")
        print("   Use Python API for now - see docs at:")
        print("   https://github.com/rdmurugan/openfinops")

    except Exception as e:
        print(f"⚠️  Error loading components: {e}")
        print("   Please check installation: pip install openfinops")


def init_command(args):
    """Initialize OpenFinOps configuration."""
    print("🔧 Initializing OpenFinOps...")
    print(f"   Config file: {args.config}")

    # TODO: Implement config initialization
    print("⚠️  Configuration initialization coming soon!")
    print("   This will create a default configuration file")


if __name__ == "__main__":
    main()
