"""Admin HTTP surface.

The admin REST API lives under ``/admin/api/*`` and is registered onto the
main FastAPI app via :func:`admin.routes.register_admin_routes`. The Gradio
UI is mounted separately at ``/admin`` (see ``main.py``).
"""
