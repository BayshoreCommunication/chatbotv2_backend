from __future__ import annotations
from collections import defaultdict
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self) -> None:
        # company_id → set of dashboard owner websockets
        self._dashboard: dict[str, set[WebSocket]] = defaultdict(set)
        # "company_id:session_id" → widget visitor websockets (a set, since the
        # same visitor can have the site open in more than one tab)
        self._widget: dict[str, set[WebSocket]] = defaultdict(set)
        # every connected admin-panel browser tab, regardless of which staff
        # account — notifications are platform-wide, not per-admin
        self._admin: set[WebSocket] = set()

    # ── Dashboard ─────────────────────────────────────────────────────────────

    async def connect_dashboard(self, company_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._dashboard[company_id].add(ws)
        logger.info("ws.dashboard.connected company_id=%s", company_id)

    def disconnect_dashboard(self, company_id: str, ws: WebSocket) -> None:
        self._dashboard[company_id].discard(ws)
        logger.info("ws.dashboard.disconnected company_id=%s", company_id)

    async def notify_dashboard(self, company_id: str, data: dict) -> None:
        dead: set[WebSocket] = set()
        for ws in list(self._dashboard.get(company_id, set())):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._dashboard[company_id].discard(ws)

    # ── Widget ────────────────────────────────────────────────────────────────

    async def connect_widget(self, session_key: str, ws: WebSocket) -> None:
        await ws.accept()
        self._widget[session_key].add(ws)
        logger.info("ws.widget.connected session_key=%s", session_key)

    def disconnect_widget(self, session_key: str, ws: WebSocket) -> None:
        sockets = self._widget.get(session_key)
        if sockets is not None:
            sockets.discard(ws)
            if not sockets:
                self._widget.pop(session_key, None)
        logger.info("ws.widget.disconnected session_key=%s", session_key)

    async def push_to_widget(self, session_key: str, data: dict) -> None:
        for ws in list(self._widget.get(session_key, set())):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect_widget(session_key, ws)

    # ── Admin (cross-company notifications) ──────────────────────────────────

    async def connect_admin(self, ws: WebSocket) -> None:
        await ws.accept()
        self._admin.add(ws)
        logger.info("ws.admin.connected total=%d", len(self._admin))

    def disconnect_admin(self, ws: WebSocket) -> None:
        self._admin.discard(ws)
        logger.info("ws.admin.disconnected total=%d", len(self._admin))

    async def notify_admin(self, data: dict) -> None:
        dead: set[WebSocket] = set()
        for ws in list(self._admin):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._admin.discard(ws)


ws_manager = WebSocketManager()
