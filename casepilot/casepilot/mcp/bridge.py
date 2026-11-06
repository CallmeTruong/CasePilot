"""Episode-scoped authenticated local IPC. Only JSON bytes cross the boundary."""

import json
import secrets
import threading
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener


class EpisodeBridge:
    def __init__(self, episode_id, handler):
        self.episode_id, self.handler = episode_id, handler
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(("127.0.0.1", 0), authkey=self.authkey)
        self.address = self.listener.address
        self._closed = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self._closed:
            try:
                connection = self.listener.accept()
            except AuthenticationError:
                continue
            except (OSError, EOFError):
                return
            with connection:
                try:
                    if not connection.poll(15):
                        continue
                    payload = json.loads(connection.recv_bytes(maxlength=8 * 1024 * 1024))
                    if payload.get("episode_id") != self.episode_id:
                        result = {"protocol_error": "episode_mismatch"}
                    elif payload.get("action") == "shutdown":
                        result = {"closed": True}
                        self._closed = True
                    else:
                        result = self.handler(payload)
                    connection.send_bytes(json.dumps(result).encode())
                except Exception:
                    try:
                        connection.send_bytes(b'{"protocol_error":"bridge_request_failed"}')
                    except OSError:
                        pass

    def request(self, payload):
        with Client(self.address, authkey=self.authkey) as connection:
            connection.send_bytes(json.dumps(payload).encode())
            if not connection.poll(15):
                raise TimeoutError("Episode bridge response timed out")
            return json.loads(connection.recv_bytes(maxlength=8 * 1024 * 1024))

    def close(self):
        if not self._closed:
            self.request({"episode_id": self.episode_id, "action": "shutdown"})
        self.thread.join(timeout=5)
        self.listener.close()


def request_from_environment(payload):
    import os

    address = ("127.0.0.1", int(os.environ["CASEPILOT_BRIDGE_PORT"]))
    key = bytes.fromhex(os.environ["CASEPILOT_BRIDGE_KEY"])
    payload = {**payload, "episode_id": os.environ["CASEPILOT_EPISODE_ID"]}
    with Client(address, authkey=key) as connection:
        connection.send_bytes(json.dumps(payload).encode())
        if not connection.poll(15):
            raise TimeoutError("Episode bridge timed out")
        return json.loads(connection.recv_bytes(maxlength=8 * 1024 * 1024))
