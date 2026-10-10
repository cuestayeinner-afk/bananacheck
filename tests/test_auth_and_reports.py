import http.server
import importlib.util
import json
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import bananacheck_server as server_module
import bananacheck_storage as storage


class AuthAndReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.database_path = str(Path(cls.temp_dir.name) / "test.db")
        cls.original_database_path = storage.DATABASE_PATH
        storage.DATABASE_PATH = cls.database_path
        storage.initialize(directory=cls.temp_dir.name, path=cls.database_path)
        storage.save_accounts([
            server_module._hash_credencial("tester", "correct-password", "USUARIO"),
            server_module._hash_credencial("admin", "admin-password", "ADMIN"),
        ])
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server_module.BananaCheckHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=3)
        storage.DATABASE_PATH = cls.original_database_path
        cls.temp_dir.cleanup()

    def setUp(self):
        server_module.SESIONES.clear()
        server_module.INTENTOS_LOGIN.clear()
        with storage.connect() as database:
            database.execute("DELETE FROM analyses")
            database.execute("DELETE FROM comparisons")
            database.execute("DELETE FROM history")

    def request(self, path, body=None, cookie=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        return response.status, response.headers, response.read()

    def login(self, username, password):
        status, headers, body = self.request("/auth/login", {"usuario": username, "contrasena": password})
        cookie = headers.get("Set-Cookie", "").split(";", 1)[0]
        return status, json.loads(body), cookie

    def test_staff_password_enters_staff_role(self):
        status, result, _ = self.login("", "STAFF")
        self.assertEqual(status, 200)
        self.assertEqual(result["rol"], "STAFF")

    def test_authenticated_user_cannot_read_private_data_directory(self):
        _, _, cookie = self.login("tester", "correct-password")
        private_dir = Path(server_module.storage.DATA_DIR)
        private_dir.mkdir(parents=True, exist_ok=True)
        probe = private_dir / "private-test-file.txt"
        probe.write_text("private", encoding="utf-8")
        try:
            status, _, _ = self.request("/.bananacheck_data/private-test-file.txt", cookie=cookie)
            self.assertEqual(status, 404)
        finally:
            probe.unlink(missing_ok=True)

    def test_normal_user_exports_only_own_analyses(self):
        storage.save_analysis("fruta", {
            "fecha_hora": "2026-10-10 10:00:00",
            "diagnostico": "=1+1",
        }, "tester", 50, "=1+1")
        storage.save_analysis("fruta", {
            "fecha_hora": "2026-10-10 11:00:00",
            "diagnostico": "OTHER_ONLY",
        }, "other", 75, "OTHER_ONLY")
        status, result, cookie = self.login("tester", "correct-password")
        self.assertEqual(status, 200)
        self.assertEqual(result["rol"], "USUARIO")
        status, headers, body = self.request("/mi-historial.csv?tipo=fruta&username=other", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertIn("text/csv", headers.get("Content-Type", ""))
        self.assertIn(b"'=1+1", body)
        self.assertNotIn(b"OTHER_ONLY", body)

    def test_staff_and_admin_cannot_export_but_staff_can_view_metrics(self):
        _, _, cookie = self.login("", "STAFF")
        status, _, _ = self.request("/mi-historial.csv", cookie=cookie)
        self.assertEqual(status, 403)
        status, _, metrics_body = self.request("/staff/model-metrics", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertIn("metrics", json.loads(metrics_body))
        _, _, admin_cookie = self.login("admin", "admin-password")
        status, _, _ = self.request("/mi-historial.csv", cookie=admin_cookie)
        self.assertEqual(status, 403)

    def test_invalid_export_date_is_rejected(self):
        _, _, cookie = self.login("tester", "correct-password")
        status, _, _ = self.request("/mi-historial.csv?desde=ayer", cookie=cookie)
        self.assertEqual(status, 400)

    def test_failed_password_attempts_are_limited(self):
        statuses = [self.login("tester", "incorrect-password")[0] for _ in range(5)]
        self.assertEqual(statuses[-1], 429)

    def test_server_reports_port_conflict_clear_error(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
            blocker.bind(("127.0.0.1", 0))
            port = blocker.getsockname()[1]
            blocker.listen(1)
            with self.assertRaises(RuntimeError):
                server_module.create_server(host="127.0.0.1", port=port)

    @unittest.skipUnless(importlib.util.find_spec("reportlab"), "ReportLab no instalado")
    def test_saved_analysis_downloads_a_pdf(self):
        _, _, cookie = self.login("tester", "correct-password")
        status, _, body = self.request("/guardar/fruta", {
            "color_cascara": "amarillo",
            "manchas_oscuras": "ninguna",
            "golpes_deformaciones": "ninguno",
            "signos_enfermedad": "sin enfermedad",
            "diagnostico": "Apta para consumo",
        }, cookie)
        self.assertEqual(status, 200)
        analysis_id = json.loads(body)["id"]
        status, headers, pdf = self.request(f"/analisis/{analysis_id}/pdf", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "application/pdf")
        self.assertTrue(pdf.startswith(b"%PDF-"))
        for username, password in (("", "STAFF"), ("admin", "admin-password")):
            _, _, privileged_cookie = self.login(username, password)
            status, _, _ = self.request(f"/analisis/{analysis_id}/pdf", cookie=privileged_cookie)
            self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()