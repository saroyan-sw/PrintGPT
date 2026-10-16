"""The Codex registration script edits only its own tables (temp config only)."""

import importlib.util
from pathlib import Path
import tomllib

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "register_codex_mcp.py"
spec = importlib.util.spec_from_file_location("register_codex_mcp", SCRIPT)
register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(register)


def test_add_replace_and_remove_preserve_other_servers(tmp_path, monkeypatch):
    monkeypatch.setattr(register, "PYTHON", Path(__file__))  # any existing file
    config = tmp_path / "config.toml"
    config.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "node"\n', encoding="utf-8")
    assert register.main(["--config", str(config), "--scans-dir", str(tmp_path)]) == 0
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    server = data["mcp_servers"]["printgpt-hardware"]
    assert server["args"] == ["-m", "printgpt_hardware_mcp"] and server["env"]["PRINTGPT_HW_ALLOW_PRINT"] == "0"
    assert server["tools"]["submit_print_job"]["approval_mode"] == "prompt"
    assert register.main(["--config", str(config), "--allow-print"]) == 0
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert data["mcp_servers"]["printgpt-hardware"]["env"]["PRINTGPT_HW_ALLOW_PRINT"] == "1"
    assert register.main(["--config", str(config), "--remove"]) == 0
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert set(data["mcp_servers"]) == {"other"} and data["model"] == "x"
    assert len(list(tmp_path.glob("config.toml.bak-printgpt-*"))) >= 1
