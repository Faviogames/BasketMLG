"""CLI y ciclo de vida live, sin red, entrenamiento ni escritura de registros."""
import importlib.util
import asyncio
import ast
import io
from pathlib import Path
import sys
import types
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main = load_source("basketmlg_cli_test", "main.py")
config_stub = types.ModuleType("config")
config_stub.MODELS_FOLDER = "./models"
config_stub.UI_MESSAGES = {"training_start": "Entrenamiento"}
with patch.dict(sys.modules, {"config": config_stub, "joblib": types.ModuleType("joblib")}):
    live_system = load_source("basketmlg_live_system_test", "analysis/live_system.py")


class CliTests(unittest.TestCase):
    def invoke(self, args, modules=None):
        output = io.StringIO()
        with patch.object(sys, "argv", ["main.py", *args]), \
                patch.dict(sys.modules, modules or {}), \
                redirect_stdout(output), redirect_stderr(output):
            code = main.main()
        return code, output.getvalue()

    def test_no_arguments_shows_help_without_prompt(self):
        with patch("builtins.input", side_effect=AssertionError("No debe pedir datos")):
            code, output = self.invoke([])
        self.assertEqual(code, 0)
        self.assertIn("--models", output)

    def test_help_does_not_import_ml_or_transport(self):
        original_import = __import__

        def guarded(name, *args, **kwargs):
            if name.startswith(("analysis", "core", "joblib", "prompt_toolkit")):
                raise AssertionError(f"Import innecesario para ayuda: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded), self.assertRaises(SystemExit) as error:
            self.invoke(["--help"])
        self.assertEqual(error.exception.code, 0)

    def test_models_lists_and_never_starts_live(self):
        module = types.ModuleType("analysis.live_system")
        module.BasketballLiveSystem = MagicMock()
        system = module.BasketballLiveSystem.return_value
        system.load_models.return_value = True
        module.run_basketball_live_system = AsyncMock()
        code, _ = self.invoke(["--models"], {"analysis.live_system": module})
        self.assertEqual(code, 0)
        system.show_available_models.assert_called_once_with()
        module.run_basketball_live_system.assert_not_called()

    def test_missing_models_reports_failure(self):
        module = types.ModuleType("analysis.live_system")
        module.BasketballLiveSystem = MagicMock()
        system = module.BasketballLiveSystem.return_value
        system.load_models.return_value = False
        code, _ = self.invoke(["--models"], {"analysis.live_system": module})
        self.assertEqual(code, 1)
        system.show_available_models.assert_not_called()

    def test_live_delegates_without_a_mode_prompt(self):
        module = types.ModuleType("analysis.live_system")
        module.run_basketball_live_system = AsyncMock()
        with patch("builtins.input", side_effect=AssertionError("No hay menú de modos")):
            code, _ = self.invoke(["--live"], {"analysis.live_system": module})
        self.assertEqual(code, 0)
        module.run_basketball_live_system.assert_awaited_once_with()

    def test_train_remains_an_explicit_separate_command(self):
        training = types.ModuleType("core.training")
        training.train_models_by_league = MagicMock()
        code, _ = self.invoke(["--train"], {"config": config_stub, "core.training": training})
        self.assertEqual(code, 0)
        training.train_models_by_league.assert_called_once_with()

    def test_conflicting_commands_are_rejected(self):
        for args in (["--train", "--live"], ["--models", "--live"], ["--models", "--train"]):
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                self.invoke(args)
            self.assertEqual(error.exception.code, 2)


class LiveFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.system = live_system.BasketballLiveSystem()
        self.model = {"league_name": "NBA", "team_names": ["Home", "Away"],
                      "historical_df": [], "features_used": ["feature"]}
        self.system.models = {"NBA": self.model}
        self.game = {"id": "123", "home_team": "Home", "away_team": "Away",
                     "home_score": 60, "away_score": 55, "status": "Halftime"}
        self.processor = MagicMock()
        self.processor.initialize = AsyncMock(return_value=True)
        self.processor.get_available_games = AsyncMock(return_value=[self.game])
        self.processor.monitor_game = AsyncMock()
        self.processor.cleanup = AsyncMock()
        self.processor.client.last_discovery_error = None
        transport = types.ModuleType("core.live_api")
        transport.LiveProcessor = MagicMock(return_value=self.processor)
        self.factory = transport.LiveProcessor
        odds = types.ModuleType("analysis.odds_api")
        odds.OddsApiClient = MagicMock()
        self.odds = odds.OddsApiClient.return_value.get_odds_for_teams
        self.odds.return_value = None
        self.modules = patch.dict(sys.modules, {"core.live_api": transport, "analysis.odds_api": odds})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    async def run_flow(self, answers):
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers) as inputs, \
                patch.object(live_system.sys.stdin, "isatty", return_value=False), \
                patch.object(live_system.random, "randint", return_value=50), redirect_stdout(output):
            await self.system.run_live()
        return inputs, output.getvalue()

    async def test_starts_from_league_and_preserves_selected_game_and_line(self):
        inputs, output = await self.run_flow(["1", "1", "238.5"])
        self.assertIn("Selecciona liga", inputs.call_args_list[0].args[0])
        self.assertNotIn("Selecciona una opción", output)
        self.factory.assert_called_once_with(self.model, headless=True)
        self.processor.monitor_game.assert_awaited_once_with(
            game_id="123", game_info=self.game, update_interval=50, bookie_line=238.5)
        self.processor.cleanup.assert_awaited_once_with()
        await self.system.cleanup()
        self.processor.cleanup.assert_awaited_once_with()  # Sin cierre doble.

    async def test_automatic_odds_need_no_extra_prompt(self):
        self.odds.return_value = 242.5
        inputs, _ = await self.run_flow(["1", "1"])
        self.assertEqual(inputs.call_count, 2)
        self.assertEqual(self.processor.monitor_game.await_args.kwargs["bookie_line"], 242.5)

    async def test_empty_list_exits_without_asking_for_id(self):
        self.processor.get_available_games.return_value = []
        inputs, output = await self.run_flow(["1"])
        self.assertEqual(inputs.call_count, 1)
        self.assertNotIn("ID", output)
        self.processor.monitor_game.assert_not_awaited()
        self.processor.cleanup.assert_awaited_once()

    async def test_discovery_error_still_cleans_up(self):
        self.processor.get_available_games.return_value = None
        self.processor.client.last_discovery_error = "Acceso denegado"
        _, output = await self.run_flow(["1"])
        self.assertIn("Acceso denegado", output)
        self.processor.monitor_game.assert_not_awaited()
        self.processor.cleanup.assert_awaited_once()

    async def test_initialization_failure_cleans_up_once(self):
        self.processor.initialize.return_value = False
        await self.run_flow(["1"])
        self.processor.get_available_games.assert_not_awaited()
        self.processor.cleanup.assert_awaited_once()
        self.assertIsNone(self.system.current_processor)

    async def test_cancel_league_does_not_initialize_client(self):
        await self.run_flow([EOFError()])
        self.factory.assert_not_called()

    async def test_cancel_game_cleans_up(self):
        await self.run_flow(["1", KeyboardInterrupt()])
        self.processor.cleanup.assert_awaited_once()
        self.processor.monitor_game.assert_not_awaited()

    async def test_cancel_line_cleans_up(self):
        await self.run_flow(["1", "1", EOFError()])
        self.processor.cleanup.assert_awaited_once()
        self.processor.monitor_game.assert_not_awaited()

    async def test_monitor_error_does_not_leave_client_open(self):
        self.processor.monitor_game.side_effect = RuntimeError("Fallo del monitor")
        with self.assertRaisesRegex(RuntimeError, "Fallo del monitor"):
            await self.run_flow(["1", "1", ""])
        self.processor.cleanup.assert_awaited_once()
        self.assertIsNone(self.system.current_processor)

    async def test_back_refreshes_list_reuses_connection_and_new_line(self):
        second = dict(self.game, id="456", home_team="Other")
        self.processor.get_available_games.side_effect = [[self.game], [second]]
        with patch.object(self.system, "_monitor_with_navigation",
                          AsyncMock(side_effect=[True, False])) as navigation:
            await self.run_flow(["1", "1", "238.5", "1", "220.5"])
        self.factory.assert_called_once()
        self.processor.initialize.assert_awaited_once()
        self.assertEqual(self.processor.get_available_games.await_count, 2)
        self.assertEqual(navigation.await_args_list[1].kwargs["game_id"], "456")
        self.assertEqual(navigation.await_args_list[1].kwargs["bookie_line"], 220.5)
        self.processor.cleanup.assert_awaited_once()

    async def test_navigation_cancels_wait_and_restores_terminal(self):
        for command, back in [("v", True), ("Q", False), ("\x03", False)]:
            with self.subTest(command=command):
                stopped = asyncio.Event()

                async def waiting(**kwargs):
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()

                self.processor.monitor_game.side_effect = waiting
                # Como Win32Input: close y raw_mode, sin protocolo with propio.
                keyboard = types.SimpleNamespace(
                    read_keys=MagicMock(side_effect=[[], [types.SimpleNamespace(data=command)]]),
                    raw_mode=MagicMock(return_value=MagicMock()),
                    close=MagicMock(),
                )
                inputs = types.ModuleType("prompt_toolkit.input")
                inputs.create_input = MagicMock(return_value=keyboard)
                with patch.dict(sys.modules, {"prompt_toolkit.input": inputs}), \
                        patch.object(live_system.sys.stdin, "isatty", return_value=True), \
                        redirect_stdout(io.StringIO()):
                    result = await self.system._monitor_with_navigation(self.processor, game_id="123")
                self.assertEqual(result, back)
                self.assertTrue(stopped.is_set())
                keyboard.raw_mode.return_value.__exit__.assert_called_once()
                keyboard.close.assert_called_once_with()
                self.processor.cleanup.assert_not_awaited()

    async def test_keyboard_closes_when_monitor_finishes_or_fails(self):
        for error in (None, RuntimeError("Fallo del monitor")):
            with self.subTest(error=error):
                self.processor.monitor_game.side_effect = error
                keyboard = types.SimpleNamespace(
                    read_keys=MagicMock(return_value=[]),
                    raw_mode=MagicMock(return_value=MagicMock()),
                    close=MagicMock(),
                )
                inputs = types.ModuleType("prompt_toolkit.input")
                inputs.create_input = MagicMock(return_value=keyboard)
                with patch.dict(sys.modules, {"prompt_toolkit.input": inputs}), \
                        patch.object(live_system.sys.stdin, "isatty", return_value=True), \
                        redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, "Fallo del monitor"):
                            await self.system._monitor_with_navigation(self.processor, game_id="123")
                    else:
                        self.assertFalse(await self.system._monitor_with_navigation(self.processor, game_id="123"))
                keyboard.raw_mode.return_value.__exit__.assert_called_once()
                keyboard.close.assert_called_once_with()

    async def test_monitor_resets_previous_match_state(self):
        tree = ast.parse((ROOT / "core/live_api.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LiveProcessor")
        method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "monitor_game")
        namespace = {"Dict": dict, "time": time, "datetime": None, "asyncio": asyncio}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "monitor", "exec"), namespace)
        processor = MagicMock()
        processor.last_prediction = 999
        processor.last_q_scores = {"old": 1}
        processor.last_game_info = {"old": 1}
        processor.last_stats = {"old": 1}
        processor.update_count = 90
        mode = types.ModuleType("analysis.live_mode")
        mode.build_pre_game_cache = MagicMock(return_value={})
        with patch.dict(sys.modules, {"analysis.live_mode": mode}), redirect_stdout(io.StringIO()):
            await namespace["monitor_game"](processor, "456", self.game, max_updates=0)
        for field in ("last_prediction", "last_q_scores", "last_game_info", "last_stats"):
            self.assertIsNone(getattr(processor, field))
        self.assertEqual(processor.update_count, 0)

    async def test_cancel_bridge_request_keeps_pairing_and_allows_next_request(self):
        bridge_module = load_source("bridge_navigation_test", "core/browser_bridge.py")
        bridge = bridge_module.BrowserBridge()
        bridge.server = object()  # Sin socket ni red.
        token = bridge.token
        request = asyncio.create_task(bridge.request("/api/v1/event/123"))
        await asyncio.sleep(0)
        self.assertIsNotNone(bridge.pending)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.assertIsNone(bridge.pending)
        request = asyncio.create_task(bridge.request("/api/v1/event/456"))
        await asyncio.sleep(0)
        bridge.pending["result"] = {"data": {"event": {"id": 456}}}
        bridge.pending["done"].set()
        self.assertEqual(await request, {"event": {"id": 456}})
        self.assertEqual(bridge.token, token)
        self.assertIsNotNone(bridge.server)


if __name__ == "__main__":
    unittest.main()
