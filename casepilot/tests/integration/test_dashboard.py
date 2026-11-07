def test_dashboard_renders_without_provider_calls():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path("dashboard/research_app.py").resolve()).run(timeout=45)
    assert not app.exception
    assert app.title[0].value == "CasePilot Lite"
    saved = next((widget for widget in app.selectbox if widget.label == "Saved run"), None)
    if saved and saved.options:
        button = next(widget for widget in app.button if widget.label == "Open saved run")
        button.click().run(timeout=45)
        assert not app.exception
        assert any(widget.label == "Episode" for widget in app.selectbox)


def test_saved_run_query_cannot_open_arbitrary_files():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path("dashboard/research_app.py").resolve())
    app.query_params["saved_run"] = "../../.env"
    app.run(timeout=45)
    assert not app.exception
    assert "simulation_folder" not in app.session_state
