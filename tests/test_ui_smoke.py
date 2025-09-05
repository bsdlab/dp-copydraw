import pytest

try:
    from copydraw.copydraw import CopyDraw
except ImportError as e:
    pytest.fail(f"Failed to import CopyDrawTask: {e}. Check dependencies and sys.path.")


def test_app_initialization():

    paradigm = CopyDraw(
        data_dir=".",
        script_dir=".",
        screen_ix=0,
        serial_nr=None,
    )

    paradigm.init_session()
    paradigm.set_block_settings()
    try:
        paradigm.init_block()
    except RuntimeError as e:
        print(
            f"Encountered RuntimeError during block initialization - OK if on CI: {e}"
        )
        pass

    # at this stage we are not sure if drawing etc work. But at least the python
    # env should be ready for trying it out
