import time

import pylsl

from copydraw.main import init_paradigm
from copydraw.vpport import VPPort


def test_regular_init():
    parad = init_paradigm()
    parad.init_session("test")

    assert parad.names["session"] == "test"


def test_vpport(capfd):
    vpp = VPPort("UNAVAILABLE_COMID")
    sname = vpp.stream_info.name()

    # stdout_msg = capfd.readoutstd()       # TODO: implement checking of stdout

    inlet_info = pylsl.resolve_stream("name", sname)[0]
    inlet = pylsl.StreamInlet(inlet_info)
    _, _ = inlet.pull_chunk()

    for i in range(5):
        vpp.write([i])
        time.sleep(0.01)

    chunk, _ = inlet.pull_chunk()
    assert chunk == [[0], [1], [2], [3], [4]]
