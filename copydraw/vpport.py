import serial
from pylsl import StreamInfo, StreamOutlet

from copydraw.utils.clock import sleep_s


class VPPort(object):
    """Class for interacting with the virtual serial
    port provided by the BV TriggerBox

    """

    def __init__(self, serial_nr, pulsewidth=0.01):
        """Open the port at the given serial_nr

        Parameters
        ----------

        serial_nr : str
            Serial number of the trigger box as can be read under windows hw manager
        pulsewidth : float
            Seconds to sleep between base and final write to the PPort

        """
        try:
            self.port = serial.Serial(serial_nr)
        except (
            serial.SerialException
        ):  # if trigger box is not available at given serial_nr (the DUMMY is used for debugging)
            print(
                "?" * 80
                + "\n\n\n\n"
                + "??? Could not connect to serial, will continue with dummy"
                + "\n\n\n"
                + "?" * 80
            )
            self.create_dummy(serial_nr)

        self.pulsewidth = pulsewidth

        # have different writer instances for different hardware triggers
        self.serial_write = self.utf8_write  # for the maastricht branch

        self.stream_info = StreamInfo(
            name="CopyDrawParadigmMarkerStream",
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="int32",  # think of changing to int64 as there might be an issue reading int32 on windows
            source_id="myuiCopyDrawParadigmMarker",
        )
        self.stream_outlet = StreamOutlet(self.stream_info)

    def write(self, data):
        """

        Parameters
        ----------

        data:  list of int(s), byte or bytearray
            data to be written to the port

        Returns
        -------
        byteswritten : int
            number of bytes written

        """
        # Send to LSL Outlet
        self.stream_outlet.push_sample(data)

        ret = self.serial_write(data)

        return ret

    def utf8_write(self, data: int | list[int]) -> int:
        """By default, we wrote lists of int"""
        data = [data] if isinstance(data, int) else data
        for d in data:
            ret = self.port.write(bytes(chr(d), encoding="utf8"))
        return ret

    def bv_trigger_box_write(self, data) -> int:

        self.port.write([0])
        sleep_s(self.pulsewidth)
        ret = self.port.write(data)

        return ret

    def __del__(self):
        """Destructor to close the port"""
        print("Closing serial port connection")
        if self.port is not None:
            self.port.close()

    def create_dummy(self, serial_nr):
        """Initialize a dummy version - used for testing"""
        print(
            "-" * 80
            + "\n\nInitializing DUMMY VPPORT\nSetup for regular VPPORT at"
            + f" at {serial_nr} failed \n No device present?\n"
            + "-" * 80
        )

        self.port = None
        self.write = self.dummy_write

    def dummy_write(self, data):
        """Overwriting the write to pp"""
        print(f"PPort would write data: {data}")
        self.stream_outlet.push_sample(data)
