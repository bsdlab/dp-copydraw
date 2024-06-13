#!/usr/bin/env python

import time
import serial

from pylsl import StreamInfo, StreamOutlet


class VPPort(object):

    """ Class for interacting with the virtual serial
        port provided by the BV TriggerBox

    """

    def __init__(self, serial_nr, pulsewidth=0.01):
        """ Open the port at the given serial_nr

        Parameters
        ----------

        serial_nr : str
            Serial number of the trigger box as can be read under windows hw manager
        pulsewidth : float
            Seconds to sleep between base and final write to the PPort

        """
        try:
            self.port = serial.Serial(serial_nr)
        except Exception as ex:     # if trigger box is not available at given serial_nr
            print("?" * 80 + "\n\n\n\n"+ f"??? Could not connect to serial, will continue with dummy: {ex=}"+ "\n\n\n" + "?" * 80)
            self.create_dummy(serial_nr)
            raise ex

        self.pulsewidth = pulsewidth

        self.stream_info = StreamInfo(
            name='CopyDrawParadigmMarkerStream',
            type='Markers', channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id='myuiCopyDrawParadigmMarker')
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

        # Set a base value as trigger will only emit once change from base is written
        self.port.write([0])
        time.sleep(self.pulsewidth)

        return self.port.write(data)

    def __del__(self):
        """Destructor to close the port"""
        print("Closing serial port connection")
        if self.port is not None:
            self.port.close()

    def create_dummy(self, serial_nr):
        """Initialize a dummy version - used for testing"""
        print("-"*80 +
              "\n\nInitializing DUMMY VPPORT\nSetup for regular VPPORT at" +
              f" at {serial_nr} failed \n No device present?\n" + "-"*80)

        self.port = None
        self.write = self.dummy_write

    def dummy_write(self, data):
        """Overwriting the write to pp"""
        print(f"PPort would write data: {data}")
