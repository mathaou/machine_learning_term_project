#!/usr/bin/env python3
import paho.mqtt.client as MQTT
import time
import json

import traceback
import re

class MQTTBroker():

    """Used as placeholders for all the devices."""

    mqtt = None

    server_in = "/hand/server"
    client_out = "/hand/client"

    """Location of mqtt broker."""
    broker_url = "localhost"
    broker_port = 1883

    subscription_list = [server_in]

    """Initialization of the server."""
    def __init__(self):

        self.mqtt = MQTT.Client("server")

        self.link_handlers()

        self.mqtt.connect(self.broker_url, self.broker_port)

        self.subcribe_to_list()

        self.handle_loop()

    """Handles all of the linking for mqtt method handlers"""
    def link_handlers(self):
        self.mqtt.on_connect = self.on_connect
        self.mqtt.on_disconnect = self.on_disconnect
        self.mqtt.on_log = self.on_log
        self.mqtt.on_message = self.on_message

    """Initial control structure before pass off to emulated device"""
    def handle_loop(self):
        while True:
            self.mqtt.loop(.1, 64)

    """On message handler gets called anytime self.mqtt recieves a subscription"""
    def on_message(self, client, userdata, msg):
        print("{}: {}".format(msg.topic, str(msg.payload.decode("utf-8"))))

        """Any data destined for host from client node"""
        if(msg.topic == self.server_in):
            self.mqtt.publish(self.client_out, "ass")

    """On connect handler gets called upon a connection request"""
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Established connection...")
        else:
            print("No connection established, returned error code {}...".format(rc))

    """On disconnect handler gets called upon a disconnect request"""
    def on_disconnect(self, client, userdata, flags, rc = 0):
        print("Disconnected with result code {}".format(rc))

    """Logs any error messages, kind of annoyting because it doesn't provide any information about WHERE the error came from but prevents outright crash"""
    def on_log(self, client, userdata, level, buf):
        print("LOG: {}".format(buf))

    """Helper method to just subscribe to any topic inside of a list"""
    def subcribe_to_list(self):
        for x in self.subscription_list:
            self.mqtt.subscribe(x)

def main():
    MQTTBroker()

if __name__ == "__main__":
    main()