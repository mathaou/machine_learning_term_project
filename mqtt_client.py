#!/usr/bin/env python3

import paho.mqtt.client as mqtt
import json

import time


class MQTTClient:

	broker_url = "localhost"
	broker_port = 1883

	mqtt = None

	accumulated_data = bytearray(b'')

	def __init__(self):
		self.mqtt = mqtt.Client("client")
		self.mqtt.connect(self.broker_url, self.broker_port)

		self.mqtt.on_connect = self.on_connect
		self.mqtt.on_disconnect = self.on_disconnect
		self.mqtt.on_log = self.on_log
		self.mqtt.on_message = self.on_message

		self.mqtt.subscribe("/hand")
	def handle_loop(self):
		self.mqtt.loop(.1)

	def process_payload(self, dict):
		return json.dumps(dict)

	def on_message(self, client, userdata, msg):
		print("{}: {}".format(msg.topic, str(msg.payload.decode("utf-8"))))
		if msg.topic == "/hand":
			print("GOTTEM")

	def on_connect(self, client, userdata, flags, rc):
		if rc == 0:
			print("Established connection...")
		else:
			print("No connection established, returned error code {}...".format(rc))

	def on_disconnect(self, client, userdata, flags, rc = 0):
		print("Disconnected with result code {}".format(rc))

	def on_log(self, client, userdata, level, buf):
		print("LOG: {}".format(buf))

def main():
	m = MQTTClient()
	while True:
		m.handle_loop()

if __name__ == "__main__":
	main()
