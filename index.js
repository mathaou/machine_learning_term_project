// import {textToBinary} from "./helper";

const BROKERPORT = 1883;
const BROKERURL = "localhost";
const mqtt = require('mqtt');
let client = null;
let client_in = "/hand/client";
let server_out = "/hand/server";

let main = function() {
    client = mqtt.connect(BROKERURL, {"clientId": "client", "port": BROKERPORT, "protocol": "MQTT"});

    handleProcessEvents();
    linkHandlers();

    client.publish(server_out, "from node");
}

function linkHandlers() {
    client.on("connect", onConnect);
    client.on("disconnect", onDisconnect);
    client.on("message", onMessage);
    client.on("error", onError);
    client.on("packetsend", onPacketSend);
}

function onConnect(connack) {
    console.log("Connecting...: " + JSON.stringify(connack));
    client.subscribe(client_in)
}

function onDisconnect(packet) {
    console.log("Disconnecting...: " + JSON.stringify(packet));
}

function onMessage(topic, message) {
    console.log(topic + ": " + message);
    if(topic == client_in){
        console.log("FROM PYTHON");
    }
}

function onError(error) {
    console.log("Error raised...: "+ JSON.stringify(error));
}

function onPacketSend(packet) {
    console.log("Client sent packet to server: " + JSON.stringify(packet));
}

function handleProcessEvents() {
    process.on("exit", () => HandleAppExit(null, {
        cleanup: true
    }))
    
    process.on("SIGINT", () => HandleAppExit(null, {
        exit: true
    }))
    
    process.on("uncaughtException", () => HandleAppExit(null, {
        exit: true
    }))
}

const HandleAppExit = (options, err) => {
    if(err){
        console.log(err.stack);
    }
    
    if(options.cleanup){
        client.end();
    }
    if(options.exit){
        process.exit();
    }
}

main();