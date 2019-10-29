const BROKERPORT = 1883;
const BROKERURL = "localhost";
const mqtt = require('mqtt');
const async_mqtt = require('async-mqtt');
let client = null, asyncClient = null;
let client_in = "/hand/client";
let server_out = "/hand/server";
var port = process.env.PORT || 8080;


const express = require("express");

const app = express();

app.use(express.static('public'));
app.use('/jquery', express.static(__dirname + '/node_modules/jquery/dist/'));

var bodyParser = require('body-parser');
app.use(bodyParser.json()); // support json encoded bodies
app.use(bodyParser.urlencoded({ extended: true })); // support encoded bodies

// const fetch = require("node-fetch");

var input = null, output = 'Select a hand and see how accurate our algorithm is!';

app.get('/', (req, res) => {
    res.render('index', {
        output: output
    });
});

app.post('/hand', (req, res) => {
    input = {
        "hand":[
            [req.body.card1, req.body.card1Suit],
            [req.body.card2, req.body.card2Suit],
            [req.body.card3, req.body.card3Suit],
            [req.body.card4, req.body.card4Suit],
            [req.body.card5, req.body.card5Suit]
        ]
    };

    output = queryNetwork(input, res);
});

app.set('view engine', 'pug');

const server = app.listen(port, () => {
    console.log(`RUNNING ON PORT ${server.address().port}`);
});

const queryNetwork = async (payload, res) => {
    asyncClient.on("message", (topic, message) => {
        if(topic == client_in){
            output = message;
            res.redirect("/");
        }
    });
    if (client === null) {
        console.log("error initializing mqtt client...");
    } else {
        await asyncClient.publish(server_out, JSON.stringify(payload));
    }
};

let main = () => {
    client = mqtt.connect(BROKERURL, {"clientId": "client", "port": BROKERPORT, "protocol": "MQTT"});

    asyncClient = new async_mqtt.AsyncClient(client);

    handleProcessEvents();
    linkHandlers();
}

function linkHandlers() {
    asyncClient.on("connect", onConnect);
    asyncClient.on("disconnect", onDisconnect);
    asyncClient.on("error", onError);
    asyncClient.on("packetsend", onPacketSend);
}

function onConnect(connack) {
    console.log("Connecting...: " + JSON.stringify(connack));
    client.subscribe(client_in)
}

function onDisconnect(packet) {
    console.log("Disconnecting...: " + JSON.stringify(packet));
}

function onError(error) {
    console.log("Error raised...: "+ JSON.stringify(error));
}

function onPacketSend(packet) {
    // console.log("Client sent packet to server: " + JSON.stringify(packet));
}

function handleProcessEvents() {
    process.on("exit", () => HandleAppExit(null, {
        cleanup: true
    }));
    
    process.on("SIGINT", () => HandleAppExit(null, {
        exit: true
    }));
    
    process.on("uncaughtException", () => HandleAppExit(null, {
        exit: true
    }));
}

const HandleAppExit = (options, err) => {
    if(err){
        process.exit();
    }
    if(options.cleanup){
        client.end();
        asyncClient.end();
    }
    if(options.exit){
        process.exit();
    }
}

main();