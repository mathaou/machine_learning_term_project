const BROKERPORT = 1883;
const BROKERURL = "localhost";
const mqtt = require('mqtt');
let client = null;
let client_in = "hand/client";
let server_out = "hand/server";
var port = process.env.PORT || 8080;

const express = require("express");
const app = express();
var ws = require('ws')

app.use(express.static('public'));
app.use(express.static('layoutit/src/52 card'))
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
    res.end();
});

const WebSocketServer = require('ws').Server;
wss = new WebSocketServer({port: 40510});
wss.on('connection', (ws) => {
    ws.on('message', (message) => {
        ws.send(JSON.stringify(message));
    });
});

app.post('/hand', (req, res) => {
    output = queryNetwork(
        `${req.body.card1},${req.body.card1Suit | 4},${req.body.card2},${req.body.card2Suit | 4},${req.body.card3},${req.body.card3Suit | 4},${req.body.card4},${req.body.card4Suit | 4},${req.body.card5},${req.body.card5Suit | 4}`);
    res.redirect('/');
});

app.set('view engine', 'pug');

const server = app.listen(port, () => {
    console.log(`RUNNING ON PORT ${server.address().port}`);
});

const queryNetwork = (payload) => {
    if (client === null) {
        console.log("error initializing mqtt client...");
    } else {
        client.publish(server_out, payload);
    }
};

let main = () => {
    client = mqtt.connect(BROKERURL, {"clientId": "client", "port": BROKERPORT, "protocol": "MQTT"});
    client.on("message", (topic, message) => {
        // console.log(Buffer.from(message).toString())
        if(message != "Connect") {
            let ev = JSON.parse(message);
            let dic = {
                0: "Nothing in hand; not a recognized poker hand",
                1: "One pair; one pair of equal ranks within five cards",
                2: "Two pairs; two pairs of equal ranks within five cards",
                3: "Three of a kind; three equal ranks within five cards",
                4: "Straight; five cards, sequentially ranked with no gaps",
                5: "Flush; five cards with the same suit",
                6: "Full house; pair + different rank three of a kind",
                7: "Four of a kind; four equal ranks within five cards",
                8: "Straight flush; straight + flush",
                9: "Royal flush; {Ace, King, Queen, Jack, Ten} + flush"
            };
            let res = "The top 5 results from our network were...\n\n";
            let i = 1
            
            ev['result'].forEach((element, index, array) => {
                res += `${i}: ${dic[element.classifier]}\n\tProbability: ${element.error}\n`;
                i++;
            });
            if(ev['expected'] != 0){
                res += `\nDid we get it right?\n\nExpected: ${dic[ev['expected']]}`;
            } else {
                res += `\nDid we get it right?`;
            }
            output = res;
            wss.clients.forEach((c) => {
                if(c != ws){
                    c.send(res);
                }
            });
            // output = 'Select a hand and see how accurate our algorithm is!';
        }
    });

    linkHandlers();
    handleProcessEvents();
}

function linkHandlers() {
    client.on("connect", onConnect);
    client.on("disconnect", onDisconnect);
    client.on("error", onError);
    client.on("packetsend", onPacketSend);
}

function onConnect(connack) {
    console.log("Connecting...: " + JSON.stringify(connack));
    client.subscribe(client_in);
}

function onDisconnect(packet) {
    console.log("Disconnecting...: " + JSON.stringify(packet));
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
    }
    if(options.exit){
        process.exit();
    }
}

main();