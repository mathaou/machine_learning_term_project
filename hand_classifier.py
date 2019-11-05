import json
import poker_hand_network as network

class Hand():
    def __init__(self, arr):
        if len(arr) != 2:
            raise Exception("Incoming data not of length 2")
        self.rank = int(arr[0])
        try:
            self.suit = int(arr[1])
            if self.suit is None:
                self.suit = 1
        except Exception:
            self.suit = 1
        # print('{0}'.format(self.__str__()))

    def __str__(self):
        suit = None
        if self.suit == 1:
            suit = "Hearts"
        elif self.suit == 2:
            suit = "Spades"
        elif self.suit == 3:
            suit = "Diamonds"
        elif self.suit == 4:
            suit = "Clubs"

        rank = None
        if self.rank > 9:
            arr = ["Jack", "Queen", "King", "Ace"]
            rank = arr[self.rank - 10]
        else:
            rank = self.rank + 1

        return '{0} of {1}'.format(rank, suit)


class HandClassifier():

    def __init__(self, mqtt = None, payload=None):
        print("Reading in payload...")
        self.payload = payload
        self.hand = []
        data = json.loads(self.payload)
        # print(data)
        for card in data["hand"]:
            # print(card)
            self.hand.append(Hand(card))
            print(card.__str__())
        self.ann = network.PokerHandANN(self.hand, mqtt)

    def __str__(self):
        ret = "Hand contains: "
        for card in self.hand:
            ret += '{0}, '.format(card.__str__())
        
        return ret[0:len(ret) - 2]
