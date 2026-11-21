"""The formality objective of the optimization experiments and the contrastive pairs of its CAA vector.

`formality_benchmark()` is a `PromptBenchmark` of 2,100 short messaging prompts (14 recipients, 50 messages, and 3
phrasings), scored by a RoBERTa formality ranker (`formality_score`). `PAIRS` are the formal and casual examples the
CAA vector of the truth grid and the live search is fitted on. The scripts import this module by its file name, since
the scripts directory is the first entry of `sys.path` when a script runs.
"""
import itertools

from steerability.algorithms.core.identity import versioned
from steerability.evaluation.items import PromptBenchmark

CLASSIFIER = "s-nlp/roberta-base-formality-ranker"
OBJECTIVE_HELD_OUT = 600
_PIPELINE = None

WHO = ("your best friend", "your roommate", "your cousin", "your teammate", "your neighbor", "your little brother",
       "an old classmate", "your gym buddy", "your sister", "a coworker you hang out with", "your uncle",
       "your bandmate", "your lab partner", "your carpool buddy")
WHAT = ("you will be late to dinner", "you found their lost keys", "the game got moved to Sunday",
        "you can't make the party", "the movie was amazing", "you need to borrow their charger",
        "the pizza is on its way", "you finally passed your driving test", "the concert tickets sold out",
        "you are running ten minutes behind", "their package arrived at your place", "the trip is still on",
        "you adopted a puppy", "you broke your phone screen", "the wifi password changed",
        "you saw their favorite band at the mall", "the barbecue starts at six", "you lost the bet",
        "you are bringing snacks", "the bus is cancelled", "you got the job", "the beach day is rained out",
        "you started a new hobby", "the team won the final", "you want to grab coffee", "the rent is due Friday",
        "you finished the book they lent you", "the store was closed", "you are cooking tonight",
        "the meeting was moved", "you need a ride tomorrow", "the dog ate your homework", "you booked the cabin",
        "the bike needs new tires", "you are running a marathon", "the cake turned out great",
        "the printer is broken again", "you saw a famous chef", "you are moving next month", "the exam got easier",
        "the lights went out", "you are stuck in traffic", "the parade starts at noon", "you lost your wallet",
        "the garden is blooming", "you are hosting game night", "the train is delayed", "you made new friends",
        "the soup is ready", "you won a raffle")
PHRASINGS = (
    "Send a quick text to {who} letting them know {what}.",
    "Write a short message to {who} to say {what}.",
    "Leave a voicemail for {who} telling them {what}.",
)
PROMPTS = {
    f"p{index:04d}": phrasing.format(who=who, what=what)
    for index, (phrasing, who, what) in enumerate(itertools.product(PHRASINGS, WHO, WHAT))
}

FORMAL = [
    "I regret to inform you that I will be delayed this evening.",
    "Please be advised that the meeting has been rescheduled to Thursday.",
    "I would be grateful if you could return the charger at your convenience.",
    "I am pleased to confirm that the arrangements remain unchanged.",
    "Kindly note that the event will commence at six o'clock.",
    "I wish to express my sincere appreciation for your assistance.",
    "It is with regret that I must decline your kind invitation.",
    "I would like to inform you that your parcel has been delivered.",
]
CASUAL = [
    "hey gonna be late tonight lol, sorry!!",
    "yo the meeting got moved to thursday btw",
    "can u bring back my charger whenever? thx!",
    "yep we're still on, super pumped!!",
    "party starts at 6, don't be late haha",
    "dude thanks so much, you're awesome!",
    "ugh can't make it, sorry guys",
    "hey ur package showed up at my place!",
]
TASKS = [
    "Tell a friend you will be late tonight.", "Tell a friend the meeting moved to Thursday.",
    "Ask a friend to return your charger.", "Confirm the plans are still on.", "Say when the party starts.",
    "Thank a friend for their help.", "Decline an invitation.", "Tell a friend their package arrived.",
]
PAIRS = {"prompts": TASKS, "positives": FORMAL, "negatives": CASUAL}


@versioned("1")
def formality_score(response, row):
    """The formality score of a response from a RoBERTa formality ranker, in [0, 1]."""
    global _PIPELINE
    if _PIPELINE is None:
        import torch
        from transformers import pipeline

        _PIPELINE = pipeline("text-classification", model=CLASSIFIER, top_k=None, truncation=True,
                             device=0 if torch.cuda.is_available() else -1)
    if not response.strip():
        return 0.0
    scores = {result["label"].lower(): result["score"] for result in _PIPELINE(response[:2000])[0]}
    return float(scores["formal"])


def formality_benchmark() -> PromptBenchmark:
    """The formality objective, with completions of at most 96 tokens."""
    return PromptBenchmark("formality", PROMPTS, formality_score, generate={"max_tokens": 96})
