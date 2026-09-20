from __future__ import annotations

import random
from typing import Any, Dict, List


COUNTERFACTUAL_EVENTS = [
    "Caldor Memory Study",
    "Veylin Archive Trial",
    "Northbridge Recall Project",
    "Meridian Listening Survey",
    "Asterwell Habit Registry",
    "Kavren Pattern Workshop",
    "Solenne Choice Inventory",
    "Tarnwick Preference Census",
    "Eldora Signal Study",
    "Briarfen Daily Log Project",
    "Nivara Attention Trial",
    "Coromar Familiarity Survey",
]

COUNTERFACTUAL_FIRST_NAMES = [
    "Mira",
    "Soren",
    "Talia",
    "Niko",
    "Elian",
    "Vera",
    "Kellan",
    "Rina",
    "Dalen",
    "Liora",
    "Marek",
    "Selene",
    "Jorin",
    "Amara",
    "Tovan",
    "Nadia",
    "Iven",
    "Corin",
    "Lysa",
    "Pavel",
    "Anika",
    "Rovan",
    "Kaia",
    "Silas",
]

COUNTERFACTUAL_LAST_NAMES = [
    "Solen",
    "Varic",
    "Morne",
    "Kavell",
    "Trin",
    "Halev",
    "Orin",
    "Velar",
    "Caspel",
    "Damar",
    "Renn",
    "Luro",
    "Fenric",
    "Aven",
    "Sareth",
    "Korin",
    "Malve",
    "Tessel",
    "Norik",
    "Varyn",
    "Delos",
    "Cairn",
    "Bren",
    "Ostel",
]

COUNTERFACTUAL_ATTRIBUTES = {
    "color": [
        ("purple", "it reminded them of old arcade machines"),
        ("teal", "it matched the paint on a childhood bicycle"),
        ("amber", "it looked like light passing through a bus window"),
        ("silver", "it made them think of rain on train tracks"),
        ("maroon", "it resembled the cover of a notebook they kept"),
        ("indigo", "it recalled the evening sky above a quiet station"),
    ],
    "dessert": [
        ("lemon tart", "the sharp flavor helped them remember summer markets"),
        ("rice pudding", "the texture reminded them of family holidays"),
        ("blackberry crumble", "the smell matched a bakery near their old school"),
        ("vanilla custard", "it was served at the first workshop they attended"),
        ("peach sorbet", "the cold taste made long meetings feel shorter"),
        ("cocoa flan", "it tasted like a cafe they visited after exams"),
    ],
    "drink": [
        ("mint tea", "the scent helped them concentrate during interviews"),
        ("pear soda", "the fizz reminded them of weekend picnics"),
        ("cold barley coffee", "the bitterness kept them alert during late sessions"),
        ("ginger lemonade", "the spice made the survey room feel less formal"),
        ("rosewater milk", "it matched a recipe from a handwritten card"),
        ("apple kefir", "it was the first drink offered at orientation"),
    ],
    "instrument": [
        ("mandolin", "the bright notes sounded like a festival they remembered"),
        ("clarinet", "its tone reminded them of morning radio programs"),
        ("marimba", "the wooden sound felt calm and orderly"),
        ("accordion", "it echoed music from a ferry terminal"),
        ("viola", "the lower strings seemed steady and familiar"),
        ("kalimba", "the small keys were easy to carry between sessions"),
    ],
    "flower": [
        ("dahlia", "its layered petals looked like folded paper"),
        ("hyacinth", "the scent reminded them of a neighbor's balcony"),
        ("iris", "the shape matched a drawing in their old diary"),
        ("camellia", "it bloomed near the clinic entrance"),
        ("zinnia", "its bright petals stood out in cloudy weather"),
        ("anemone", "it appeared on the first form they completed"),
    ],
    "weekday": [
        ("Tuesday", "the study meetings were quietest on that day"),
        ("Thursday", "it gave them enough time to prepare notes"),
        ("Saturday", "the city felt slower and easier to notice"),
        ("Monday", "it made the week feel organized from the start"),
        ("Wednesday", "it sat evenly between busy obligations"),
        ("Friday", "it made routine tasks feel lighter"),
    ],
    "season": [
        ("autumn", "the cooler air made walks after the sessions pleasant"),
        ("spring", "new leaves appeared near the study building"),
        ("winter", "quiet streets helped them focus on small details"),
        ("summer", "long evenings made the workshops feel unhurried"),
    ],
    "snack": [
        ("sesame crackers", "the crisp sound made breaks feel deliberate"),
        ("dried apricots", "the sweetness reminded them of a train platform kiosk"),
        ("salted almonds", "they were easy to share during group tasks"),
        ("oat biscuits", "they tasted like the waiting room refreshments"),
        ("fig rolls", "the filling reminded them of a shop near the archive"),
        ("plantain chips", "the salt helped them stay awake after lunch"),
    ],
}


def _counterfactual_person_name(rng: random.Random, used_names: set[str]) -> str:
    for _ in range(1000):
        first = rng.choice(COUNTERFACTUAL_FIRST_NAMES)
        last = rng.choice(COUNTERFACTUAL_LAST_NAMES)
        name = f"{first} {last}"
        if name not in used_names:
            used_names.add(name)
            return name
    name = f"{rng.choice(COUNTERFACTUAL_FIRST_NAMES)} {rng.choice(COUNTERFACTUAL_LAST_NAMES)} {len(used_names)}"
    used_names.add(name)
    return name


def build_counterfactual_gold_passage(
    *,
    person: str,
    event: str,
    attribute_type: str,
    answer: str,
    reason: str,
) -> str:
    return (
        f"{person} is a fictional participant in the {event}.\n"
        f"{person}'s favorite {attribute_type} is {answer}.\n"
        f"{person} chose {answer} because {reason}."
    )


def build_counterfactual_hotpotqa_rows(*, count: int = 400, seed: int = 13) -> List[Dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    used_names: set[str] = set()
    attr_types = sorted(COUNTERFACTUAL_ATTRIBUTES)
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        attribute_type = attr_types[idx % len(attr_types)]
        value, reason = rng.choice(COUNTERFACTUAL_ATTRIBUTES[attribute_type])
        person = _counterfactual_person_name(rng, used_names)
        event = rng.choice(COUNTERFACTUAL_EVENTS)
        question = f"What is {person}'s favorite {attribute_type}?"
        base_gold_passage = build_counterfactual_gold_passage(
            person=person,
            event=event,
            attribute_type=attribute_type,
            answer=value,
            reason=reason,
        )
        rows.append(
            {
                "id": f"cf_hotpotqa_{seed}_{idx:04d}",
                "question": question,
                "answer": value,
                "person": person,
                "event": event,
                "attribute_type": attribute_type,
                "reason": reason,
                "base_gold_passage": base_gold_passage,
                "gold_passage": base_gold_passage,
            }
        )
    return rows


def build_counterfactual_qa_rows(*, count: int = 400, seed: int = 13) -> List[Dict[str, Any]]:
    return build_counterfactual_hotpotqa_rows(count=count, seed=seed)
