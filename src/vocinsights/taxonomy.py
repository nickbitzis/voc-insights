"""The hidden theme taxonomy behind the synthetic feedback.

Fernhollow is a made-up company that sells Fernhollow Planner, a family calendar, lists, and notes
app on phone, tablet, and web. Each theme has phrase banks by sentiment, a planted effect on the
NPS score, and how often each source mentions it. The analysis code never reads the phrase banks:
it only sees the theme names (as the label set a product team would agree on) and labeled examples.

`NEW_ISSUE` is not in the known taxonomy. It is the release bug that appears in quarter 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SLOTS = {
    "when": [
        "every morning",
        "since last week",
        "all the time",
        "twice today",
        "every single day",
        "lately",
        "since the weekend",
        "most days",
    ],
    "dev": [
        "my phone",
        "the iPad",
        "my laptop",
        "my wife's phone",
        "the web app",
        "my tablet",
        "my Android",
        "the desktop app",
    ],
    "very": ["really", "very", "super", "so", "incredibly", "pretty", "quite"],
    "list": [
        "grocery list",
        "school calendar",
        "chore list",
        "meal plan",
        "packing list",
        "shared notes",
        "soccer schedule",
    ],
}


@dataclass(frozen=True)
class Theme:
    name: str
    label: str
    weight: float  # relative mention rate
    p_pos: float  # share of mentions that are positive
    neg: tuple[str, ...]
    pos: tuple[str, ...] = ()
    neu: tuple[str, ...] = ()
    sarcasm: tuple[str, ...] = ()  # negative meaning, positive words
    rule: str = ""  # high precision keyword regex for the rules tier
    effect_neg: float = 0.0  # planted change in 0 to 10 likelihood to recommend
    effect_pos: float = 0.0
    p_neu: float = 0.0
    affinity: dict = field(default_factory=lambda: {"nps": 1.0, "review": 1.0, "ticket": 1.0})


THEMES: tuple[Theme, ...] = (
    Theme(
        "login_problems",
        "Login problems",
        1.3,
        0.03,
        neg=(
            "I can't log in on {dev}",
            "login keeps failing {when}",
            "it logs me out {when} and the password reset email never shows up",
            "stuck on the sign in screen",
            "two factor code never arrives so I cannot get into my account",
            "keeps saying wrong password even after a reset",
            "signing in with Google just spins forever",
        ),
        pos=("logging in with face ID is quick",),
        sarcasm=("love typing my password five times a day", "great, logged out again, my favorite part of the day"),
        rule=r"\b(log ?in|sign ?in|logged out|logs me out|password|two factor|2fa)\b",
        effect_neg=-1.6,
        effect_pos=0.2,
        affinity={"nps": 0.7, "review": 1.0, "ticket": 2.6},
    ),
    Theme(
        "slow_sync",
        "Slow sync",
        1.4,
        0.04,
        neg=(
            "sync between {dev} and my laptop takes forever",
            "changes take hours to show up on {dev}",
            "the {list} does not sync until I restart the app",
            "syncing is {very} slow {when}",
            "my husband adds an event and I don't see it for ages",
            "sync is stuck on 'updating'",
        ),
        pos=("sync between devices is instant",),
        sarcasm=("nice how sync only takes half a day",),
        rule=r"\b(sync|syncing|synced|syncs)\b",
        effect_neg=-1.4,
        effect_pos=0.5,
        affinity={"nps": 1.0, "review": 1.1, "ticket": 1.6},
    ),
    Theme(
        "pricing_too_high",
        "Pricing too high",
        1.2,
        0.12,
        neg=(
            "the subscription is too expensive for what it does",
            "the price is way too high now",
            "$9.99 a month for a calendar is a lot",
            "family plan costs too much",
            "can't justify the price",
            "pricing went up and I am thinking of cancelling",
        ),
        pos=("worth every penny of the subscription", "the family plan is a good price"),
        sarcasm=("love paying more every year for the same features",),
        rule=r"\b(price|pricing|expensive|subscription|costs? too much|overpriced)\b",
        effect_neg=-1.0,
        effect_pos=0.6,
        affinity={"nps": 1.4, "review": 1.2, "ticket": 0.4},
    ),
    Theme(
        "love_the_design",
        "Love the design",
        1.3,
        0.92,
        neg=("the new look is ugly and cluttered", "the redesign made everything harder to read"),
        pos=(
            "love the design",
            "the app looks {very} clean and pretty",
            "beautiful interface, the colors are lovely",
            "gorgeous design",
            "the layout is clean and calm",
            "nicest looking planner app I have used",
        ),
        rule=r"\b(design|interface|beautiful|gorgeous|looks (?:so |very |really )?(?:clean|nice|great))\b",
        effect_neg=-0.6,
        effect_pos=0.9,
        affinity={"nps": 1.3, "review": 1.6, "ticket": 0.1},
    ),
    Theme(
        "missing_export",
        "Missing export",
        0.8,
        0.0,
        neg=(
            "there is no way to export my data",
            "please add export to CSV or PDF",
            "I need to export the {list} and can't",
            "no export option is a dealbreaker for me",
            "wish I could print or export the calendar",
        ),
        neu=("would be nice to have an export to spreadsheet", "is there a way to export events to a file?"),
        p_neu=0.35,
        rule=r"\b(export|exporting|csv|pdf)\b",
        effect_neg=-0.5,
        affinity={"nps": 1.1, "review": 0.9, "ticket": 0.9},
    ),
    Theme(
        "notifications_spam",
        "Notifications spam",
        1.0,
        0.0,
        neg=(
            "way too many notifications",
            "it sends notifications {when} for things I didn't ask for",
            "the reminders are spammy and I can't turn them off",
            "notification spam is driving me nuts",
            "push alerts at 6am for a promo",
        ),
        sarcasm=("great, another notification at 3am, love that", "thanks for the ten reminders about nothing"),
        rule=r"\b(notification|notifications|push alerts?|reminders? (?:are|is) spammy|spammy)\b",
        effect_neg=-0.8,
        affinity={"nps": 1.0, "review": 1.4, "ticket": 0.5},
    ),
    Theme(
        "app_stability",
        "Crashes and freezes",
        1.0,
        0.02,
        neg=(
            "the app crashes {when}",
            "it freezes when I open the {list}",
            "keeps crashing on {dev}",
            "app closes itself randomly",
            "it froze and I had to force quit",
        ),
        pos=("no crashes in two years of use",),
        sarcasm=("love how it crashes right when I need it",),
        rule=r"\b(crash|crashes|crashing|crashed|freezes|froze|force quit)\b",
        effect_neg=-2.2,
        effect_pos=0.2,
        affinity={"nps": 0.8, "review": 1.5, "ticket": 1.2},
    ),
    Theme(
        "battery_drain",
        "Battery drain",
        0.6,
        0.0,
        neg=(
            "it drains my battery",
            "the app eats battery in the background",
            "battery life tanks since I installed it",
            "uses 30 percent of my battery a day",
        ),
        rule=r"\b(battery|drains?)\b",
        effect_neg=-0.9,
        affinity={"nps": 0.6, "review": 1.7, "ticket": 0.4},
    ),
    Theme(
        "great_support",
        "Great support",
        0.8,
        0.97,
        neg=("support was polite but did not solve it",),
        pos=(
            "support was {very} helpful",
            "the support team fixed my issue in minutes",
            "customer service was kind and fast",
            "shout out to Maya in support",
            "support answered on a Sunday",
        ),
        rule=r"\b(support team|customer service|support was (?:very |really |super |so )?(?:helpful|great|kind))\b",
        effect_neg=-0.3,
        effect_pos=1.4,
        affinity={"nps": 1.2, "review": 0.8, "ticket": 0.6},
    ),
    Theme(
        "slow_support",
        "Slow support",
        0.7,
        0.0,
        neg=(
            "support never replied to my email",
            "waited a week for a response from support",
            "no answer from support after three tickets",
            "support takes days to respond",
            "the chat bot just loops",
        ),
        sarcasm=("great support, if you like waiting a week",),
        rule=r"\b(never replied|no answer from support|waited (?:a|two) weeks?|takes days to respond|chat bot)\b",
        effect_neg=-1.5,
        affinity={"nps": 1.1, "review": 0.8, "ticket": 1.3},
    ),
    Theme(
        "billing_errors",
        "Billing errors",
        0.7,
        0.0,
        neg=(
            "I was charged twice this month",
            "billed after I cancelled",
            "please refund the duplicate charge",
            "my card was charged for the annual plan without asking",
            "the receipt shows the wrong amount",
        ),
        rule=r"\b(charged twice|double charged|duplicate charge|refund|billed|receipt)\b",
        effect_neg=-2.4,
        affinity={"nps": 0.5, "review": 0.6, "ticket": 2.6},
    ),
    Theme(
        "easy_to_use",
        "Easy to use",
        1.3,
        0.9,
        neg=("it is confusing to use day to day", "too many taps to add a simple event"),
        pos=(
            "{very} easy to use",
            "simple and intuitive",
            "my kids figured it out in a minute",
            "adding events is quick and easy",
            "does exactly what I need without fuss",
            "straightforward to set up the {list}",
        ),
        rule=r"\b(easy to use|intuitive|simple to use|user friendly)\b",
        effect_neg=-0.7,
        effect_pos=1.2,
        affinity={"nps": 1.5, "review": 1.5, "ticket": 0.1},
    ),
    Theme(
        "search_broken",
        "Search broken",
        0.6,
        0.0,
        neg=(
            "search never finds my old notes",
            "search returns nothing even for exact titles",
            "searching the {list} is useless",
            "the search bar is broken",
        ),
        rule=r"\b(search|searching)\b",
        effect_neg=-0.9,
        affinity={"nps": 0.9, "review": 1.0, "ticket": 1.1},
    ),
    Theme(
        "offline_mode",
        "Offline mode",
        0.6,
        0.0,
        neg=(
            "it doesn't work without internet",
            "please add an offline mode",
            "on the plane I couldn't open anything",
            "nothing loads when I have no signal",
        ),
        neu=("does it have an offline mode?",),
        p_neu=0.2,
        rule=r"\b(offline|without internet|no signal|no wifi)\b",
        effect_neg=-0.4,
        affinity={"nps": 0.9, "review": 1.1, "ticket": 0.8},
    ),
    Theme(
        "dark_mode_request",
        "Dark mode request",
        0.5,
        0.0,
        neg=("the white screen hurts my eyes at night",),
        neu=(
            "please add a dark mode",
            "would love a dark theme",
            "dark mode would be nice",
            "any plans for dark mode?",
        ),
        p_neu=0.75,
        rule=r"\b(dark mode|dark theme)\b",
        effect_neg=-0.2,
        affinity={"nps": 1.0, "review": 1.3, "ticket": 0.4},
    ),
    Theme(
        "data_loss",
        "Data loss",
        0.5,
        0.0,
        neg=(
            "all my notes disappeared",
            "I lost a month of events",
            "the {list} vanished after a sync",
            "half my calendar is gone",
            "my data got wiped and support could not recover it",
        ),
        sarcasm=("thanks for deleting my whole calendar, very helpful",),
        rule=r"\b(disappeared|vanished|lost (?:all|a month|my)|wiped|is gone|deleting)\b",
        effect_neg=-3.2,
        affinity={"nps": 0.8, "review": 1.0, "ticket": 1.8},
    ),
    Theme(
        "ads_intrusive",
        "Intrusive ads",
        0.6,
        0.0,
        neg=(
            "too many ads in the free version",
            "full screen ads after every action",
            "ads cover the add button",
            "a video ad played while I was adding an event",
        ),
        sarcasm=("love watching an ad every time I add milk to the list",),
        rule=r"\b(ads?|advert|adverts|video ad)\b",
        effect_neg=-1.1,
        affinity={"nps": 0.8, "review": 1.8, "ticket": 0.3},
    ),
    Theme(
        "onboarding_confusing",
        "Confusing onboarding",
        0.6,
        0.0,
        neg=(
            "setup was confusing",
            "I didn't understand how to invite my family",
            "the tutorial skips the important parts",
            "took me ages to figure out how to share a calendar",
        ),
        rule=r"\b(setup|set up was|tutorial|onboarding|invite)\b",
        effect_neg=-0.6,
        affinity={"nps": 1.0, "review": 1.0, "ticket": 1.0},
    ),
    Theme(
        "collaboration",
        "Family sharing",
        1.0,
        0.75,
        neg=("sharing the {list} with my partner never works", "family members can't see shared events"),
        pos=(
            "sharing the {list} with the family is great",
            "the whole family uses it to stay organised",
            "shared lists keep our house running",
            "my partner and I finally plan together",
        ),
        rule=r"\b(shared|sharing|the (?:whole )?family|my partner)\b",
        effect_neg=-0.8,
        effect_pos=1.0,
        affinity={"nps": 1.4, "review": 1.1, "ticket": 0.5},
    ),
    Theme(
        "performance_fast",
        "Fast and responsive",
        0.8,
        0.85,
        neg=("the app is sluggish to open", "everything feels laggy since the redesign"),
        pos=(
            "the app is fast",
            "opens instantly",
            "{very} snappy and responsive",
            "quick to load even with a big {list}",
        ),
        rule=r"\b(fast|snappy|responsive|instantly|quick to load|sluggish|laggy)\b",
        effect_neg=-0.8,
        effect_pos=0.8,
        affinity={"nps": 1.1, "review": 1.3, "ticket": 0.2},
    ),
)

NEW_ISSUE = Theme(
    "widget_crash_v6",
    "Widget crash after 6.0",
    0.0,
    0.0,
    neg=(
        "since the 6.0 update the home screen widget crashes the app",
        "after updating to version 6 the widget is blank and tapping it crashes",
        "the new widget makes the whole app crash on launch since the update",
        "v6 update broke the widget, app closes when I open it",
        "home widget stopped working after the update and the app keeps quitting",
    ),
    effect_neg=-2.6,
)

THEME_NAMES: tuple[str, ...] = tuple(t.name for t in THEMES)
BY_NAME: dict[str, Theme] = {t.name: t for t in THEMES} | {NEW_ISSUE.name: NEW_ISSUE}
LABELS = {t.name: t.label for t in (*THEMES, NEW_ISSUE)}
SOURCES = ("nps", "review", "ticket")
SENTIMENTS = ("negative", "neutral", "positive")
POLARITY = {"negative": -1, "neutral": 0, "positive": 1}

# Implicit mentions with none of the obvious keywords. Prefix gives the sentiment: + positive,
# - negative, = neutral. These are what make keyword rules miss and the classifier work for it.
VAGUE: dict[str, tuple[str, ...]] = {
    "login_problems": (
        "-took me four tries to get in this morning",
        "-the verification code thing is a mess",
        "-keeps kicking me back to the start screen",
        "-locked out of my own account again",
    ),
    "slow_sync": (
        "-my partner's edits show up on my side the next day",
        "-the calendar on my phone and the web never match",
        "-updates from the kids' tablet lag way behind",
    ),
    "pricing_too_high": (
        "-not paying that every month",
        "-the free tier got gutted to push the upgrade",
        "-cheaper apps do the same thing",
        "+fair deal for what you get",
    ),
    "love_the_design": (
        "+really pleasant to look at",
        "+the icons and fonts are lovely",
        "+looks like a magazine",
        "-the colors are garish now",
    ),
    "missing_export": (
        "-I want my events in a spreadsheet for taxes",
        "=how do I get my notes out of here?",
        "-need to back up my lists to a file and can't",
    ),
    "notifications_spam": (
        "-my phone buzzes nonstop because of this app",
        "-pinged me about a sale during dinner",
        "-had to mute the app entirely",
    ),
    "app_stability": (
        "-it just quits on me",
        "-screen goes white and then I'm on the home screen",
        "-had to reinstall twice this week",
    ),
    "battery_drain": (
        "-phone is dead by lunch since I installed it",
        "-my phone gets hot with this running",
        "-background usage is off the charts",
    ),
    "great_support": (
        "+Maya sorted me out quickly",
        "+got a personal reply within the hour",
        "+the help team went above and beyond",
    ),
    "slow_support": (
        "-still waiting to hear back",
        "-emailed the help team twice, nothing",
        "-the help center goes in circles",
    ),
    "billing_errors": (
        "-two payments came out of my account",
        "-my bank shows a payment I never approved",
        "-they took money after I quit the plan",
    ),
    "easy_to_use": (
        "+my mom picked it up with no help",
        "+no learning curve at all",
        "+does what it says without making me think",
    ),
    "search_broken": (
        "-can't find anything I wrote last year",
        "-typing a title shows no results",
        "-finding an old note is hopeless",
    ),
    "offline_mode": ("-useless on the subway", "-on a flight it shows a spinner forever", "-needs wifi for everything"),
    "dark_mode_request": (
        "-at night the screen is blinding",
        "=an option for a black background would help",
        "-my eyes hurt using it in bed",
    ),
    "data_loss": (
        "-where did all my stuff go",
        "-a whole week of entries is just missing",
        "-my recipes are nowhere to be found",
    ),
    "ads_intrusive": (
        "-every other screen is a promo video",
        "-the banners are everywhere in the free plan",
        "-so many sponsored popups",
    ),
    "onboarding_confusing": (
        "-first day I had no idea what I was doing",
        "-the getting started guide made no sense",
        "-adding my kids to the account was a puzzle",
    ),
    "collaboration": (
        "+my husband and I are finally on the same page",
        "+everyone in the house can add to the shopping list",
        "+the kids check their chores on their own phones",
    ),
    "performance_fast": (
        "+opens before I finish blinking",
        "+zero lag even on my old phone",
        "-takes ages to open in the morning",
    ),
}

# Comments with no theme at all, only a general feeling.
GENERAL = (
    ("positive", "Love it!"),
    ("positive", "Five stars from me."),
    ("positive", "Best app ever."),
    ("positive", "Does the job."),
    ("neutral", "It's fine."),
    ("neutral", "Meh."),
    ("neutral", "No comment."),
    ("negative", "Worst app ever."),
    ("negative", "Not impressed."),
    ("negative", "Garbage."),
    ("positive", "Keep it up"),
    ("neutral", "ok"),
)
