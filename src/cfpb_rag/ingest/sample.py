"""Synthetic complaint generator with the same schema as the real CFPB CSV.

This exists ONLY so the pipeline, tests and CI can run without downloading the
real dataset. Company names are fictional and the narratives are templated.
Never report evaluation numbers from this data as if they were real results.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

COMPANIES = {
    "Mortgage": [
        ("Northgate Home Lending", "escrow"),
        ("Harborview Servicing", "portal"),
        ("Cedar Ridge Mortgage", "forbearance"),
        ("Blue Finch Loans", "appraisal"),
        ("Summit Path Financial", "payoff"),
    ],
    "Credit reporting": [
        ("Orion Credit Bureau", "dispute"),
        ("Meridian Data Services", "inquiry"),
        ("Atlas Consumer Reports", "identity"),
        ("Pinecrest Credit Info", "tradeline"),
    ],
}

SIGNATURE = {
    "escrow": "The escrow analysis kept changing and nobody could explain the shortage.",
    "portal": "The online portal was down for days and I could not see my balance.",
    "forbearance": "My forbearance plan was not honored on the account.",
    "appraisal": "The appraisal came back far below the contract price.",
    "payoff": "The payoff statement had a different number every time I called.",
    "dispute": "My dispute letter was marked as frivolous without any review.",
    "inquiry": "A hard inquiry appeared that I never authorized.",
    "identity": "An account opened with my stolen identity still shows on my file.",
    "tradeline": "A closed tradeline keeps showing as open with a balance.",
}

ISSUES = {
    "Mortgage": {
        "Trouble during payment process": [
            "I made my mortgage payment of ${amount} in {month} but {company} applied it late and charged a late fee.",
            "{company} returned my payment of ${amount} and said the account number was wrong even though I used the same one for years.",
            "My automatic mortgage payment was drafted twice in {month} and {company} took {days} days to refund the extra ${amount}.",
            "I could not get {company} to tell me where my ${amount} payment went for {days} days.",
        ],
        "Struggling to pay mortgage": [
            "I lost hours at work in {month} and asked {company} about hardship options but they only offered a plan I cannot afford.",
            "{company} denied my loan modification request twice and I am now {days} days behind.",
            "I requested payment relief from {company} after a medical bill of ${amount} and received no response.",
        ],
        "Closing on a mortgage": [
            "The closing was delayed {days} days because {company} kept requesting the same documents.",
            "{company} changed my closing costs by ${amount} the day before closing.",
            "My rate lock with {company} expired in {month} because of their processing delays.",
        ],
        "Applying for a mortgage or refinancing an existing mortgage": [
            "I applied to refinance with {company} and they lost my paystubs three times over {days} days.",
            "{company} denied my refinance without a clear reason after I paid ${amount} in application fees.",
            "The loan officer at {company} quoted one rate and the written offer in {month} was higher.",
        ],
    },
    "Credit reporting": {
        "Incorrect information on your report": [
            "My credit report from {company} lists a ${amount} collection account that does not belong to me.",
            "{company} shows a late payment from {month} but I have proof it was paid on time.",
            "An address I never lived at appears on my report from {company}.",
        ],
        "Problem with a credit reporting company's investigation into an existing problem": [
            "I disputed an error with {company} and after {days} days they said it was verified without sending any proof.",
            "{company} closed my dispute in {month} and did not remove the ${amount} account that I proved was fraudulent.",
            "I sent documents to {company} twice and both disputes were rejected as duplicates.",
        ],
        "Improper use of your report": [
            "{company} sent my report to a company I never applied with.",
            "A landlord pulled my credit through {company} in {month} without my permission.",
        ],
        "Unable to get your credit report or credit score": [
            "I requested my free annual report from {company} and received nothing after {days} days.",
            "{company} would not let me access my score because their identity check kept failing.",
        ],
    },
}

CLOSERS = [
    "I have contacted them several times without a resolution.",
    "I would like this investigated and corrected.",
    "This has damaged my finances and caused a lot of stress.",
    "I have all of the documents if needed.",
    "No one has given me a straight answer.",
]
CONTEXT = [
    "I have been a customer for {years} years.",
    "I spoke with at least {calls} different representatives.",
    "I am a single parent and this is a serious burden.",
    "I work full time and cannot take more days off to deal with this.",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
STATES = ["CA", "TX", "FL", "NY", "IL", "GA", "NC", "PA", "OH", "MI", "VA", "AZ"]
RESPONSES = ["Closed with explanation", "Closed with monetary relief",
             "Closed with non-monetary relief", "In progress"]

HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Consumer consent provided?", "Submitted via",
    "Date sent to company", "Company response to consumer", "Timely response?",
    "Consumer disputed?", "Complaint ID",
]


def _narrative(rng: random.Random, group: str, issue: str, company: str, quirk: str) -> str:
    pool = ISSUES[group][issue]
    picks = rng.sample(pool, k=min(2, len(pool)))
    fmt = dict(
        company=company,
        amount=rng.choice([85, 240, 410, 1200, 2750, 5400]),
        month=rng.choice(MONTHS),
        days=rng.choice([7, 14, 30, 45, 60, 90]),
        years=rng.randint(1, 15),
        calls=rng.randint(2, 7),
    )
    parts = [s.format(**fmt) for s in picks]
    if rng.random() < 0.6:
        parts.append(SIGNATURE[quirk])
    parts.append(rng.choice(CONTEXT).format(**fmt))
    if rng.random() < 0.5:
        parts.append("On XX/XX/2025 I called XXXX and spoke with XXXX about it.")
    parts.append(rng.choice(CLOSERS))
    return " ".join(parts)


def generate_sample(out_csv: Path, n: int = 3000, seed: int = 0) -> Path:
    """Write a synthetic CSV (including messy rows that exercise the cleaner)."""
    rng = random.Random(seed)
    start, end = date(2023, 6, 1), date(2026, 10, 15)
    span = (end - start).days
    rows = []
    for i in range(n):
        group = rng.choice(list(COMPANIES))
        company, quirk = rng.choice(COMPANIES[group])
        issue = rng.choice(list(ISSUES[group]))
        d = start + timedelta(days=rng.randint(0, span))
        product = "Mortgage" if group == "Mortgage" else "Credit reporting or other personal consumer reports"
        narrative = _narrative(rng, group, issue, company, quirk)
        roll = rng.random()
        if roll < 0.05:
            narrative = ""                                   # no narrative
        elif roll < 0.09:
            narrative = "Bad service."                       # too short
        elif roll < 0.14:
            product, issue = "Student loan", "Dealing with your lender or servicer"
            narrative = "My student loan servicer keeps misapplying payments to the wrong loan. " * 3
        row = {h: "" for h in HEADER}
        row.update({
            "Date received": d.isoformat(), "Product": product, "Issue": issue,
            "Consumer complaint narrative": narrative, "Company": company,
            "State": rng.choice(STATES), "Submitted via": "Web",
            "Company response to consumer": rng.choice(RESPONSES),
            "Complaint ID": str(7_000_000 + i),
        })
        rows.append(row)

    # messy rows: duplicate IDs and copy-pasted narratives under new IDs
    for j in range(max(1, n // 50)):
        rows.append(dict(rng.choice(rows)))
    for j in range(max(1, n // 50)):
        clone = dict(rng.choice(rows))
        clone["Complaint ID"] = str(9_000_000 + j)
        rows.append(clone)

    df = pd.DataFrame(rows, columns=HEADER)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
