"""A deterministic synthetic court system. No court, judge, party or case here is real: the districts and names are
invented, and every motion's outcome is drawn from a hidden model (judge tendencies, motion type, case features), so
the analytics can be scored against the truth. Nothing in this file should be read as describing real litigation."""
import datetime
import functools
import math
import random

COURTS = ["D. North Calder", "D. South Calder", "E.D. Marrow", "W.D. Marrow"]
SURNAMES = ["Halvorsen", "Okonkwo", "Baptiste", "Lindqvist", "Marchetti", "Nakamura", "Oyelaran", "Petrov", "Quinlan", "Rasool", "Sandoval", "Thackeray", "Ueda", "Varga", "Whitlock", "Yilmaz", "Zamora", "Ashworth",
            "Brennan", "Castellanos", "Dunmore", "Ekström", "Fairweather", "Gallardo", "Hargreave", "Ibekwe", "Jaramillo", "Kowalczyk", "Lachance", "Montague", "Northcott", "Obuya", "Pemberton", "Rahimi", "Stavros", "Trevino"]
MOTIONS = {"dismiss": -0.4, "summary_judgment": -0.2, "compel": 0.6, "preliminary_injunction": -1.2}        # base log-odds of being granted
MOTION_NAME = {"dismiss": "Motion to dismiss", "summary_judgment": "Motion for summary judgment", "compel": "Motion to compel discovery", "preliminary_injunction": "Motion for preliminary injunction"}
# nature of suit -> fact patterns -> sentences. Two sentences of a case's facts come from its pattern, one from the suit's shared pool.
PATTERNS = {
    "employment": {
        "retaliation": ["Plaintiff alleges she was terminated weeks after reporting safety violations to a regulator.", "The complaint describes a demotion following an internal complaint about unsafe equipment.", "Plaintiff reported the hazard in writing and was dismissed within the month.", "Supervisors allegedly referred to the safety report when announcing the termination."],
        "promotion": ["Plaintiff alleges he was passed over for promotion in favour of a less experienced colleague.", "The complaint cites three promotion cycles in which plaintiff was not selected.", "Plaintiff claims the selection criteria were changed after he applied.", "A comparator with fewer years of service was promoted instead."],
        "wage": ["Plaintiffs allege unpaid overtime for work performed before and after scheduled shifts.", "The complaint describes automatic meal-break deductions for breaks that were not taken.", "Time records allegedly were edited to remove hours over forty.", "The class consists of hourly warehouse staff paid a flat weekly amount."],
        "harassment": ["Plaintiff alleges a pattern of hostile comments by a direct supervisor.", "The complaint describes repeated unwelcome remarks at team meetings.", "Human resources allegedly took no action after two complaints.", "Plaintiff transferred departments to avoid the supervisor."],
        "leave": ["Plaintiff alleges her position was eliminated while she was on approved medical leave.", "The complaint describes a refusal to reinstate plaintiff after surgery.", "Leave paperwork was allegedly approved and then questioned.", "Plaintiff returned from leave to a lower-paid role."]},
    "contract": {
        "nonpayment": ["Plaintiff supplier alleges defendant failed to pay invoices for delivered goods.", "The complaint lists eleven unpaid invoices over two quarters.", "Defendant accepted the shipments without objection.", "A payment plan was allegedly agreed and then abandoned."],
        "defect": ["Plaintiff alleges the delivered machinery failed to meet the agreed specification.", "The complaint describes repeated breakdowns within the warranty period.", "Defendant's technicians allegedly could not repair the fault.", "An independent inspection found the tolerance out of range."],
        "termination": ["Plaintiff alleges defendant terminated the distribution agreement without the required notice.", "The complaint cites a ninety-day notice clause that was not observed.", "Defendant appointed a replacement distributor the following week.", "Plaintiff had invested in inventory in reliance on the term."],
        "noncompete": ["Plaintiff employer alleges a former executive joined a competitor in breach of a restrictive covenant.", "The complaint describes solicitation of three key clients.", "The covenant runs for twelve months across the region.", "Defendant allegedly took a client list on departure."],
        "license": ["Plaintiff alleges defendant used licensed software beyond the permitted number of seats.", "The complaint cites an audit showing use at unlicensed sites.", "Defendant disputes how the licence counts users.", "Licence fees for the excess use were demanded and refused."]},
    "civil_rights": {
        "force": ["Plaintiff alleges officers used excessive force during a traffic stop.", "The complaint describes injuries requiring hospital treatment.", "Body-camera footage allegedly contradicts the incident report.", "Plaintiff was not charged with any offence."],
        "search": ["Plaintiff alleges his home was searched without a valid warrant.", "The complaint asserts the warrant named a different address.", "Property seized in the search was not returned.", "Officers allegedly entered before announcing themselves."],
        "detention": ["Plaintiff alleges he was held for nine days without a hearing.", "The complaint describes detention beyond the period the statute allows.", "Requests to contact counsel were allegedly ignored.", "The charge was later withdrawn."],
        "speech": ["Plaintiff alleges her permit to hold a public rally was denied because of its message.", "The complaint describes comparable permits granted to other groups.", "The city cited a policy adopted after the application.", "Plaintiff was removed from a public meeting after speaking."],
        "conditions": ["Plaintiff alleges he was denied necessary medical care while in custody.", "The complaint describes weeks without prescribed medication.", "Grievances filed through the facility process went unanswered.", "A specialist appointment was cancelled three times."]},
    "patent": {
        "software": ["Plaintiff alleges defendant's application infringes a patent on a data-synchronisation method.", "The complaint maps each claim element to the accused product.", "Defendant argues the claims are directed to an abstract idea.", "The accused feature was released after the patent issued."],
        "device": ["Plaintiff alleges defendant's valve assembly infringes a patent on a sealing mechanism.", "The complaint includes teardown photographs of the accused device.", "Defendant contends its design uses a different seal.", "Plaintiff sold a competing product for six years."],
        "pharma": ["Plaintiff alleges defendant's proposed generic infringes a formulation patent.", "The complaint follows a regulatory filing by defendant.", "Defendant asserts the claimed ratio was obvious.", "The patent expires in four years."],
        "design": ["Plaintiff alleges defendant's packaging copies a patented ornamental design.", "The complaint presents side-by-side images of the two products.", "Defendant points to earlier designs with the same outline.", "Retail buyers allegedly confused the products."],
        "standard": ["Plaintiff alleges defendant's handsets practise a standard-essential patent without a licence.", "The complaint describes failed licensing negotiations.", "Defendant argues the offered rate was not reasonable.", "The standard was adopted before the accused sales."]},
    "injury": {
        "premises": ["Plaintiff alleges she slipped on an unmarked wet floor in defendant's store.", "The complaint asserts no warning sign was placed.", "An employee had mopped the aisle minutes earlier.", "Plaintiff fractured her wrist in the fall."],
        "product": ["Plaintiff alleges a space heater sold by defendant overheated and caused a fire.", "The complaint cites earlier consumer reports of overheating.", "The unit lacked an automatic shut-off.", "Plaintiff suffered burns and property damage."],
        "vehicle": ["Plaintiff alleges defendant's delivery driver ran a red light and struck her car.", "The complaint asserts the driver was on a timed route.", "A witness saw the van enter the junction on red.", "Plaintiff missed four months of work."],
        "medical": ["Plaintiff alleges a surgical instrument was left in place after an operation.", "The complaint describes a second operation to remove it.", "The count sheet for the procedure was incomplete.", "Plaintiff developed an infection."],
        "workplace": ["Plaintiff contractor alleges he fell from scaffolding erected by defendant.", "The complaint asserts guard rails were missing.", "The site had been cited for the same hazard.", "Plaintiff suffered a spinal injury."]},
    "securities": {
        "disclosure": ["Plaintiffs allege the company concealed a regulatory investigation from investors.", "The complaint cites statements that operations were fully compliant.", "The share price fell by a third on disclosure.", "Executives sold shares during the period."],
        "accounting": ["Plaintiffs allege revenue was recognised before delivery to inflate quarterly results.", "The complaint relies on a later restatement.", "A former controller is quoted describing the practice.", "Auditors resigned before the restatement."],
        "forecast": ["Plaintiffs allege management issued forecasts it knew could not be met.", "The complaint cites internal projections below public guidance.", "Guidance was withdrawn two months later.", "Analysts had relied on the guidance."],
        "offering": ["Plaintiffs allege the offering documents omitted the loss of a major customer.", "The complaint asserts the customer gave notice before the offering.", "The customer accounted for a fifth of revenue.", "Shares fell below the offer price within weeks."],
        "insider": ["Plaintiffs allege officers traded on undisclosed trial results.", "The complaint lists sales in the week before the announcement.", "No trading plan covered the sales.", "The results were known internally by then."]}}
SHARED = {"employment": ["Plaintiff exhausted administrative remedies before filing.", "Defendant employs more than five hundred people in the district.", "Plaintiff seeks back pay and reinstatement."],
          "contract": ["The agreement contains a choice-of-law clause.", "The parties had dealt with each other for several years.", "Plaintiff seeks damages and interest."],
          "civil_rights": ["Defendants are sued in their individual and official capacities.", "Plaintiff filed a notice of claim with the municipality.", "Plaintiff seeks compensatory and punitive damages."],
          "patent": ["Plaintiff is the assignee of the patent in suit.", "Defendant was notified of the patent by letter.", "Plaintiff seeks an injunction and a reasonable royalty."],
          "injury": ["Plaintiff incurred medical expenses and lost earnings.", "Defendant's insurer was notified of the claim.", "Plaintiff seeks compensatory damages."],
          "securities": ["The action is brought on behalf of purchasers during the class period.", "Lead plaintiff was appointed by the court.", "Plaintiffs seek damages for the class."]}
STRONG = ["The key events are documented in contemporaneous written records.", "Plaintiff attaches correspondence supporting each allegation."]
WEAK = ["The complaint identifies no written record of the key events.", "The allegations rest on plaintiff's recollection alone."]
TREATMENT = {
    "followed": ["The court follows {r} and reaches the same result.", "As in {r}, the allegations are sufficient at this stage.", "{r} controls here.", "The reasoning of {r} applies with equal force.", "Consistent with {r}, the motion is resolved the same way.", "The court adopts the analysis in {r}."],
    "distinguished": ["{r} is distinguishable because the record there was fuller.", "Unlike {r}, the plaintiff here pleaded the missing element.", "The court does not follow {r}, which involved a different statute.", "{r} does not control: the facts differ in a material respect.", "Defendant's reliance on {r} is misplaced on these facts.", "{r} turned on a concession not made here."],
    "criticized": ["The court respectfully disagrees with the reasoning of {r}.", "{r} has been questioned and the court declines to extend it.", "The analysis in {r} is unpersuasive.", "{r} rests on a reading the court cannot follow.", "Later decisions have cast doubt on {r}."],
    "overruled": ["{r} is no longer good law after the appellate decision last term.", "{r} was abrogated and cannot be followed.", "To the extent {r} holds otherwise, it has been overruled.", "{r} did not survive the intervening change in the governing standard."]}
DATA_CUT = datetime.date(2024, 12, 31)
SPLIT = datetime.date(2023, 1, 1)                 # models learn from motions decided before this and are scored on the rest
SUIT_SCALE = {"employment": 520, "contract": 420, "civil_rights": 600, "patent": 780, "injury": 480, "securities": 700}        # typical days to resolution


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@functools.lru_cache(maxsize=4)
def build(seed=9, n_cases=9000):
    r = random.Random(seed)
    names = r.sample(SURNAMES, len(SURNAMES))
    judges = []
    for i, n in enumerate(names):
        new = i >= len(names) - 4                               # four recent appointees with almost no record
        judges.append({"name": f"Hon. {r.choice('ABCDEFGHJKLMNPRSTW')}. {n}", "court": COURTS[i % 4], "appointed": datetime.date(2024 if new else r.randint(1998, 2014), r.randint(1, 12), 1),
                       "lean": r.gauss(0, 0.7), "by_motion": {m: r.gauss(0, 0.35) for m in MOTIONS}, "speed": math.exp(r.gauss(0, 0.25))})
    suits = list(PATTERNS)
    suit_fx = {(s, m): r.gauss(0, 0.3) for s in suits for m in MOTIONS}
    pat_fx = {(s, p, m): r.gauss(0, 0.4) for s in suits for p in PATTERNS[s] for m in MOTIONS}
    cases, motions, cites = [], [], []
    by_pattern, by_suit = {}, {}
    start = datetime.date(2015, 1, 1)
    for k in range(n_cases):
        filed = start + datetime.timedelta(days=int((DATA_CUT - start).days * k / n_cases))
        elig = [j for j, jd in enumerate(judges) if jd["appointed"] <= filed]
        j = r.choice(elig)
        suit = r.choice(suits)
        pat = r.choice(list(PATTERNS[suit]))
        strength = r.gauss(0, 1)
        documented = strength + r.gauss(0, 0.8) > 0
        facts = " ".join(r.sample(PATTERNS[suit][pat], 2) + [r.choice(SHARED[suit]), r.choice(STRONG if documented else WEAK) if r.random() < 0.8 else r.choice(SHARED[suit])])
        feat = {"pro_se": r.random() < 0.15, "amended": r.choice([0, 0, 0, 1, 1, 2]), "claims": r.randint(1, 8), "large_firm_defence": r.random() < 0.35}
        days = int(SUIT_SCALE[suit] * judges[j]["speed"] * r.weibullvariate(1, 1.4))
        closed = filed + datetime.timedelta(days=max(30, days))
        cases.append({"court": judges[j]["court"], "docket": f"{filed.year % 100}-cv-{k % 9000 + 1000:05d}", "filed": filed, "closed": closed if closed <= DATA_CUT else None, "true_days": max(30, days), "suit": suit, "pattern": pat,
                      "judge": j, "facts": facts, "strength": strength, **feat})
        for m in r.sample(list(MOTIONS), r.choice([0, 1, 1, 2, 2, 3])):
            mfiled = filed + datetime.timedelta(days=r.randint(40, 300))
            decided = mfiled + datetime.timedelta(days=int(r.lognormvariate(4.3, 0.5) * judges[j]["speed"]))
            if decided > DATA_CUT:
                continue
            defence = m in ("dismiss", "summary_judgment")       # the defendant moves; strong, documented claims resist
            z = (MOTIONS[m] + judges[j]["lean"] + judges[j]["by_motion"][m] + suit_fx[suit, m] + pat_fx[suit, pat, m] + (0.6 * feat["pro_se"] + 0.35 * feat["amended"] - 0.5 * strength + 0.3 * feat["large_firm_defence"] if defence else 0.4 * strength))
            p = sigmoid(z)
            motions.append({"case": k, "type": m, "filed": mfiled, "decided": decided, "granted": r.random() < p, "p_true": p})
            pool = by_pattern.get((suit, pat), [])[-60:] * 8 + by_suit.get(suit, [])[-60:] * 3 + [r.randrange(k) for _ in range(3)] if k else []
            for c in dict.fromkeys(r.choices(pool, k=r.randint(2, 5))) if pool else []:
                t = r.choices(list(TREATMENT), [62, 28, 8, 2])[0]
                ref = f"{cases[c]['docket']} ({cases[c]['court']})"
                cites.append({"motion": len(motions) - 1, "cited": c, "treatment": t, "context": r.choice(TREATMENT[t]).format(r=ref)})
        by_pattern.setdefault((suit, pat), []).append(k)
        by_suit.setdefault(suit, []).append(k)
    return {"seed": seed, "judges": judges, "cases": cases, "motions": motions, "cites": cites}


def contexts(seed, n=4000):
    """Labelled citation sentences for fitting and scoring the treatment classifier."""
    r = random.Random(seed)
    return [(r.choice(TREATMENT[t]).format(r=f"{r.randint(15, 24)}-cv-{r.randint(1000, 9999):05d} ({r.choice(COURTS)})"), t) for t in (r.choice(list(TREATMENT)) for _ in range(n))]
