"""Frozen BASE completion probes and scoring shared by bounded learning runs."""

import re
from decimal import Decimal

SCORING_VERSION = 8
NUMERIC_LITERAL = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
HAMLET_ANSWER = r'(?:(?:the )?English playwright(?: and poet)?,? )?(?:William )?Shakespeare\b'


def is_base_checkpoint(metadata, path=None):
    """Accept explicit BASE metadata or the older complete pretrain contract.

    Legacy checkpoints lacked stage. Require their pretrain filename together
    with corpus/sampler metadata; an explicit non-pretrain stage always wins.
    Model and tokenizer compatibility must still be validated by the caller.
    """
    from pathlib import Path
    if not isinstance(metadata, dict):
        return False
    if metadata.get('stage') is not None:
        return metadata['stage'] == 'pretrain'
    return bool(path and Path(path).name.startswith('pretrain_')
                and isinstance(metadata.get('config'), dict) and metadata['config']
                and isinstance(metadata.get('sampler'), dict) and metadata['sampler']
                and isinstance(metadata.get('processed_data_manifest_sha256'), str)
                and len(metadata['processed_data_manifest_sha256']) == 64
                and isinstance(metadata.get('tokens_processed'), int)
                and metadata['tokens_processed'] >= 0)


def suite():
    rows = []
    # (id, prompt, accepted-answer regex, canonical answer, plausible rivals). The
    # regex accepts valid variants in generations; the canonical answer and rivals
    # are the likelihood choices and the competing claims in a first clause.
    facts = [
        ('france', 'The capital of France is', r'Paris\b', 'Paris', ['Lyon', 'Marseille', 'Nice']),
        ('japan', 'The capital of Japan is', r'Tokyo\b', 'Tokyo', ['Osaka', 'Kyoto', 'Yokohama']),
        ('italy', 'The capital of Italy is', r'Rome\b', 'Rome', ['Milan', 'Venice', 'Naples', 'Florence']),
        ('germany', 'The capital of Germany is', r'Berlin\b', 'Berlin', ['Munich', 'Frankfurt', 'Hamburg', 'Bonn']),
        ('uk', 'The capital of the United Kingdom is', r'London\b', 'London', ['Manchester', 'Edinburgh', 'Birmingham']),
        ('australia', 'The capital of Australia is', r'Canberra\b', 'Canberra', ['Sydney', 'Melbourne', 'Brisbane']),
        ('water', 'The chemical formula for water is', r'H\s*2\s*O\b', 'H2O', ['CO2', 'H2O2', 'O2']),
        ('gold', 'The chemical symbol for gold is', r'Au\b', 'Au', ['Ag', 'Go', 'Gd', 'Pt']),
        ('oxygen', 'The chemical symbol for oxygen is', r'O\b', 'O', ['Ox', 'Os', 'N', 'H']),
        ('earth', 'The planet we live on is called', r'(?:the )?Earth\b', 'Earth', ['Mars', 'Venus', 'Gaia']),
        ('mars', 'The planet known as the Red Planet is', r'Mars\b', 'Mars', ['Venus', 'Jupiter', 'Mercury']),
        ('largest_planet', 'The largest planet in our solar system is', r'Jupiter\b', 'Jupiter', ['Saturn', 'Neptune', 'Earth']),
        ('moon', "Earth's natural satellite is the", r'Moon\b', 'Moon', ['Sun', 'Mars', 'Titan']),
        ('sun', 'The star at the center of our solar system is the', r'Sun\b', 'Sun', ['Moon', 'Earth', 'Sirius']),
        ('heart', 'The organ that pumps blood around the human body is the', r'heart\b', 'heart', ['lungs', 'brain', 'liver']),
        ('lungs', 'Humans breathe air using their', r'lungs\b', 'lungs', ['gills', 'mouths', 'noses']),
        ('photosynthesis', 'Plants convert sunlight into chemical energy through', r'photosynthesis\b',
         'photosynthesis', ['respiration', 'digestion', 'fermentation']),
        ('freeze', 'At standard atmospheric pressure, water freezes at',
         r'(?:0|zero)\s*(?:degrees?|°)?\s*(?:Celsius|C\b)', '0 degrees Celsius',
         ['100 degrees Celsius', '32 degrees Celsius', '10 degrees Celsius']),
        ('boil', 'At sea level, water boils at', r'(?:100|one hundred)\s*(?:degrees?|°)?\s*(?:Celsius|C\b)',
         '100 degrees Celsius', ['0 degrees Celsius', '212 degrees Celsius', '90 degrees Celsius']),
        ('week', 'There are seven days in a', r'week\b', 'week', ['month', 'year', 'day']),
        ('year', 'There are twelve months in a', r'year\b', 'year', ['month', 'week', 'decade']),
        ('triangle', 'A triangle has', r'(?:three|3)\s+(?:sides|vertices|angles)\b', 'three sides',
         ['four sides', 'two sides', 'five sides']),
        ('hamlet', 'The play Hamlet was written by', HAMLET_ANSWER, 'William Shakespeare',
         ['Christopher Marlowe', 'Charles Dickens', 'Ben Jonson']),
        ('pride', 'Pride and Prejudice was written by', r'Jane Austen\b', 'Jane Austen',
         ['Charlotte Bronte', 'Emily Bronte', 'Charles Dickens']),
    ]
    for name, prompt, expected, answer, rivals in facts:
        rows.append(dict(id='fact_'+name, category='facts', prompt=prompt, expected_regex=expected,
                         answer=answer, rivals=rivals, max_new_tokens=24))
    groups = [
        ('add', '2 + 3 = 5\n4 + 1 = 5\n', [(2,2),(7,5),(17,26),(38,47),(125,38),(9,8)], '+'),
        ('subtract', '8 - 3 = 5\n9 - 2 = 7\n', [(9,4),(15,7),(42,17),(100,36),(23,8),(71,29)], '-'),
        ('multiply', '2 * 3 = 6\n4 * 2 = 8\n', [(3,4),(6,7),(8,9),(12,5),(7,7),(11,11)], '*'),
        ('divide', '6 / 2 = 3\n8 / 4 = 2\n', [(10,2),(18,3),(24,6),(81,9),(100,10),(144,12)], '/'),
    ]
    for name, prefix, pairs, op in groups:
        for i,(a,b) in enumerate(pairs):
            answer = {'+':a+b, '-':a-b, '*':a*b, '/':a//b}[op]
            rows.append(dict(id=f'{name}_{i}',category='arithmetic',prompt=f'{prefix}{a} {op} {b} =',expected_regex=rf'{answer}(?:\.0+)?(?![\d.])',
                             answer=str(answer),max_new_tokens=16))
    # Sequence and code continuations: the next item is the claim, so these keep
    # start-of-completion scoring. Canonical answers and rivals feed likelihood.
    patterns = [
        ('code_len', '>>> len("cat")\n3\n>>> len("hello")\n', r'5(?!\d)', '5', ['4', '6', '3']),
        ('code_sum', '>>> sum([1, 2])\n3\n>>> sum([2, 3, 4])\n', r'9(?!\d)', '9', ['8', '10', '5']),
        ('code_upper', ">>> 'cat'.upper()\n'CAT'\n>>> 'dog'.upper()\n", r"['\"]DOG['\"]", "'DOG'",
         ["'dog'", "'Dog'", "'CAT'"]),
        ('code_index', '>>> [10, 20, 30][0]\n10\n>>> [10, 20, 30][1]\n', r'20(?!\d)', '20', ['10', '30', '2']),
        ('code_bool', '>>> 2 < 3\nTrue\n>>> 5 < 2\n', r'False\b', 'False', ['True', 'None']),
        ('code_add', 'def add(a, b):\n    return', r'a\s*\+\s*b\b', 'a + b', ['a - b', 'a * b', 'b + 1']),
        ('text_opposite', 'hot -> cold\nup -> down\nleft ->', r'right\b', 'right', ['left', 'down', 'up']),
        ('text_plural', 'one cat -> two cats\none dog -> two dogs\none book -> two', r'books\b', 'books',
         ['book', 'dogs', 'cats']),
        ('text_alphabet', 'A, B, C, D,', r'E\b', 'E', ['F', 'D', 'G']),
        ('text_count', '1, 2, 3, 4,', r'5(?!\d)', '5', ['6', '4', '10']),
        ('text_even', '2, 4, 6, 8,', r'10(?!\d)', '10', ['9', '12', '8']),
        ('text_days', 'Monday, Tuesday, Wednesday,', r'Thursday\b', 'Thursday', ['Friday', 'Wednesday', 'Saturday']),
    ]
    for name,prompt,expected,answer,rivals in patterns:
        rows.append(dict(id=name,category='code' if name.startswith('code') else 'text_patterns',prompt=prompt,
                         expected_regex=expected,answer=answer,rivals=rivals,max_new_tokens=24))
    continuations = [
        ('story', 'Mara found a small brass key under the kitchen table. She picked it up and noticed'),
        ('explanation', 'Rain forms when water vapor in the atmosphere'),
        ('garden', 'To grow tomatoes in a small garden, begin by'),
        ('history', 'The invention of the printing press changed the spread of knowledge because'),
        ('science', 'A battery stores chemical energy. When it is connected to a circuit,'),
        ('everyday', 'After missing the last bus, Daniel decided to walk home. Halfway down the road,'),
        ('code_long', 'def is_even(number):\n    """Return True if number is even, otherwise False."""\n    return'),
        ('exposition', 'Learning a new language takes practice. One useful habit is'),
    ]
    for name,prompt in continuations:
        rows.append(dict(id='continue_'+name,category='continuation',prompt=prompt,max_new_tokens=128))
    return rows


NUMERIC_CATEGORIES = {'arithmetic', 'arithmetic_equation', 'arithmetic_words', 'code_length',
                      'context_retrieve', 'counterfactual_retrieve', 'memorization',
                      'word_problem', 'equation', 'grounding'}
# The next item of a sequence or code line is its claim, so these keep
# start-of-completion scoring; all other answer categories score the first claim.
CONTINUATION_CATEGORIES = {'code', 'text_patterns'}
_NUMBER = re.compile(r'(?<![\w.])(' + NUMERIC_LITERAL + r')(?!\w|[.,]\d)')
_NEGATION = re.compile(r"\bnot\b|n't\b|\bnever\b", re.I)


def first_clause(text):
    """The completion up to its first newline, ';', or sentence-ending . ! ? (not a decimal point)."""
    text = text.lstrip()
    match = re.search(r'\n|;|[.!?](?!\d)', text)
    return text[:match.start()] if match else text


def _cut_off(text, clause, budget_reached):
    """Whether the clause runs to the last generated character of an unfinished generation."""
    return budget_reached is not False and len(clause) == len(text.lstrip())


def score(text, pattern):
    """Start-of-completion match, for sequence and code continuations."""
    return bool(re.match(r'^\s*'+pattern, text, flags=re.I))


def numeric_answer(row):
    """The numeric oracle for a numeric-category row."""
    target = row.get('answer')
    if target is None:
        match = re.match(r'^([+-]?\d+)', row.get('expected_regex', ''))
        if match is None:
            raise ValueError('Numeric probe requires an explicit numeric oracle')
        target = match[1]
    return Decimal(str(target).replace(',', ''))


def prompt_entities(prompt):
    """People whose codes the prompt states, for subject checks in grounding answers."""
    return re.findall(r'The badge code for (\w+) is', prompt)


def numeric_claim(text, *, budget_reached=None, entities=(), subject=None):
    """The first numeric claim of a completion and how it was made.

    Status is 'claimed', 'no_claim', 'hedged' (negated or offered with an
    alternative), 'other_subject' (asserted about another named person), or
    'undecided' (the generation stopped where the number might continue).
    A numeric equation's operands are skipped in favor of its first right-hand
    side, but a standalone number before a later equation remains the claim.
    """
    clause = first_clause(text)
    start = 0
    match = _NUMBER.search(clause)
    if match is not None and '=' in clause:
        equals = clause.index('=')
        # Skip operands only when the first number belongs to the equation's
        # left-hand side; prose between it and '=' preserves the earlier claim.
        if match.start() < equals and re.fullmatch(
                r'[\s()+*/%^×÷−-]*',
                re.sub(NUMERIC_LITERAL, '', clause[match.start():equals])):
            start = equals + 1
            match = _NUMBER.search(clause, start)
    if match is None:
        status = 'undecided' if _cut_off(text, clause, budget_reached) else 'no_claim'
        return {'status': status, 'value': None}
    lead, tail = clause[start:match.start()], clause[match.end():]
    value = Decimal(match[1].replace(',', ''))
    others = {e.lower() for e in entities} - ({subject.lower()} if subject else set())
    named = {w.lower() for w in re.findall(r'\w+', lead)}
    if others & named and not (subject and subject.lower() in named):
        status = 'other_subject'
    elif (_NEGATION.search(lead) or re.match(r'\s*(?:or|to|-|–)\s*' + NUMERIC_LITERAL, tail, re.I)
          or re.search(r'\bis not\b|\bisn.t\b', tail, re.I)):
        status = 'hedged'
    elif not tail and _cut_off(text, clause, budget_reached):
        status = 'undecided'
    else:
        status = 'claimed'
    return {'status': status, 'value': str(value)}


def choice_claim(text, pattern, rivals, *, budget_reached=None):
    """Which listed answer the first clause claims first.

    The correct answer (``pattern``) and each rival compete by position in the
    first clause. A negated or alternative-offering first claim is 'hedged'.
    """
    clause = first_clause(text)
    hits = []
    match = re.search(r'(?<!\w)(?:' + pattern + ')', clause, re.I)
    if match:
        hits.append((match.start(), match.end(), True))
    for rival in rivals:
        found = re.search(r'(?<!\w)' + re.escape(rival) + r'(?!\w)', clause, re.I)
        if found:
            hits.append((found.start(), found.end(), False))
    if not hits:
        status = 'undecided' if _cut_off(text, clause, budget_reached) else 'no_claim'
        return {'status': status, 'correct': False}
    begin, finish, correct = min(hits)
    if _NEGATION.search(clause[:begin]) or re.match(r'\s*,?\s*or\b', clause[finish:], re.I):
        return {'status': 'hedged', 'correct': False}
    return {'status': 'claimed', 'correct': correct}


def claim(text, row, *, budget_reached=None):
    """First-claim verdict for an answer row: {'status', 'correct', ...}."""
    category = row['category']
    if category in NUMERIC_CATEGORIES:
        answer = numeric_answer(row)
        entities = prompt_entities(row.get('prompt', '')) if category == 'grounding' else ()
        subject = None
        if entities:
            asked = re.search(r'What is the badge code for ([^?]+)\?', row['prompt'])
            subject = asked[1] if asked else None
        result = numeric_claim(text, budget_reached=budget_reached, entities=entities, subject=subject)
        result['correct'] = result['status'] == 'claimed' and Decimal(result['value']) == answer
        return result
    if category in CONTINUATION_CATEGORIES:
        correct = score(text, row['expected_regex'])
        return {'status': 'claimed' if correct else 'no_claim', 'correct': correct}
    pattern = row.get('expected_regex') or re.escape(row['answer']) + r'(?!\w)'
    return choice_claim(text, pattern, row.get('rivals', ()), budget_reached=budget_reached)


def score_completion(text, row, *, budget_reached=None):
    """Pass/fail: the completion's first claim is the correct answer."""
    return claim(text, row, budget_reached=budget_reached)['correct']


def likelihood_choices(row):
    """Likelihood candidates for a row, correct answer first, or None when not applicable.

    Rows with ``rivals`` use their canonical answer and rivals. Numeric rows rank
    the answer against nearby values and every number the prompt states, so
    copying an example or another person's value counts as a rival.
    """
    if 'rivals' in row:
        return [str(row['answer'])] + [str(r) for r in row['rivals']]
    if row['category'] not in NUMERIC_CATEGORIES:
        return None
    answer = numeric_answer(row)
    candidates = [answer + delta for delta in (-10, -2, -1, 1, 2, 10)]
    candidates += [Decimal(m.replace(',', '')) for m in _NUMBER.findall(row.get('prompt', ''))]
    rivals = []
    for value in candidates:
        if value != answer and (value >= 0 or answer < 0) and value not in rivals:
            rivals.append(value)
    return [str(v) for v in [answer] + rivals]


def repetition(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    return 1-len(set(grams))/len(grams) if grams else 0.0
