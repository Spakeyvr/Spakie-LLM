"""Frozen BASE completion probes and scoring shared by bounded learning runs."""

import re


def suite():
    rows = []
    facts = [
        ('france', 'The capital of France is', r'Paris\b'),
        ('japan', 'The capital of Japan is', r'Tokyo\b'),
        ('italy', 'The capital of Italy is', r'Rome\b'),
        ('germany', 'The capital of Germany is', r'Berlin\b'),
        ('uk', 'The capital of the United Kingdom is', r'London\b'),
        ('australia', 'The capital of Australia is', r'Canberra\b'),
        ('water', 'The chemical formula for water is', r'H\s*2\s*O\b'),
        ('gold', 'The chemical symbol for gold is', r'Au\b'),
        ('oxygen', 'The chemical symbol for oxygen is', r'O\b'),
        ('earth', 'The planet we live on is called', r'(?:the )?Earth\b'),
        ('mars', 'The planet known as the Red Planet is', r'Mars\b'),
        ('largest_planet', 'The largest planet in our solar system is', r'Jupiter\b'),
        ('moon', "Earth's natural satellite is the", r'Moon\b'),
        ('sun', 'The star at the center of our solar system is the', r'Sun\b'),
        ('heart', 'The organ that pumps blood around the human body is the', r'heart\b'),
        ('lungs', 'Humans breathe air using their', r'lungs\b'),
        ('photosynthesis', 'Plants convert sunlight into chemical energy through', r'photosynthesis\b'),
        ('freeze', 'At standard atmospheric pressure, water freezes at', r'(?:0|zero)\s*(?:degrees?|°)?\s*(?:Celsius|C\b)'),
        ('boil', 'At sea level, water boils at', r'(?:100|one hundred)\s*(?:degrees?|°)?\s*(?:Celsius|C\b)'),
        ('week', 'There are seven days in a', r'week\b'),
        ('year', 'There are twelve months in a', r'year\b'),
        ('triangle', 'A triangle has', r'(?:three|3)\s+(?:sides|vertices|angles)\b'),
        ('hamlet', 'The play Hamlet was written by', r'(?:William )?Shakespeare\b'),
        ('pride', 'Pride and Prejudice was written by', r'Jane Austen\b'),
    ]
    for name, prompt, expected in facts:
        rows.append(dict(id='fact_'+name, category='facts', prompt=prompt, expected_regex=expected, max_new_tokens=24))
    groups = [
        ('add', '2 + 3 = 5\n4 + 1 = 5\n', [(2,2),(7,5),(17,26),(38,47),(125,38),(9,8)], '+'),
        ('subtract', '8 - 3 = 5\n9 - 2 = 7\n', [(9,4),(15,7),(42,17),(100,36),(23,8),(71,29)], '-'),
        ('multiply', '2 * 3 = 6\n4 * 2 = 8\n', [(3,4),(6,7),(8,9),(12,5),(7,7),(11,11)], '*'),
        ('divide', '6 / 2 = 3\n8 / 4 = 2\n', [(10,2),(18,3),(24,6),(81,9),(100,10),(144,12)], '/'),
    ]
    for name, prefix, pairs, op in groups:
        for i,(a,b) in enumerate(pairs):
            answer = {'+':a+b, '-':a-b, '*':a*b, '/':a//b}[op]
            rows.append(dict(id=f'{name}_{i}',category='arithmetic',prompt=f'{prefix}{a} {op} {b} =',expected_regex=rf'{answer}(?:\.0+)?(?![\d.])',max_new_tokens=16))
    patterns = [
        ('code_len', '>>> len("cat")\n3\n>>> len("hello")\n', r'5(?!\d)'),
        ('code_sum', '>>> sum([1, 2])\n3\n>>> sum([2, 3, 4])\n', r'9(?!\d)'),
        ('code_upper', ">>> 'cat'.upper()\n'CAT'\n>>> 'dog'.upper()\n", r"['\"]DOG['\"]"),
        ('code_index', '>>> [10, 20, 30][0]\n10\n>>> [10, 20, 30][1]\n', r'20(?!\d)'),
        ('code_bool', '>>> 2 < 3\nTrue\n>>> 5 < 2\n', r'False\b'),
        ('code_add', 'def add(a, b):\n    return', r'a\s*\+\s*b\b'),
        ('text_opposite', 'hot -> cold\nup -> down\nleft ->', r'right\b'),
        ('text_plural', 'one cat -> two cats\none dog -> two dogs\none book -> two', r'books\b'),
        ('text_alphabet', 'A, B, C, D,', r'E\b'),
        ('text_count', '1, 2, 3, 4,', r'5(?!\d)'),
        ('text_even', '2, 4, 6, 8,', r'10(?!\d)'),
        ('text_days', 'Monday, Tuesday, Wednesday,', r'Thursday\b'),
    ]
    for name,prompt,expected in patterns:
        rows.append(dict(id=name,category='code' if name.startswith('code') else 'text_patterns',prompt=prompt,expected_regex=expected,max_new_tokens=24))
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


def score(text, pattern):
    # Only the immediate answer counts: no credit for answers appearing later.
    return bool(re.match(r'^\s*'+pattern, text, flags=re.I))


def repetition(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
    return 1-len(set(grams))/len(grams) if grams else 0.0
