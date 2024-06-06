import itertools

strings = ['age', 'gender', 'chest_pain']

def get_comb(smh):
    combination = []

    for r in range(1, len(smh) + 1):
        print(sorted(itertools.combinations(smh, r)))
        combination.extend(itertools.combinations(smh, r))
    
    yield from combination
